"""Source-only calibration robustness and exact paired scheduled-view replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from domainbed.scripts.train_subset_irm_pacs import make_schedule
from .centered_model import CenteredGateSIRM
from .hard_router_diagnosis import outputs
from .bias_control import scores
from .shrinkage import ROOT, digest, write


def domain_mean(x, domains, mask=None):
    if mask is None: mask = np.ones(len(domains), bool)
    return np.stack([x[mask & (domains == d)].mean(0)
                     for d in sorted(set(domains[mask].tolist()))]).mean(0)


def evaluate(g, correction, y, domains, bias):
    logits = g + correction - bias
    score, pred = scores(logits, y, domains)
    base = g.argmax(1)
    score.update(repairs=int(((base != y) & (pred == y)).sum()),
                 harms=int(((base == y) & (pred != y)).sum()))
    return score


def main():
    torch.set_num_threads(4)
    dest = ROOT / 'results/sirm_research_loop/iteration_11'
    out = dest / 'composed_bias_source_diagnosis'; out.mkdir(exist_ok=True)
    path = dest / 'experiment/runs/L38/OFF/source_best.pkl'
    manifest_path = dest / 'experiment/subset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['selected_step'] == 399
    c = payload['config']; backbone = DeiTFeaturizer(pretrained=False, model_name=c['backbone'])
    model = CenteredGateSIRM(backbone, 384, 10, 3, c)
    model.load_state_dict({k.removeprefix('model.'): v for k,v in payload['model_dict'].items()})
    model.configure_stage(); model.requires_grad_(False); model.eval()
    caches = ROOT / 'results/sirm_research_loop/iteration_09/source_router_diagnosis'
    sources, arrays, ids_by_split = {}, {}, {}
    plan = dict(code_sha256=digest(Path(__file__)), checkpoint_sha256=digest(path),
        manifest_sha256=digest(manifest_path), target_read=False, fitting=False,
        scope='Saved399 state only. Source-in mean calibration; source-out validation and '
              'six exact paired source-in H2/original batches. No historical gradients, '
              'independent validation, target, optimizer, hyperparameter or selection changes.',
        source_cache_hashes={s:digest(caches/f'{s}_original.npz') for s in ['source_in','source_out']})
    with torch.no_grad():
        for split in ['source_in','source_out']:
            with np.load(caches/f'{split}_original.npz') as a:
                z,y,d = [torch.from_numpy(a[k]) for k in ['z','y','domains']]
                ids = a['ids'].tolist()
            assert set(d.tolist()) == {0,2,3}
            f = outputs(model,z,y); corr=f['full']-f['g']
            sources[split]=(z,y,d,ids)
            arrays[split] = (f['g'].numpy(),corr.numpy(),y.numpy(),d.numpy())
            ids_by_split[split]=set(ids)
    assert not ids_by_split['source_in'] & ids_by_split['source_out']
    frozen_cache=torch.load(dest/'source_feature_cache.pt',map_location='cpu',weights_only=False)
    assert sources['source_in'][3]==frozen_cache['ids']
    assert torch.equal(sources['source_in'][0],frozen_cache['features'])
    old_payload=torch.load(ROOT/'results/sirm_research_loop/iteration_08/experiment/runs/L38/OFF/last.pkl',
                           map_location='cpu',weights_only=False)['model_dict']
    old_payload={k.removeprefix('model.'):v for k,v in old_payload.items()}
    for name in ['featurizer','global_head']:
        assert all(torch.equal(v,old_payload[name+'.'+k])
                   for k,v in getattr(model,name).state_dict().items())
    g, corr, y, domains = arrays['source_in']
    fold = np.zeros(len(y),int)
    for d in [0,2,3]:
        for cls in sorted(set(y[domains==d].tolist())):
            ii=np.where((domains==d)&(y==cls))[0]
            ii=sorted(ii,key=lambda i:hashlib.sha256(sources['source_in'][3][i].encode()).hexdigest())
            fold[ii]=np.arange(len(ii))%2
    biases={'all_source_in':domain_mean(corr,domains),
            'half0_source_in':domain_mean(corr,domains,fold==0),
            'half1_source_in':domain_mean(corr,domains,fold==1)}
    original_lock=json.loads((dest/'bias_control/source_bias_lock.json').read_text())
    assert np.allclose(biases['all_source_in'],original_lock['vector'],atol=1e-5)
    sg, sc, sy, sd = arrays['source_out']
    calibration={k:evaluate(sg,sc,sy,sd,b) for k,b in biases.items()}
    baseline=evaluate(sg,sc,sy,sd,np.zeros(10))
    erm=evaluate(sg,np.zeros_like(sc),sy,sd,np.zeros(10))
    held=[]
    for d in [0,2,3]:
        bias=domain_mean(corr,domains,domains!=d); mask=sd==d
        held.append(dict(held_domain=d,calibration_domains=[a for a in [0,2,3] if a!=d],
            raw=evaluate(sg[mask],sc[mask],sy[mask],sd[mask],np.zeros(10)),
            demeaned=evaluate(sg[mask],sc[mask],sy[mask],sd[mask],bias),
            global_only=evaluate(sg[mask],np.zeros_like(sc[mask]),sy[mask],sd[mask],np.zeros(10))))
    feature_lookup={image:z for image,z in zip(sources['source_in'][3],sources['source_in'][0])}
    lock=json.loads((dest/'experiment/runs/L38/OFF/source_schedules.json').read_text())
    old=ROOT/'results/sirm_research_loop/iteration_08'
    plan['anchor_sha256']=json.loads((dest/'plan.json').read_text())['anchor_sha256']
    assert json.loads((old/'plan.json').read_text())['anchor_sha256']==plan['anchor_sha256']
    assert digest(old/'experiment/subset_manifest.json')==plan['manifest_sha256']
    schedules={d:make_schedule(list(range(len(lock[str(d)]['split_keys']))),400,32,0,d) for d in [0,2,3]}
    assert all(hashlib.sha256(schedules[d].tobytes()).hexdigest()==lock[str(d)]['schedule_sha256'] for d in schedules)
    paired=[]; h2_corrections=[]
    for step in [200,215,250,300,350,395]:
        cache=old/'pair_shared_parameter_audit'/f'H2_step{step}.pt'
        batch=torch.load(cache,map_location='cpu',weights_only=False)
        image_ids=[]; expected_labels=[]
        for d in [0,2,3]:
            for i in schedules[d][step]:
                entry=manifest['sources'][str(d)]['images'][lock[str(d)]['split_keys'][int(i)]]
                image_ids.append(entry['id']);expected_labels.append(entry['class'])
        assert batch['y'].tolist()==expected_labels and batch['domains'].tolist()==[d for d in [0,2,3] for _ in range(32)]
        zz={'original':torch.stack([feature_lookup[i] for i in image_ids]),'H2':batch['z']}
        with torch.no_grad():
            for mode in ['saved','batch']:
                ff={v:outputs(model,z,batch['y'],mode) for v,z in zz.items()}
                for view,f in ff.items():
                    cg=(f['full']-f['g']).numpy(); by=batch['y'].numpy();bd=batch['domains'].numpy()
                    raw=evaluate(f['g'].numpy(),cg,by,bd,np.zeros(10))
                    removed=evaluate(f['g'].numpy(),cg,by,bd,biases['all_source_in'])
                    paired.append(dict(step=step,view=view,expert_center=mode,cache_sha256=digest(cache),
                        raw=raw,source_bias_removed=removed,composed_mean=domain_mean(cg,bd).tolist(),
                        paired_prediction_agreement=float((ff['original']['full'].argmax(1)==ff['H2']['full'].argmax(1)).double().mean())))
                    if mode=='saved' and view=='H2':h2_corrections.append(cg)
    h2_bias=np.concatenate(h2_corrections).mean(0)
    calibration['six_H2_source_batches']=evaluate(sg,sc,sy,sd,h2_bias)
    result=dict(plan=plan,source_out_global=erm,source_out_raw=baseline,
        source_out_bias_removed=calibration,held_domain_calibration=held,
        bias_vectors={k:v.tolist() for k,v in biases.items()},H2_bias=h2_bias.tolist(),
        H2_original_bias_distance=float(np.linalg.norm(h2_bias-biases['all_source_in'])),
        paired_batches=paired,shared_support_or_invariance_claim=False)
    with torch.no_grad():
        z,ty,td,_=sources['source_in'];f=outputs(model,z,ty)
        delta=f['delta'].numpy();pi=f['pi'][:,:6].numpy();dd=td.numpy()
        pi_mean=domain_mean(pi,dd);delta_mean=domain_mean(delta,dd)
        per_expert=domain_mean(pi[:,:,None]*delta,dd)
        covariance=domain_mean((pi-pi_mean)[...,None]*(delta-delta_mean),dd)
        identity_error=float(np.max(np.abs(per_expert-covariance-pi_mean[:,None]*delta_mean)))
        assert identity_error<1e-6
        result['composed_bias_decomposition']=dict(per_expert_mean=per_expert.tolist(),
            routing_delta_covariance=covariance.tolist(),expert_delta_mean=delta_mean.tolist(),
            identity_max_abs_error=identity_error,
            scope='Empirical equal-domain source-in covariance identity, not causal routing effect.')
    assert digest(path)==plan['checkpoint_sha256']
    write(out/'results.json',result)
    print(json.dumps({k:result[k] for k in ['source_out_global','source_out_raw','source_out_bias_removed','held_domain_calibration']},indent=2))


if __name__=='__main__':main()
