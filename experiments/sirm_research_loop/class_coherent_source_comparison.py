"""Source-only selected/common399 comparison against the completed image-hard bank."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from domainbed.deit_transformer import DeiTFeaturizer
from .centered_model import CenteredGateSIRM
from .hard_router_diagnosis import outputs, metrics
from .router_weight import live
from .shrinkage import ROOT, digest, write


def main():
    torch.set_num_threads(4)
    dest=ROOT/'results/sirm_research_loop/iteration_12'; prior=ROOT/'results/sirm_research_loop/iteration_11'
    out=dest/'source_comparison';out.mkdir(exist_ok=True)
    pid=json.loads((dest/'worker_pid.json').read_text())['pid']
    bank=dest/'experiment/runs/L38/OFF'
    while not (bank/'done').exists():
        assert live(pid),'controller terminal before OFF done; inspect failure, do not restart'
        write(out/'progress.json',dict(status='waiting_for_OFF',controller_pid=pid,verified_live=True,observed_unix=time.time()))
        time.sleep(10)
    assert json.loads((prior/'completion_audit.json').read_text())['analysis_complete']
    plans={key:json.loads((folder/'plan.json').read_text()) for key,folder in [('11',prior),('12',dest)]}
    assert plans['11']['anchor_sha256']==plans['12']['anchor_sha256']
    assert plans['11']['manifest_sha256']==plans['12']['manifest_sha256']
    assert plans['11']['feature_cache_sha256']==plans['12']['feature_cache_sha256']
    for plan in plans.values():
        for group in ['base_code_hashes','variant_code_hashes']:
            assert all(digest(ROOT/p)==h for p,h in plan[group].items())
    models={};locks={}
    for iteration,folder in [('11',prior),('12',dest)]:
        for state,filename in [('selected','source_best.pkl'),('last','last.pkl')]:
            path=folder/'experiment/runs/L38/OFF'/filename
            payload=torch.load(path,map_location='cpu',weights_only=False); c=payload['config']
            step=payload['selected_step'] if state=='selected' else payload['step']
            if state=='last':assert step==399
            model=CenteredGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c)
            model.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()})
            model.configure_stage();model.eval();model.requires_grad_(False)
            key=iteration+'_'+state;models[key]=model
            locks[key]=dict(path=str(path),checkpoint_sha256=digest(path),step=step)
    a,b=models['11_last'].config,models['12_last'].config
    diff={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    assert set(diff)<= {'protocol','center_feature_cache','discovery_scope'}
    for model in models.values():
        for mod in ['featurizer','global_head']:
            old=getattr(models['11_last'],mod).state_dict();new=getattr(model,mod).state_dict()
            assert all(torch.equal(v,new[k]) for k,v in old.items())
    proof=json.loads((prior/'composed_bias_source_diagnosis/results.json').read_text())
    result=dict(code_sha256=digest(Path(__file__)),locks=locks,config_differences=diff,
                target_read=False,training=False,fitting=False,selection_changes=False,
                scope='Original source-in/out only. Selected checkpoint times may differ; common399 endpoint is separate. '
                      'Label-dependent teacher/capacity/winner statistics are exploratory oracles, not deployable policies.',splits={})
    identities={}
    for split in ['source_in','source_out']:
        path=ROOT/'results/sirm_research_loop/iteration_09/source_router_diagnosis'/f'{split}_original.npz'
        assert digest(path)==proof['plan']['source_cache_hashes'][split]
        with np.load(path) as cache:
            z,y,d=[torch.from_numpy(cache[k]) for k in ['z','y','domains']];ids=cache['ids'].tolist()
        identities[split]=set(ids);records={};replays={}
        with torch.no_grad():
            for key,model in models.items():
                f=outputs(model,z,y);replays[key]=f
                m=metrics(f,y,d);wrong=f['g'].argmax(1)!=y
                unscaled=(f['g'][:,None]+f['delta']).argmax(2)==y[:,None]
                routed=(f['g'][:,None]+f['pi'][:,:6,None]*f['delta']).argmax(2)==y[:,None]
                composed=f['full']-f['g'];flat=f['delta'].permute(1,0,2).flatten(1).double();unit=F.normalize(flat,dim=1)
                cos=unit@unit.T;offdiag=~torch.eye(6,dtype=torch.bool)
                group=[]
                for cls in y.unique(sorted=True):
                    mask=y==cls;w=f['winner'][mask]
                    loads=F.one_hot(w,6).double().mean(0)
                    group.append(dict(label=int(cls),n=int(mask.sum()),winner_load=loads.tolist(),
                                      modal_expert_fraction=float(loads.max()),
                                      unscaled_global_error_repairs_per_expert=(unscaled & wrong[:,None] & mask[:,None]).sum(0).tolist()))
                records[key]=dict(metrics=m,global_errors=int(wrong.sum()),
                    unscaled_repair_capacity=int((unscaled.any(1)&wrong).sum()),
                    routed_single_repair_capacity=int((routed.any(1)&wrong).sum()),
                    mean_output_cosine=float(cos[offdiag].mean()),
                    expert_RMS=float(f['delta'].double().square().mean().sqrt()),
                    composed_correction_RMS=float(composed.double().square().mean().sqrt()),
                    correction_global_RMS=float(composed.double().square().mean().sqrt()/f['g'].double().square().mean().sqrt()),
                    label_groups=group)
            assert all(torch.equal(replays['11_last']['g'],f['g']) for f in replays.values())
            changes={}
            for state in ['selected','last']:
                old=replays['11_'+state]['full'];new=replays['12_'+state]['full']
                oc,nc=old.argmax(1)==y,new.argmax(1)==y;mask=oc!=nc
                changes[state]=dict(new_correct=int((~oc&nc).sum()),new_wrong=int((oc&~nc).sum()),
                    cases=[dict(image_id=ids[i],domain=int(d[i]),label=int(y[i]),
                                global_prediction=int(replays['11_last']['g'][i].argmax()),
                                old_prediction=int(old[i].argmax()),new_prediction=int(new[i].argmax()),
                                old_CE=float(F.cross_entropy(old[i:i+1],y[i:i+1])),
                                new_CE=float(F.cross_entropy(new[i:i+1],y[i:i+1])))
                           for i in mask.nonzero().flatten().tolist()])
            result['splits'][split]=dict(cache_sha256=digest(path),models=records,correctness_changes=changes)
            np.savez_compressed(out/(split+'_replay.npz'),ids=np.array(ids),y=y.numpy(),domains=d.numpy(),
                                **{key+'_'+name: f[name].numpy() for key,f in replays.items() for name in ['g','full','delta','pi']})
    assert not identities['source_in'] & identities['source_out']
    assert all(digest(Path(v['path']))==v['checkpoint_sha256'] for v in locks.values())
    write(out/'results.json',result)
    write(out/'progress.json',dict(status='source_comparison_complete',target_read=False,
                                  remaining='original-image comparison review after complete primary pair',scientific_goal_achieved=False))
    print(json.dumps({s:{k:dict(actual=v['metrics']['equal_domain']['actual'],capacity=v['unscaled_repair_capacity']) for k,v in r['models'].items()} for s,r in result['splits'].items()},indent=2))

if __name__=='__main__':main()
