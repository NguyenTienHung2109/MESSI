"""Source-only discovery gradient allocation, saved state, without optimizer updates."""
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover.math import center_logits,multiple_choice_loss
from .centered_model import CenteredGateSIRM
from .shrinkage import ROOT,digest,write


def vector(loss,params):
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().double().flatten() for p,g in zip(params,gs)])


def assess(model,z,y,domains,centering):
    params=list(model.residuals.parameters())
    with torch.no_grad():g=model.global_head(z);pi=model.router(z).softmax(1)
    raw=center_logits(torch.stack([e(z) for e in model.residuals],1))
    delta=raw-(raw.mean(0,keepdim=True) if centering=='batch' else model.expert_source_mean[None])
    full=g+(pi[:,:6,None]*delta).sum(1)
    per=F.cross_entropy((g[:,None]+delta).flatten(0,1),y[:,None].expand(-1,6).flatten(),reduction='none').reshape(-1,6).amin(1)
    clsper=F.cross_entropy(full,y,reduction='none')
    wrong=g.argmax(1)!=y;correct=~wrong;n=len(y)
    wd=model.config['lambda_discover']
    dg={name:vector(wd*(per*mask).sum()/n,params) for name,mask in [('correct',correct),('wrong',wrong)]}
    total=vector(wd*per.mean(),params)
    assert torch.allclose(dg['correct']+dg['wrong'],total,atol=1e-7,rtol=1e-5)
    cls=vector(clsper.mean(),params)
    cv,wv=dg['correct'],dg['wrong'];tn=float(total.norm());an=float(cv.norm());bn=float(wv.norm())
    def cosine(a,b):
        return float(F.cosine_similarity(a,b,dim=0)) if a.norm()>0 and b.norm()>0 else None
    rows={}
    for name,mask in [('correct',correct),('wrong',wrong)]:
        v=dg[name];count=int(mask.sum())
        rows[name]=dict(n=count,weighted_loss_contribution=float(wd*(per*mask).sum()/n),
            conditional_mean_CE=float(per[mask].mean()) if count else None,
            gradient_norm=float(v.norm()),conditional_mean_gradient_norm=float(v.norm())*n/count if count else None,
            cosine_to_full_classification=cosine(v,cls),
            projection_fraction_of_discovery=float(v.dot(total)/(total.dot(total))) if tn else None)
    domain_wrong={int(d):vector(wd*(per*(wrong & domains.eq(d))).sum()/n,params)
                  for d in domains.unique(sorted=True)}
    assert torch.allclose(sum(domain_wrong.values()),dg['wrong'],atol=1e-7,rtol=1e-5)
    domain_pairs=[dict(domains=[a,b],cosine=cosine(domain_wrong[a],domain_wrong[b]))
                  for a in domain_wrong for b in domain_wrong if a<b]
    conditional=[]
    if len(y)>96:
        for clslabel in y.unique(sorted=True).tolist():
            vectors={};counts={}
            for d in domains.unique(sorted=True).tolist():
                mask=wrong & y.eq(clslabel) & domains.eq(d);count=int(mask.sum())
                if count<2:continue
                vectors[d]=vector(wd*(per*mask).sum()/count,params);counts[d]=count
            for a in vectors:
                for b in vectors:
                    if a<b:conditional.append(dict(label=clslabel,domains=[a,b],counts=[counts[a],counts[b]],
                        cosine=cosine(vectors[a],vectors[b])))
    assert all(p.grad is None for p in model.parameters())
    result=dict(n=n,global_error_fraction=float(wrong.double().mean()),groups=rows,
        weighted_discovery_gradient_norm=tn,classification_gradient_norm=float(cls.norm()),
        wrong_to_correct_contribution_norm=bn/an if an else None,
        correct_wrong_gradient_cosine=cosine(cv,wv),discovery_classification_cosine=cosine(total,cls),
        wrong_case_domain_gradient_pairs=domain_pairs,
        class_matched_wrong_domain_pairs=conditional,
        additive_gradient_error=float((cv+wv-total).abs().max()),
        norm_scope='Residual parameters only. Contributions divide each masked loss sum by full batch N. '
                   'Batch-centering couples cases: these are gradients of group losses, not isolated per-image gradients.')
    result['_vectors']={'discovery':total,'classification':cls,
        'discovery_correct':cv,'discovery_wrong':wv,
        'classification_correct':vector((clsper*correct).sum()/n,params),
        'classification_wrong':vector((clsper*wrong).sum()/n,params)}
    return result


def main():
    torch.set_num_threads(4)
    root=ROOT/'results/sirm_research_loop/iteration_11';out=root/'discovery_error_gradient_diagnosis';out.mkdir(exist_ok=True)
    path=root/'experiment/runs/L38/OFF/source_best.pkl'
    payload=torch.load(path,map_location='cpu',weights_only=False);assert payload['selected_step']==399
    c=payload['config'];model=CenteredGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});model.configure_stage();model.eval()
    assert all(not p.requires_grad for mod in [model.featurizer,model.global_head] for p in mod.parameters())
    before={k:v.clone() for k,v in model.state_dict().items()}
    validated=json.loads((root/'composed_bias_source_diagnosis/results.json').read_text())
    assert validated['plan']['checkpoint_sha256']==digest(path)
    plan=dict(code_sha256=digest(Path(__file__)),checkpoint_sha256=digest(path),lambda_discover=c['lambda_discover'],
        target_read=False,training=False,fitting=False,parameter_updates=False,
        scope='Single saved399 bank: source original caches and six saved scheduled H2 batches. '
              'Not historical gradients or all-update statistics. No optimizer, model selection or hyperparameter changes.')
    records=[]
    for split in ['source_in','source_out']:
        p=ROOT/'results/sirm_research_loop/iteration_09/source_router_diagnosis'/f'{split}_original.npz'
        with np.load(p) as a:z,y,d=[torch.from_numpy(a[k]) for k in ['z','y','domains']]
        assert digest(p)==validated['plan']['source_cache_hashes'][split]
        assert set(d.tolist())=={0,2,3}
        records.append(dict(split=split,view='original',centering='saved',cache_sha256=digest(p),metrics=assess(model,z,y,d,'saved')))
    for step in [200,215,250,300,350,395]:
        p=ROOT/'results/sirm_research_loop/iteration_08/pair_shared_parameter_audit'/f'H2_step{step}.pt'
        b=torch.load(p,map_location='cpu',weights_only=False)
        previous=next(row for row in validated['paired_batches'] if row['step']==step)
        assert digest(p)==previous['cache_sha256']
        for mode in ['saved','batch']:
            records.append(dict(split='source_in_scheduled',scheduled_step=step,view='H2',centering=mode,
                cache_sha256=digest(p),metrics=assess(model,b['z'],b['y'],b['domains'],mode)))
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert digest(path)==plan['checkpoint_sha256']
    reference=records[1]['metrics']['_vectors'];alignment=[]
    for row in records:
        if row['split']=='source_out':continue
        vectors=row['metrics']['_vectors']
        values={}
        for loss in ['discovery','discovery_correct','discovery_wrong','classification']:
            for group in ['classification','classification_correct','classification_wrong']:
                a,b=vectors[loss],reference[group]
                values[loss+'_to_source_out_'+group]=float(F.cosine_similarity(a,b,dim=0)) if a.norm()>0 and b.norm()>0 else None
        alignment.append(dict(split=row['split'],scheduled_step=row.get('scheduled_step'),centering=row['centering'],cosines=values))
    for row in records:del row['metrics']['_vectors']
    write(out/'results.json',dict(plan=plan,records=records,source_validation_direction_alignment=alignment,state_tensors_unchanged=True))
    print(json.dumps([{k:v for k,v in row.items() if k!='cache_sha256'} for row in records],indent=2))


if __name__=='__main__':main()
