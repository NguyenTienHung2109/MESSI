"""Source-only saved-bank class/domain winner audit, without updates or fitting."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover.math import center_logits, expert_ce
from .centered_model import CenteredGateSIRM
from .class_coherent_discovery import class_coherent_cost
from .discovery_error_gradient_diagnosis import vector
from .shrinkage import ROOT, digest, write


def costs(losses, y, domains):
    blocks=[]; objectives={name:losses.sum()*0 for name in ['image_weighted','equal_domain','worst_domain']}
    for label in y.unique(sorted=True):
        mask=y.eq(label);ds=domains[mask].unique(sorted=True)
        domain_means=torch.stack([losses[mask & domains.eq(d)].mean(0) for d in ds])
        counts=torch.stack([(mask & domains.eq(d)).sum() for d in ds])
        means=losses[mask].mean(0); balanced=domain_means.mean(0);worst=domain_means.amax(0)
        assert torch.allclose(means,(domain_means*counts[:,None]).sum(0)/counts.sum(),atol=2e-6,rtol=2e-6)
        choices={k:v.eq(v.amin()).nonzero().flatten().tolist() for k,v in [('image_weighted',means),('equal_domain',balanced),('worst_domain',worst)]}
        majority=counts.eq(counts.max()).nonzero().flatten().tolist()
        dw=[v.eq(v.amin()).nonzero().flatten().tolist() for v in domain_means]
        def overlaps(a,b):return bool(set(a)&set(b))
        blocks.append(dict(label=int(label),count=int(mask.sum()),domains=ds.tolist(),domain_counts=counts.tolist(),
            domain_expert_CE=domain_means.detach().tolist(),image_weighted_expert_CE=means.detach().tolist(),
            equal_domain_expert_CE=balanced.detach().tolist(),worst_domain_expert_CE=worst.detach().tolist(),
            winners=choices,domain_winners=dw,majority_domain_indices=majority,
            cross_domain=int(len(ds))>=2,minimum_two_in_two_domains=int((counts>=2).sum())>=2,
            image_equal_disagree=not overlaps(choices['image_weighted'],choices['equal_domain']),
            image_worst_disagree=not overlaps(choices['image_weighted'],choices['worst_domain']),
            image_agrees_majority=any(overlaps(choices['image_weighted'],dw[i]) for i in majority),
            all_domains_share_winner=bool(set.intersection(*(set(v) for v in dw))),
            equal_domain_regret_of_image_winner=float(balanced[choices['image_weighted']].mean().detach()-balanced.amin().detach())))
        for name,cost in [('image_weighted',means),('equal_domain',balanced),('worst_domain',worst)]:
            objectives[name]=objectives[name]+mask.sum()/len(y)*cost.amin()
    assert torch.allclose(objectives['image_weighted'],class_coherent_cost(losses,y),atol=1e-6)
    summaries={}
    for scope,group in [('all',blocks),('cross_domain',[r for r in blocks if r['cross_domain']]),('minimum_two_in_two_domains',[r for r in blocks if r['minimum_two_in_two_domains']])]:
        n=sum(r['count'] for r in group);summary=dict(blocks=len(group),image_occurrences=n)
        for key in ['image_equal_disagree','image_worst_disagree','image_agrees_majority','all_domains_share_winner']:
            summary[key+'_blocks']=sum(r[key] for r in group)
            summary[key+'_fraction']=sum(r[key] for r in group)/len(group) if group else None
            summary[key+'_image_weight_fraction']=sum(r['count'] for r in group if r[key])/n if n else None
        summary['mean_equal_domain_regret']=float(np.mean([r['equal_domain_regret_of_image_winner'] for r in group])) if group else None
        summaries[scope]=summary
    return objectives,blocks,summaries


def assess(model,z,y,domains,mode,reference=None):
    assert set(domains.tolist())=={0,2,3}
    with torch.no_grad():g=model.global_head(z);pi=model.router(z).softmax(1)
    raw=center_logits(torch.stack([e(z) for e in model.residuals],1))
    delta=raw-(raw.mean(0,keepdim=True) if mode=='batch' else model.expert_source_mean[None])
    losses=expert_ce(g[:,None]+delta,y);objectives,blocks,summary=costs(losses,y,domains)
    full=g+(pi[:,:6,None]*delta).sum(1)
    # Equal-domain source validation CE is the locked selection convention.
    cls=torch.stack([F.cross_entropy(full[domains.eq(d)],y[domains.eq(d)]) for d in domains.unique()]).mean()
    params=list(model.residuals.parameters());vs={k:vector(model.config['lambda_discover']*v,params) for k,v in objectives.items()}
    cv=vector(cls,params)
    def cos(a,b):return float(F.cosine_similarity(a,b,dim=0)) if a.norm() and b.norm() else None
    grads={k:dict(loss=float(objectives[k].detach()),weighted_gradient_norm=float(v.norm()),cosine_to_local_equal_domain_classification=cos(v,cv),
                 cosine_to_source_out_classification=cos(v,reference) if reference is not None else None,
                 cosine_to_current_discovery=cos(v,vs['image_weighted'])) for k,v in vs.items()}
    return dict(n=len(y),centering=mode,summaries=summary,blocks=blocks,gradients=grads),cv


def main():
    torch.set_num_threads(4);root=ROOT/'results/sirm_research_loop/iteration_12';out=root/'class_domain_winner_diagnosis';out.mkdir(exist_ok=True)
    plan=json.loads((root/'plan.json').read_text())
    for key in ['base_code_hashes','variant_code_hashes']:assert all(digest(ROOT/p)==h for p,h in plan[key].items())
    proof=json.loads((root/'prelaunch_gradient_audit.json').read_text());records=[];locks={}
    for name in ['source_best','last']:
        path=root/'experiment/runs/L38/OFF'/f'{name}.pkl';payload=torch.load(path,map_location='cpu',weights_only=False);step=payload['selected_step'] if 'selected_step' in payload else payload['step'];c=payload['config']
        model=CenteredGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c)
        model.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});model.configure_stage();model.eval()
        before={k:v.clone() for k,v in model.state_dict().items()};locks[name]=dict(sha256=digest(path),step=step)
        cache={}
        for split in ['source_out','source_in']:
            file=ROOT/'results/sirm_research_loop/iteration_09/source_router_diagnosis'/f'{split}_original.npz'
            row=next(r for r in proof['records'] if r['split']==split);assert digest(file)==row['cache_sha256']
            with np.load(file) as a:cache[split]=[torch.from_numpy(a[k]) for k in ['z','y','domains']]
        val,reference=assess(model,*cache['source_out'],'saved');records.append(dict(checkpoint=name,checkpoint_step=step,view='original',split='source_out',metrics=val))
        train,_=assess(model,*cache['source_in'],'saved',reference);records.append(dict(checkpoint=name,checkpoint_step=step,view='original',split='source_in',metrics=train))
        for scheduled in [200,215,250,300,350,395]:
            file=ROOT/'results/sirm_research_loop/iteration_08/pair_shared_parameter_audit'/f'H2_step{scheduled}.pt'
            assert all(digest(file)==r['cache_sha256'] for r in proof['records'] if r.get('step')==scheduled)
            b=torch.load(file,map_location='cpu',weights_only=False);assert b['scheduled_step']==scheduled
            for mode in ['saved','batch']:
                metric,_=assess(model,b['z'],b['y'],b['domains'],mode,reference)
                records.append(dict(checkpoint=name,checkpoint_step=step,view='H2',split='source_in_scheduled',scheduled_batch_step=scheduled,metrics=metric))
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items());assert all(p.grad is None for p in model.parameters())
        assert all(not p.requires_grad for mod in [model.featurizer,model.global_head] for p in mod.parameters())
    result=dict(code_sha256=digest(Path(__file__)),checkpoint_locks=locks,training=False,fitting=False,target_read=False,parameter_updates=False,
                state_tensors_unchanged=True,source_selection_unchanged=True,records=records,
                scope='Saved selected225 and endpoint399 banks replayed on original source features and six exact scheduled H2 batches. These are not historical per-step costs/gradients. Equal-domain and worst-domain objectives are diagnostic counterfactuals only; outer class image weighting retained. Gradient norms cover residual parameters, lambda_discover unchanged. No changed Q or training objective.')
    write(out/'results.json',result)
    print(json.dumps([{k:v for k,v in r.items() if k!='metrics'}|{'cross_domain':r['metrics']['summaries']['cross_domain'],'gradients':r['metrics']['gradients']} for r in records],indent=2))

if __name__=='__main__':main()
