"""Source-only measurement audit. Existing training code is unchanged."""
import argparse,itertools,json,time,os
from pathlib import Path
import torch
import torch.nn.functional as F
from experiments.sirm_h1_h6.statistics import (fit_gap_probes,evaluate_gaps,expert_losses,feasible_supports,classifier_gradient_gaps)
from experiments.sirm_h1_h6.controls import subset_cache

def write(p,x):
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def gradient_audit(cache):
    z,l,y,d=[cache[k].double() if k in ['expert_features','expert_logits'] else cache[k] for k in ['expert_features','expert_logits','labels','domains']]
    gradients={}; errors=[]
    for domain in range(3):
        for c in range(l.shape[-1]):
            mask=(d==domain)&(y==c)
            if mask.sum()<2:continue
            x=torch.cat([z[mask],torch.ones_like(z[mask,:,:1])],-1)
            residual=l[mask].softmax(-1)-F.one_hot(y[mask],l.shape[-1])[:,None]
            g=torch.einsum('nmc,nmf->mcf',residual,x)/mask.sum()
            gradients[domain,c]=g.flatten(1)
            # Perturb the cached head logits with a linear weight increment.
            delta=torch.zeros_like(g,requires_grad=True)
            loss=sum(F.cross_entropy(l[mask,m]+x[:,m]@delta[m].T,y[mask]) for m in range(6))
            actual=torch.autograd.grad(loss,delta)[0]
            errors.append((actual-g).abs().max().item())
    old=torch.zeros(8,6,dtype=torch.double); angular=old.clone(); counts=old.clone(); cos=[]
    norms=torch.cat([g.norm(dim=1) for g in gradients.values()])
    for subset in [3,5,6,7]:
        for a,b in itertools.combinations([i for i in range(3) if subset&(1<<i)],2):
            for c in range(l.shape[-1]):
                if (a,c) not in gradients or (b,c) not in gradients:continue
                ga,gb=gradients[a,c],gradients[b,c]
                valid=(ga.norm(dim=1)>1e-8)&(gb.norm(dim=1)>1e-8)
                similarity=F.cosine_similarity(ga,gb,dim=1)
                old[subset]+=F.relu(-similarity)*valid
                angular[subset]+=(1-similarity)*valid
                counts[subset]+=valid;cos.extend(similarity[valid].tolist())
    old/=counts.clamp_min(1);angular/=counts.clamp_min(1)
    return dict(gradient_norm_min=norms.min().item(),gradient_norm_max=norms.max().item(),
        autograd_max_error=max(errors),cosine_min=min(cos),cosine_max=max(cos),negative_cosine_fraction=sum(x<0 for x in cos)/len(cos),
        original_gap=old.tolist(),angular_gap=angular.tolist()),angular.float()

def objectives(risk,gap):
    supports=feasible_supports();masks=sum(supports[:,d].long()*(1<<d) for d in range(3));active=masks>0
    r=(supports*risk[None]).sum((1,2))/6
    c=(gap[masks,torch.arange(6).expand_as(masks)]*active).sum(1)/active.sum(1)
    idx=((r-r.mean())/r.std(unbiased=False).clamp_min(1e-8)+(c-c.mean())/c.std(unbiased=False).clamp_min(1e-8)).argmin().item()
    return dict(risk_index=r.argmin().item(),normalized_index=idx,compatibility_std=c.std(unbiased=False).item()),c

def partition(cache,seed):
    mask=torch.zeros(len(cache['labels']),dtype=torch.bool);gen=torch.Generator().manual_seed(seed)
    for d in range(3):
        for c in cache['labels'].unique():
            ids=((cache['domains']==d)&(cache['labels']==c)).nonzero().flatten()
            ids=ids[torch.randperm(len(ids),generator=gen)];mask[ids[:len(ids)//2]]=True
    return subset_cache(cache,mask),subset_cache(cache,~mask)

def synthetic():
    gen=torch.Generator().manual_seed(937);n=240
    d=torch.arange(3).repeat_interleave(n);y=torch.randint(2,(3*n,),generator=gen)
    signal=(2*y-1).float();x=torch.randn(3*n,6,3,generator=gen)*.15
    # Expert 0 is compatible for domains 0/1, with reversed class relation in domain 2.
    x[:,:,0]=signal[:,None];x[d==2,0,0]=-signal[d==2]
    cache=dict(expert_features=x,expert_logits=torch.zeros(3*n,6,2),labels=y,domains=d,image_ids=list(range(3*n)))
    audit,angular=gradient_audit(cache);fit,test=partition(cache,44)
    probes=fit_gap_probes(fit);gap,se=evaluate_gaps(probes,test)
    original,_=classifier_gradient_gaps(test)
    passed=bool(original[3,0]<original[5,0] and angular[3,0]<angular[5,0] and gap[3,0]<gap[5,0])
    return dict(passed=passed,original_grad=original.tolist(),angular_grad=angular.tolist(),regret=gap.tolist(),regret_se=se.tolist(),audit=audit)

def main():
    p=argparse.ArgumentParser();p.add_argument('--banks',required=True);p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=5);args=p.parse_args()
    torch.set_num_threads(4);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    write(out/'plan.json',dict(banks=12,repeats=args.repeats,linear_probe_fits=12*args.repeats*42,target_used=False,angular_metric_role='diagnostic candidate, not confirmed replacement'))
    syn=synthetic();write(out/'synthetic.json',syn)
    if not syn['passed']:raise RuntimeError('synthetic known-answer test failed')
    rows=[];total=12*args.repeats
    for env in range(4):
        for seed in range(3):
            cache=torch.load(Path(args.banks)/'native'/f'env{env}'/f'seed{seed}'/'cache/source_in_original.pt',map_location='cpu',weights_only=False)
            audit,angular=gradient_audit(cache);write(out/f'grad_env{env}_seed{seed}.json',audit)
            for repeat in range(args.repeats):
                t=time.time();fit,test=partition(cache,10000+repeat)
                probes=fit_gap_probes(fit,max_iter=300);gap,se=evaluate_gaps(probes,test)
                losses=expert_losses(test['expert_logits'],test['labels']);risk=torch.stack([losses[test['domains']==d].mean(0) for d in range(3)])
                regret,c=objectives(risk,gap);grad,_=objectives(risk,angular)
                supports=feasible_supports();masks=sum(supports[:,d].long()*(1<<d) for d in range(3));active=masks>0
                typical=((se[masks,torch.arange(6).expand_as(masks)]*active).sum(1)/active.sum(1)).mean().item()
                row=dict(env=env,seed=seed,repeat=repeat,seconds=time.time()-t,regret=regret,angular_grad=grad,signal_to_se=regret['compatibility_std']/max(typical,1e-12),objective_scores=c.tolist(),gap=gap.tolist(),se=se.tolist(),probe_gradient_max=max(v['gradient_max'] for v in probes.values()))
                rows.append(row);write(out/f'env{env}_seed{seed}_repeat{repeat}.json',row)
                elapsed=time.time()-start;remaining=sum(r['seconds'] for r in rows)/len(rows)*(total-len(rows))
                write(out/'progress.json',dict(pid=os.getpid(),completed=len(rows),total=total,elapsed_seconds=elapsed,estimated_remaining_seconds=remaining,status='running'))
                print(f'{len(rows)}/{total} env{env} seed{seed} split{repeat}: {row["seconds"]:.1f}s ETA {remaining/60:.1f}min',flush=True)
    groups=[]
    for env in range(4):
        for seed in range(3):
            group=[r for r in rows if r['env']==env and r['seed']==seed];idx=[r['regret']['normalized_index'] for r in group]
            scores=torch.tensor([r['objective_scores'] for r in group]);spread=scores.mean(0).std(unbiased=False).item();noise=scores.std(0,unbiased=False).mean().item()
            groups.append(dict(env=env,seed=seed,regret_assignment_indices=idx,modal_assignment_fraction=max(idx.count(i) for i in set(idx))/len(idx),between_assignment_spread=spread,resplit_noise=noise,spread_to_resplit_noise=spread/max(noise,1e-12)))
    write(out/'report.json',dict(synthetic_passed=syn['passed'],elapsed_seconds=time.time()-start,banks=groups,rows=rows))
    write(out/'progress.json',dict(completed=total,total=total,status='completed',elapsed_seconds=time.time()-start,estimated_remaining_seconds=0))

if __name__=='__main__':main()
