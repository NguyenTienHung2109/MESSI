"""Independent source resplits for angular Grad and aggregated Regret."""
import argparse,json,time
from pathlib import Path
import torch
from experiments.sirm_measurement.run import write,partition,gradient_audit,objectives,synthetic
from experiments.sirm_h1_h6.statistics import fit_gap_probes,evaluate_gaps,expert_losses,feasible_supports

def risk_scores(cache):
    losses=expert_losses(cache['expert_logits'],cache['labels'])
    risk=torch.stack([losses[cache['domains']==d].mean(0) for d in range(3)])
    return risk,(feasible_supports()*risk[None]).sum((1,2))/6

def pick(r,c):
    return int(((r-r.mean())/r.std(unbiased=False).clamp_min(1e-8)+(c-c.mean())/c.std(unbiased=False).clamp_min(1e-8)).argmin())

def assess(r,c):
    r,c=r.double(),c.double();mean_r,mean_c=r.mean(0),c.mean(0);idx=pick(mean_r,mean_c)
    gen=torch.Generator().manual_seed(819);choices=[]
    for _ in range(500):
        ids=torch.randint(len(r),(len(r),),generator=gen);choices.append(pick(r[ids].mean(0),c[ids].mean(0)))
    frequency=choices.count(idx)/len(choices);halves=[pick(r[:5].mean(0),c[:5].mean(0)),pick(r[5:].mean(0),c[5:].mean(0))]
    noise=c.std(0,unbiased=True).mean()/len(c)**.5;signal=mean_c.std(unbiased=False)
    ratio=float(signal/noise.clamp_min(1e-12));changed=idx!=int(mean_r.argmin())
    return dict(assignment_index=idx,risk_index=int(mean_r.argmin()),differs_from_risk=changed,
        bootstrap_selection_frequency=frequency,discovery_confirmation_indices=halves,halves_agree=halves[0]==halves[1],
        signal_to_mean_resplit_noise=ratio,eligible=changed and frequency>=.8 and halves[0]==halves[1] and ratio>=2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--banks',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(4);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    write(out/'plan.json',dict(banks=12,resplits=10,split_seeds=list(range(20000,20010)),source_only=True,
        gate='per bank: differs from Risk; bootstrap selection >=0.8; first5 vs last5 identical Q; compatibility spread / mean-resplit noise >=2. Criterion needs >=8/12 banks.',
        proposed_full_runs_per_passing_criterion=12,angular_metric='1 - cosine, class-conditioned head gradients',regret_metric='mean gap over ten disjoint-fit/stat source partitions'))
    syn=synthetic();write(out/'synthetic.json',syn)
    if not syn['passed']:raise RuntimeError('synthetic test failed')
    banks=[];durations=[]
    for env in range(4):
        for seed in range(3):
            cache=torch.load(Path(args.banks)/'native'/f'env{env}'/f'seed{seed}'/'cache/source_in_original.pt',map_location='cpu',weights_only=False)
            rows=[];rs=[];gs=[];cs=[]
            for repeat in range(10):
                t=time.time();fit,test=partition(cache,20000+repeat);risk,r=risk_scores(test)
                audit,angular=gradient_audit(test);_,g=objectives(risk,angular)
                probes=fit_gap_probes(fit,max_iter=300);gap,se=evaluate_gaps(probes,test);_,c=objectives(risk,gap)
                rs.append(r);gs.append(g);cs.append(c)
                rows.append(dict(repeat=repeat,risk=r.tolist(),angular=g.tolist(),regret=c.tolist(),autograd_error=audit['autograd_max_error']))
                durations.append(time.time()-t);done=len(durations);eta=sum(durations)/done*(120-done)
                write(out/'progress.json',dict(completed=done,total=120,status='running',elapsed_seconds=time.time()-start,estimated_remaining_seconds=eta))
                print(f'{done}/120 env{env} seed{seed} split{repeat} {durations[-1]:.2f}s ETA {eta/60:.1f}min',flush=True)
            bank=dict(env=env,seed=seed,Grad=assess(torch.stack(rs),torch.stack(gs)),Regret=assess(torch.stack(rs),torch.stack(cs)),rows=rows)
            banks.append(bank);write(out/f'env{env}_seed{seed}.json',bank)
    counts={k:sum(b[k]['eligible'] for b in banks) for k in ['Grad','Regret']}
    write(out/'report.json',dict(banks=banks,eligible_counts=counts,passing_criteria=[k for k,v in counts.items() if v>=8],elapsed_seconds=time.time()-start,target_used=False))
    write(out/'progress.json',dict(completed=120,total=120,status='completed',elapsed_seconds=time.time()-start,estimated_remaining_seconds=0))

if __name__=='__main__':main()
