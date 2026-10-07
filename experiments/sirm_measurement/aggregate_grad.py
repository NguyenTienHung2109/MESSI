"""Hard feasible H6 assignments from cross-fitted angular gradient aggregation."""
import argparse,json,time
from pathlib import Path
import torch
from experiments.sirm_measurement.run import write,partition,gradient_audit,objectives,synthetic
from experiments.sirm_measurement.stability import risk_scores,pick

def assess(r,c,seed):
    r,c=r.double(),c.double();idx=pick(r.mean(0),c.mean(0));gen=torch.Generator().manual_seed(seed)
    choices=[]
    for _ in range(500):
        ids=torch.randint(len(r),(len(r),),generator=gen);choices.append(pick(r[ids].mean(0),c[ids].mean(0)))
    half=len(r)//2;halves=[pick(r[:half].mean(0),c[:half].mean(0)),pick(r[half:].mean(0),c[half:].mean(0))]
    ratio=float(c.mean(0).std(unbiased=False)/(c.std(0,unbiased=True).mean()/len(c)**.5).clamp_min(1e-12))
    frequency=choices.count(idx)/len(choices);changed=idx!=int(r.mean(0).argmin())
    return dict(index=idx,risk_index=int(r.mean(0).argmin()),frequency=frequency,halves=halves,signal_to_noise=ratio,
        differs_from_risk=changed,eligible=frequency>=.8 and halves[0]==halves[1] and ratio>=2 and changed)

def main():
    p=argparse.ArgumentParser();p.add_argument('--banks',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(4);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    write(out/'plan.json',dict(criterion='angular Grad',alpha=1,hard_feasible_support=True,discovery_resplits=80,confirmation_resplits=80,
        source_only=True,split_seeds=list(range(30000,30160)),gate='both independent resplit groups pass original hard-assignment gates; discovery and confirmation Q identical; >=8/12 banks',
        uncertainty_scope='conditional resplit stability, not independent-data or target generalization proof'))
    syn=synthetic();write(out/'synthetic.json',syn)
    if not syn['passed']:raise RuntimeError('synthetic test failed')
    banks=[];done=0
    for env in range(4):
        for seed in range(3):
            cache=torch.load(Path(args.banks)/'native'/f'env{env}'/f'seed{seed}'/'cache/source_in_original.pt',map_location='cpu',weights_only=False)
            rs=[];cs=[]
            for repeat in range(160):
                _,test=partition(cache,30000+repeat);risk,r=risk_scores(test);audit,angular=gradient_audit(test);_,c=objectives(risk,angular)
                if audit['autograd_max_error']>1e-10:raise RuntimeError('gradient mismatch')
                rs.append(r);cs.append(c);done+=1
                if done%20==0:
                    elapsed=time.time()-start;eta=elapsed/done*(1920-done)
                    write(out/'progress.json',dict(completed=done,total=1920,elapsed_seconds=elapsed,estimated_remaining_seconds=eta,status='running'))
                    print(f'{done}/1920 ETA {eta/60:.1f}min',flush=True)
            r,c=torch.stack(rs),torch.stack(cs);discovery=assess(r[:80],c[:80],777);confirmation=assess(r[80:],c[80:],778)
            bank=dict(env=env,seed=seed,discovery=discovery,confirmation=confirmation,
                eligible=discovery['eligible'] and confirmation['eligible'] and discovery['index']==confirmation['index'])
            torch.save(dict(risk=r,angular=c),out/f'env{env}_seed{seed}_scores.pt');banks.append(bank);write(out/f'env{env}_seed{seed}.json',bank)
            print(f'bank env{env}/seed{seed} eligible={bank["eligible"]}',flush=True)
    count=sum(b['eligible'] for b in banks)
    write(out/'report.json',dict(banks=banks,eligible_count=count,passed=count>=8,elapsed_seconds=time.time()-start,target_used=False))
    write(out/'progress.json',dict(completed=1920,total=1920,status='completed',elapsed_seconds=time.time()-start,estimated_remaining_seconds=0))

if __name__=='__main__':main()
