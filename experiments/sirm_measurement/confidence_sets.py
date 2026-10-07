"""Source resplit confidence sets; no target selection or legacy edits."""
import argparse,json,time
from pathlib import Path
import torch
from experiments.sirm_measurement.run import write

def confidence_set(r,c,seed=410):
    r,c=r.double(),c.double();n=len(r)
    rs=r.mean(0).std(unbiased=False).clamp_min(1e-8);cs=c.mean(0).std(unbiased=False).clamp_min(1e-8)
    # Hold scaling fixed during bootstrap so uncertainty pertains to one objective.
    samples=r/rs+c/cs;mean=samples.mean(0);best=int(mean.argmin())
    gen=torch.Generator().manual_seed(seed);ids=torch.randint(n,(2000,n),generator=gen)
    errors=samples[ids].mean(1)-mean
    contrast_errors=errors-errors[:,best,None]
    radius=float(contrast_errors.abs().amax(1).quantile(.95))
    members=((mean-mean[best])<=radius).nonzero().flatten().tolist()
    risk_mean=r.mean(0);chosen=min(members,key=lambda i:float(risk_mean[i]))
    return dict(best=best,members=members,simultaneous_radius=radius,chosen_by_lowest_risk=chosen,risk_best=int(risk_mean.argmin()),risk_in_set=int(risk_mean.argmin()) in members,total=len(mean))

def analyze(r,c):
    discovery=confidence_set(r[:5],c[:5]);confirmation=confidence_set(r[5:],c[5:],411);full=confidence_set(r,c,412)
    a,b=set(discovery['members']),set(confirmation['members']);jaccard=len(a&b)/len(a|b)
    risk_excluded=not discovery['risk_in_set'] and not confirmation['risk_in_set']
    selective=max(len(a),len(b))<=discovery['total']/2
    validated=discovery['chosen_by_lowest_risk'] in b
    return dict(discovery=discovery,confirmation=confirmation,full=full,jaccard=jaccard,risk_excluded_both=risk_excluded,
        selective=selective,discovery_choice_confirmed=validated,eligible=jaccard>=.5 and selective and validated and risk_excluded)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(4);start=time.time();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    write(out/'plan.json',dict(source_only=True,bootstrap_repeats=2000,confidence=.95,uncertainty_scope='resplit variability conditional on existing bank and data; not population confidence',
        policy='select lowest-risk assignment in near-optimal set',gate='discovery/confirmation Jaccard >=0.5; each set <=half feasible supports; discovery policy choice belongs to confirmation set; Risk optimum excluded in both; >=8/12 banks'))
    source=json.loads((Path(args.source)/'report.json').read_text());banks=[]
    for bank in source['banks']:
        r=torch.tensor([x['risk'] for x in bank['rows']]);result=dict(env=bank['env'],seed=bank['seed'])
        for criterion,key in [('Grad','angular'),('Regret','regret')]:
            c=torch.tensor([x[key] for x in bank['rows']]);result[criterion]=analyze(r,c)
        banks.append(result);write(out/f'env{bank["env"]}_seed{bank["seed"]}.json',result)
    counts={k:sum(b[k]['eligible'] for b in banks) for k in ['Grad','Regret']}
    write(out/'report.json',dict(banks=banks,eligible_counts=counts,passing_criteria=[k for k,v in counts.items() if v>=8],elapsed_seconds=time.time()-start,target_used=False,new_full_runs=0))
    print(json.dumps(dict(eligible_counts=counts,elapsed_seconds=time.time()-start)))

if __name__=='__main__':main()
