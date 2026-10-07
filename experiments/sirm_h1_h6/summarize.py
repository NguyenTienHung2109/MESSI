"""Aggregate H1-H6 metrics and paired training-seed deltas across a suite."""
import argparse
import json
from pathlib import Path
import numpy as np


def flatten(data,prefix=''):
    result={}
    for key,value in data.items():
        name=f'{prefix}/{key}' if prefix else key
        if isinstance(value,dict):result.update(flatten(value,name))
        elif isinstance(value,(int,float)) and not isinstance(value,bool):result[name]=float(value)
    return result


def stats(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return dict(n=0,mean=None,sd=None)
    return dict(n=len(values),mean=float(values.mean()),sd=float(values.std(ddof=1)) if len(values)>1 else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--include-smoke',action='store_true')
    args=p.parse_args()
    groups={}
    for path in sorted(Path(args.root).glob('**/target_metrics.json')):
        data=json.loads(path.read_text());meta=data['metadata']
        if meta['smoke'] and not args.include_smoke:continue
        variant,env,seed=meta['variant'],meta['target_env'],meta['seed']
        rows=groups.setdefault(variant,{})
        if (env,seed) in rows:raise ValueError(f'duplicate run: {variant},env{env},seed{seed}')
        manifest=json.loads((path.parent.parent/'manifest.json').read_text())
        rows[env,seed]=dict(metadata=meta,manifest=manifest,values=flatten({k:data[k] for k in ['controls','H2','H3','H6','paired_vs_global','input_vs_shuffled_or_fixed']}))
    summary={}
    for variant,rows in groups.items():
        metrics=sorted(set.intersection(*(set(row['values']) for row in rows.values())))
        seeds=sorted({seed for env,seed in rows})
        complete=[seed for seed in seeds if all((env,seed) in rows for env in range(4))]
        summary[variant]=dict(complete_four_target_seeds=complete,
                             pilot_complete=all(seed in complete for seed in range(3)),
                             confirm_complete=all(seed in complete for seed in range(5)),metrics={})
        for name in metrics:
            summary[variant]['metrics'][name]=dict(
                per_target={str(env):stats([row['values'][name] for (e,s),row in rows.items() if e==env]) for env in range(4)},
                mean4_across_seeds=stats([np.mean([rows[env,seed]['values'][name] for env in range(4)]) for seed in complete]))
    paired={}
    if 'native' in groups:
        baseline=groups['native']
        for variant,rows in groups.items():
            if variant=='native':continue
            common=set(rows)&set(baseline)
            deltas={}
            for key in common:
                first,second=rows[key],baseline[key]
                for field in ['split_hash','initialization_sha256','schedule_sha256']:
                    if first['manifest'][field]!=second['manifest'][field]:
                        raise ValueError(f'unpaired runs {variant}, {key}: {field}')
                deltas[key]={metric:first['values'][f'controls/native/{metric}']-second['values'][f'controls/native/{metric}']
                             for metric in ['accuracy','nll','routing_regret']}
            complete=[seed for seed in range(5) if all((env,seed) in common for env in range(4))]
            paired[variant]={metric:dict(per_target={str(env):stats([row[metric] for (e,s),row in deltas.items() if e==env]) for env in range(4)},
                mean4_seed_delta=stats([np.mean([deltas[env,seed][metric] for env in range(4)]) for seed in complete])) for metric in ['accuracy','nll','routing_regret']}
    result=dict(variants=summary,paired_vs_native=paired,smoke_included=args.include_smoke,
                uncertainty='Training uncertainty is SD over paired seeds; per-image bootstrap intervals in individual evaluations are not training-run confidence intervals.')
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'aggregated {sum(len(rows) for rows in groups.values())} runs: {args.output}')


if __name__=='__main__':main()
