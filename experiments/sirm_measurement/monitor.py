"""Read-only suite audit plus a separate live status artifact."""
import argparse,json,os,statistics,time,datetime
from pathlib import Path
from experiments.sirm_measurement.run import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--banks',required=True);args=p.parse_args()
    root=Path(args.suite);progress=json.loads((root/'progress.json').read_text());plan=json.loads((root/'plan.json').read_text())
    finished=list(root.glob('*/env*/seed*/completed.json'))
    pairs=[];problems=[]
    for env in range(4):
        for seed in range(3):
            manifests=[]
            for c in ['Risk','Grad']:
                path=root/c/f'env{env}'/f'seed{seed}';mp=path/'manifest.json'
                if not mp.exists():continue
                m=json.loads(mp.read_text());manifests.append((c,m))
                q=m['config']['locked_q']
                if not all(sum(v>0 for v in row)==2 and abs(sum(row)-1)<1e-8 for row in q):problems.append(f'invalid Q {path}')
                if not all(sum(q[d][j]>0 for d in range(3)) in [0,2,3] for j in range(6)):problems.append(f'invalid sharing {path}')
                if m['target_read_during_training'] or m['steps']!=5001 or m['smoke']:problems.append(f'protocol violation {path}')
                assignment=path/'assignment_log.jsonl'
                if assignment.exists() and any(json.loads(line)['q']!=q for line in assignment.read_text().splitlines()):problems.append(f'unlocked Q {path}')
            if len(manifests)==2:
                a,b=[m for _,m in manifests]
                match=all(a[k]==b[k] for k in ['initialization_sha256','schedule_sha256','split_hash'])
                if not match:problems.append(f'unmatched env{env}/seed{seed}')
                pairs.append(dict(env=env,seed=seed,matched=match))
    priors=[json.loads(p.read_text())['seconds'] for p in (Path(args.banks)/'native').glob('env*/seed*/summary.json')]
    seconds_per_run=statistics.median(priors)+180
    current=progress.get('current');current_log=None
    if isinstance(current,dict):
        path=root/current['criterion']/f'env{current["env"]}'/f'seed{current["seed"]}'/'train_log.jsonl'
        if path.exists():
            rows=[json.loads(x) for x in path.read_text().splitlines()];current_log=rows[-1]
            if len(rows)>=2:
                rate=(rows[-1]['seconds']-rows[0]['seconds'])/(rows[-1]['step']-rows[0]['step'])
                seconds_per_run=rate*5001+180
    remaining=max(0,(24-len(finished))*seconds_per_run-(current_log['seconds'] if current_log else 0))
    status=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),suite_status=progress['status'],
        completed_artifacts=len(finished),total=24,current=current,last_checkpoint_step=current_log['step'] if current_log else None,
        matched_pairs_audited=pairs,protocol_problems=problems,estimated_remaining_hours=remaining/3600,eta_basis='observed current training rate if >=2 checkpoints, otherwise native pilot median; add estimated180s per run for postprocessing',
        source_dod_passed=plan['source_gate']['passed'])
    write(root/'live_status.json',status);print(json.dumps(status,indent=2))
    if problems:raise RuntimeError('suite protocol audit failed')

if __name__=='__main__':main()
