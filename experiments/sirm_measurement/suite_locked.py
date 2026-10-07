"""Full matched Risk/Grad suite gated by the locked source-only DoD report."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from experiments.sirm_measurement.run import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--banks',required=True);p.add_argument('--scores',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    report=json.loads((Path(a.scores)/'report.json').read_text())
    if not report['passed'] or report['eligible_count']<8:raise RuntimeError('source DoD not met')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    write(out/'plan.json',dict(full_runs=24,criteria=['Risk','Grad'],seeds=[0,1,2],targets=[0,1,2,3],steps=5001,
        stage='matched source-prior bank fine-tuning with fixed hard assignment; not dynamic original H6',selection='source-out only',assignment_criterion='angular Grad alpha1 averaged discovery80; confirmation80 gate only',
        statistical_scope='exploratory; pilot target already observed; paired3seed mean4 intervals',source_gate=report))
    results=[]
    for env in range(4):
        for seed in range(3):
            for criterion in ['Risk','Grad']:
                dest=out/criterion/f'env{env}'/f'seed{seed}';dest.parent.mkdir(parents=True,exist_ok=True)
                cmd=[sys.executable,'-u','-m','experiments.sirm_measurement.train_locked','--banks',a.banks,'--scores',a.scores,'--output',str(dest),'--criterion',criterion,'--target-env',str(env),'--seed',str(seed)]
                write(out/'progress.json',dict(status='running',completed=len(results),total=24,current=dict(env=env,seed=seed,criterion=criterion),elapsed_seconds=time.time()-start))
                print(f'start {len(results)+1}/24 {criterion} env{env} seed{seed}',flush=True)
                with dest.with_suffix('.log').open('w') as stream:
                    child=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
                if child.returncode:
                    write(out/'progress.json',dict(status='failed',completed=len(results),current=str(dest),returncode=child.returncode));raise RuntimeError(f'failed: {dest}')
                metrics=json.loads((dest/'evaluation/target_metrics.json').read_text())['controls']['native']
                results.append(dict(env=env,seed=seed,criterion=criterion,metrics=metrics));write(out/'results_partial.json',results)
                print(f'completed {len(results)}/24 {criterion} env{env} seed{seed}',flush=True)
    paired=[]
    for seed in range(3):
        values={c:[r['metrics']['accuracy'] for r in results if r['seed']==seed and r['criterion']==c] for c in ['Risk','Grad']}
        paired.append(sum(values['Grad'])/4-sum(values['Risk'])/4)
    import statistics,math
    mean=statistics.mean(paired);sd=statistics.stdev(paired);margin=4.302652729911275*sd/math.sqrt(3)
    write(out/'report.json',dict(results=results,paired_seed_accuracy_delta=paired,accuracy_delta_mean=mean,accuracy_delta_ci95=[mean-margin,mean+margin],elapsed_seconds=time.time()-start))
    write(out/'progress.json',dict(status='completed',completed=24,total=24,elapsed_seconds=time.time()-start))

if __name__=='__main__':main()
