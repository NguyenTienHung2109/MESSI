"""Serial, locked 12-run controller; target evaluation follows source selection."""
import argparse,copy,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from .subsets import ROOT,RESULT,MANIFEST,verify

NAMES=['L100','L38','L43','L46'];METHODS=['ERM_H2','OFF','ON']
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def params(target,method,bank,smoke=False):
    hp=copy.deepcopy(json.loads((ROOT/'results/sirm_discover_binary/terra_L38_seed0_lambda1/plan.json').read_text())['config']['R1'])
    hp.update(sirm_output_dir=str(bank),sirm_source_envs=[i for i in range(4) if i!=target],diagnostic_subset_manifest=str(MANIFEST),diagnostic_defer_target_eval=True)
    c=hp['sirm_discover_config'];c.update(protocol='terra_10pct_factor_diagnosis_501steps',global_steps=100,discover_steps=100,invariance_steps=301,utility_interval=5,sirm_ramp_steps=50,checkpoint_freq=25,diagnostic_steps=[199,200,350,500]);c['lambda_sirm']=1. if method=='ON' else 0.
    if smoke:c.update(global_steps=1,discover_steps=1,invariance_steps=1)
    return hp

def command(target,method,bank,smoke=False):
    hp=params(target,method,bank,smoke)
    return [sys.executable,'-u','-m','domainbed.scripts.train','--dataset','TerraIncognita','--algorithm','ERMH2Diagnostic' if method=='ERM_H2' else 'SIRMDiscoverV2','--test_envs',str(target),'--source_envs',*[str(e) for e in hp['sirm_source_envs']],'--seed','0','--trial_seed','0','--hparams_seed','0','--steps',str(sum(hp['sirm_discover_config'][k] for k in ['global_steps','discover_steps','invariance_steps'])),'--checkpoint_freq','1' if smoke else str(hp['sirm_discover_config']['checkpoint_freq']),'--num_workers','2','--eval_num_workers','2','--eval_batch_size','64','--data_dir',str(ROOT/'domainbed/data'),'--output_dir',str(bank),'--hparams',json.dumps(hp)]

def hashes():
    paths=list((ROOT/'experiments/terra_10pct_factor_diagnosis').glob('*.py'))+[ROOT/p for p in ['domainbed/scripts/train.py','domainbed/algorithms.py','domainbed/datasets.py','domainbed/sirm_discover.py','experiments/sirm_discover_gate/model.py','experiments/sirm_discover/model.py','experiments/sirm_discover/math.py','experiments/sirm_h1_h6/run.py','experiments/sirm_discover_gate/metrics.py','experiments/sirm_discover_gate/data.py','experiments/sirm_discover_binary/run.py','experiments/sirm_discover_binary_analysis/run.py','domainbed/deit_transformer.py','domainbed/scripts/train_subset_irm_pacs.py','domainbed/lib/misc.py']]
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}

def main(smoke=False):
    manifest=json.loads(MANIFEST.read_text());verify(manifest)
    root=RESULT/'smoke' if smoke else RESULT;root.mkdir(parents=True,exist_ok=True)
    jobs=[(1,m) for m in METHODS] if smoke else [(t,m) for t in range(4) for m in METHODS]
    lock=dict(seed=0,manifest_sha256=digest(MANIFEST),code_hashes=hashes(),target_evaluation='post-source-selection only',selection_window='steps 200..500, cadence 25, earliest best equal-source-domain out accuracy; same window for all methods',jobs=[dict(target=NAMES[t],method=m,hparams=params(t,m,root/'runs'/NAMES[t]/m,smoke)) for t,m in jobs])
    plan=root/'plan.json'
    if plan.exists():assert json.loads(plan.read_text())==lock,'locked code/config changed; refusing mixed suite'
    else:
        write(plan,lock)
        for name in lock['code_hashes']:
            dest=root/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    started=time.time()
    for j,(target,method) in enumerate(jobs):
        assert hashes()==lock['code_hashes'] and digest(MANIFEST)==lock['manifest_sha256']
        bank=root/'runs'/NAMES[target]/method;bank.mkdir(parents=True,exist_ok=True)
        if not (bank/'done').exists():
            assert not (bank/'diagnostics.jsonl').exists(),'Partial run requires explicit recovery; refusing overwrite'
            with (bank/'console.log').open('w') as log:
                proc=subprocess.Popen(command(target,method,bank,smoke),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                while proc.poll() is None:
                    step=None;p=bank/'diagnostics.jsonl'
                    if p.exists():
                        lines=p.read_text().splitlines()
                        for line in reversed(lines[-2:]):
                            try:step=json.loads(line)['step'];break
                            except json.JSONDecodeError:pass
                    write(root/'progress.json',dict(status='training',completed=j,total=len(jobs),target=NAMES[target],method=method,pid=proc.pid,step=step,elapsed_seconds=time.time()-started))
                    time.sleep(10)
                if proc.returncode:
                    write(root/'progress.json',dict(status='failed',target=NAMES[target],method=method,returncode=proc.returncode));raise RuntimeError(f'{bank}: training failed; see console.log')
        from .report import audit_training
        audit_training(bank,target)
        if method=='ON':
            from .report import audit_controls
            audit_controls(root/'runs'/NAMES[target],smoke)
        if not smoke:
            write(root/'progress.json',dict(status='posthoc',completed=j,total=len(jobs),target=NAMES[target],method=method))
            from .report import analyze,aggregate
            analyze(bank,target,method,manifest);aggregate(root,manifest)
    if smoke:
        schedules=[json.loads((root/'runs/L38'/m/'source_schedules.json').read_text()) for m in METHODS];assert schedules[0]==schedules[1]==schedules[2]
        write(root/'verification.json',dict(passed=True,three_methods_trained=True,identical_source_schedules=True,target_scores_absent_during_training=True,source_selected_checkpoints_saved=True))
    write(root/'progress.json',dict(status='completed',completed=len(jobs),total=len(jobs),elapsed_seconds=time.time()-started))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    try:main(args.smoke)
    except Exception as exc:
        root=RESULT/'smoke' if args.smoke else RESULT
        progress=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else {}
        write(root/'progress.json',dict(progress,status='failed',error=str(exc)));raise
