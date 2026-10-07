"""Preregistered iteration 2: one router-weight change, paired OFF/ON."""
import copy,json,os,subprocess,sys,time
from pathlib import Path
from experiments.terra_10pct_factor_diagnosis.run import params,command,hashes,digest,write,ROOT,RESULT,MANIFEST

DEST=ROOT/'results/sirm_research_loop/iteration_02'
EXPERIMENT=DEST/'experiment'

def configuration(method,bank):
    hp=params(1,method,bank)
    hp['sirm_discover_config']['lambda_router']=1.
    hp['sirm_discover_config']['protocol']='sirm_research_iter02_router_weight1'
    return hp

def invocation(method,bank):
    cmd=command(1,method,bank)
    cmd[cmd.index('--hparams')+1]=json.dumps(configuration(method,bank))
    return cmd

def live(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] not in ['Z','X']
    except FileNotFoundError:return False

def lock():
    assert (DEST/'preregistration.md').exists()
    source_plan=json.loads((RESULT/'plan.json').read_text())
    assert hashes()==source_plan['code_hashes'],'baseline training implementation changed'
    baseline={m:digest(RESULT/'runs/L38'/m/'source_best.pkl') for m in ['ERM_H2','OFF','ON']}
    plan=dict(manifest_sha256=digest(MANIFEST),code_hashes=hashes(),runner_sha256=digest(Path(__file__)),baseline_checkpoints=baseline,hparams={m:configuration(m,EXPERIMENT/'runs/L38'/m) for m in ['OFF','ON']},conceptual_change='lambda_router 0.1 -> 1; paired lambda_sirm 0/1',GPU_policy='wait for baseline controller terminal completion; serial jobs')
    if (DEST/'plan.json').exists():assert json.loads((DEST/'plan.json').read_text())==plan
    else:
        write(DEST/'plan.json',plan)
        for name in plan['code_hashes']:
            p=DEST/'code_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
        (DEST/'code_snapshot/router_weight.py').write_bytes(Path(__file__).read_bytes())
    return plan

def check(plan):
    assert hashes()==plan['code_hashes'] and digest(MANIFEST)==plan['manifest_sha256']
    for m,sha in plan['baseline_checkpoints'].items():assert digest(RESULT/'runs/L38'/m/'source_best.pkl')==sha

def main():
    plan=lock();pid=json.loads((RESULT/'controller_pid.json').read_text())['pid']
    # A state file alone never proves that training has stopped.
    while live(pid):
        check(plan)
        write(DEST/'progress.json',dict(status='waiting_for_baseline_GPU',baseline_controller_pid=pid,baseline_controller_live=True,baseline_progress=json.loads((RESULT/'progress.json').read_text()),observed_unix=time.time()))
        time.sleep(20)
    status=json.loads((RESULT/'progress.json').read_text())
    assert status['status']=='completed','baseline controller stopped without completing: inspect before launching'
    check(plan)
    EXPERIMENT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(MANIFEST.read_text());write(EXPERIMENT/'subset_manifest.json',manifest)
    trio=EXPERIMENT/'runs/L38';trio.mkdir(parents=True,exist_ok=True)
    erm=trio/'ERM_H2'
    if not erm.exists():erm.symlink_to(RESULT/'runs/L38/ERM_H2',target_is_directory=True)
    assert erm.resolve()==(RESULT/'runs/L38/ERM_H2').resolve()
    for method in ['OFF','ON']:
        check(plan);bank=trio/method;bank.mkdir(exist_ok=True)
        if not (bank/'done').exists():
            assert not (bank/'diagnostics.jsonl').exists(),'refusing to overwrite partial training'
            with (bank/'console.log').open('w') as log:
                proc=subprocess.Popen(invocation(method,bank),cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    step=None;p=bank/'diagnostics.jsonl'
                    if p.exists():
                        for line in reversed(p.read_text().splitlines()[-2:]):
                            try:step=json.loads(line)['step'];break
                            except ValueError:pass
                    write(DEST/'progress.json',dict(status='training',method=method,pid=proc.pid,step=step,observed_unix=time.time()))
                    time.sleep(10)
                assert proc.returncode==0,f'{method} failed: console.log'
        from experiments.terra_10pct_factor_diagnosis.report import audit_training,analyze,audit_controls
        audit_training(bank,1)
        if method=='ON':audit_controls(trio)
        write(DEST/'progress.json',dict(status='posthoc',method=method))
        analyze(bank,1,method,manifest)
    from experiments.terra_10pct_factor_diagnosis.report import aggregate
    aggregate(EXPERIMENT,manifest)
    check(plan)
    write(DEST/'progress.json',dict(status='paired_training_complete',remaining='source alignment audit, shrinkage controls, mechanism analysis and iteration verdict'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(DEST/'progress.json',dict(status='failed',error=str(exc)));raise
