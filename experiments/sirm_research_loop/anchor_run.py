"""Serial paired strong-anchor experiment, preregistered iteration3."""
import json,os,subprocess,time
from pathlib import Path
from experiments.terra_10pct_factor_diagnosis.run import ROOT,RESULT,MANIFEST,params,command,hashes,digest,write
from .router_weight import live

DEST=ROOT/'results/sirm_research_loop/iteration_03'
EXPERIMENT=DEST/'experiment'
ANCHOR=RESULT/'runs/L38/ERM_H2/last.pkl'

def configuration(method,bank):
    hp=params(1,method,bank);c=hp['sirm_discover_config']
    c.update(protocol='sirm_research_iter03_frozen_erm400',strong_anchor_checkpoint=str(ANCHOR),strong_anchor_sha256=digest(ANCHOR),strong_anchor_load_at_start=True,global_steps=0,discover_steps=200,invariance_steps=200,diagnostic_steps=[0,199,200,399])
    return hp

def invocation(method,bank):
    cmd=command(1,method,bank);cmd[cmd.index('domainbed.scripts.train')]='experiments.sirm_research_loop.anchor_train'
    cmd[cmd.index('--steps')+1]='400'
    cmd[cmd.index('--hparams')+1]=json.dumps(configuration(method,bank));return cmd

def main():
    assert (DEST/'preregistration.md').exists()
    source=json.loads((RESULT/'plan.json').read_text());assert hashes()==source['code_hashes']
    files=[Path(__file__),Path(__file__).with_name('anchor_train.py'),Path(__file__).with_name('frozen_erm_audit.py')]
    plan=dict(manifest_sha256=digest(MANIFEST),base_code_hashes=hashes(),variant_code_hashes={str(p.relative_to(ROOT)):digest(p) for p in files},anchor_sha256=digest(ANCHOR),hparams={m:configuration(m,EXPERIMENT/'runs/L38'/m) for m in ['OFF','ON']},selection='source-only earliest Stage3 best; same cadence/window',target='L38',seed=0)
    if (DEST/'plan.json').exists():assert json.loads((DEST/'plan.json').read_text())==plan
    else:
        write(DEST/'plan.json',plan)
        for name in {**plan['base_code_hashes'],**plan['variant_code_hashes']}:
            p=DEST/'code_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    def check():
        assert hashes()==plan['base_code_hashes'] and digest(MANIFEST)==plan['manifest_sha256'] and digest(ANCHOR)==plan['anchor_sha256']
        assert all(digest(ROOT/p)==sha for p,sha in plan['variant_code_hashes'].items())
    pid=json.loads((RESULT/'controller_pid.json').read_text())['pid']
    while live(pid):
        check();write(DEST/'progress.json',dict(status='waiting_for_baseline_GPU',controller_pid=pid,controller_live=True,observed_unix=time.time()));time.sleep(20)
    assert json.loads((RESULT/'progress.json').read_text())['status']=='completed'
    check();EXPERIMENT.mkdir(parents=True,exist_ok=True);manifest=json.loads(MANIFEST.read_text());write(EXPERIMENT/'subset_manifest.json',manifest)
    trio=EXPERIMENT/'runs/L38';trio.mkdir(parents=True,exist_ok=True)
    from .frozen_erm_audit import evaluate_B0,audit_pair,audit_anchor
    evaluate_B0(EXPERIMENT,ANCHOR,manifest)
    erm=trio/'ERM_H2'
    for method in ['OFF','ON']:
        check();bank=trio/method;bank.mkdir(exist_ok=True)
        if not (bank/'done').exists():
            assert not (bank/'diagnostics.jsonl').exists(),'refusing overwrite of partial run'
            with (bank/'console.log').open('w') as log:
                proc=subprocess.Popen(invocation(method,bank),cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    step=None;p=bank/'diagnostics.jsonl'
                    if p.exists():
                        for line in reversed(p.read_text().splitlines()[-2:]):
                            try:step=json.loads(line)['step'];break
                            except ValueError:pass
                    write(DEST/'progress.json',dict(status='training',method=method,pid=proc.pid,step=step,observed_unix=time.time()));time.sleep(10)
                assert proc.returncode==0,f'{method} failed; see console.log'
        from experiments.terra_10pct_factor_diagnosis.report import audit_training,analyze,aggregate
        audit_training(bank,1)
        audit_anchor(bank,ANCHOR)
        if method=='ON':audit_pair(trio)
        write(DEST/'progress.json',dict(status='posthoc',method=method));analyze(bank,1,method,manifest)
        for split in ['source_out','target_in','target_out']:
            old=json.loads((erm/'posthoc/complete.json').read_text())['splits'][split]['source_mean']
            new=json.loads((bank/'posthoc/complete.json').read_text())['splits'][split]['global_mean']
            assert old==new,'strong global replay does not match ERM'
    aggregate(EXPERIMENT,manifest);check()
    write(DEST/'progress.json',dict(status='paired_training_complete',remaining='tensor anchor audit, factor/transfer/SIRM diagnostics, shrinkage control and verdict'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(DEST/'progress.json',dict(status='failed',error=str(exc)));raise
