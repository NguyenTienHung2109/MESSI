"""Protocol correction: paired residuals on source-selected completed ERM run."""
import json,os,shutil,subprocess,time
from pathlib import Path
from experiments.terra_10pct_factor_diagnosis.run import ROOT,RESULT,MANIFEST,params,command,hashes,digest,write
from .router_weight import live
from .frozen_erm_audit import audit_anchor,audit_pair

DEST=ROOT/'results/sirm_research_loop/iteration_04'
EXPERIMENT=DEST/'experiment'
ANCHOR=RESULT/'runs/L38/ERM_H2/source_best.pkl'

def configuration(method,bank):
    hp=params(1,method,bank);c=hp['sirm_discover_config']
    c.update(protocol='sirm_research_iter04_source_selected_frozen_erm400',strong_anchor_checkpoint=str(ANCHOR),strong_anchor_sha256=digest(ANCHOR),strong_anchor_load_at_start=True,global_steps=0,discover_steps=200,invariance_steps=200,diagnostic_steps=[0,199,200,399])
    return hp

def invocation(method,bank):
    cmd=command(1,method,bank);cmd[cmd.index('domainbed.scripts.train')]='experiments.sirm_research_loop.anchor_train'
    cmd[cmd.index('--steps')+1]='400';cmd[cmd.index('--hparams')+1]=json.dumps(configuration(method,bank));return cmd

def main():
    assert (DEST/'preregistration.md').exists()
    source=read(RESULT/'plan.json');assert hashes()==source['code_hashes']
    erm=ANCHOR.parent;selection=read(erm/'selection.json');report=read(erm/'posthoc/complete.json')
    assert (erm/'done').exists() and not selection['target_used']
    assert report['checkpoint_sha256']==digest(ANCHOR) and report['selected_step']==selection['selected_step']
    logs=[json.loads(x) for x in (erm/'diagnostics.jsonl').read_text().splitlines()]
    assert [x['step'] for x in logs]==list(range(501))
    variant=[Path(__file__),Path(__file__).with_name('anchor_train.py'),Path(__file__).with_name('frozen_erm_audit.py')]
    plan=dict(anchor_sha256=digest(ANCHOR),anchor_selection=selection,anchor_training_budget_updates=501,base_code_hashes=hashes(),variant_code_hashes={str(p.relative_to(ROOT)):digest(p) for p in variant},manifest_sha256=digest(MANIFEST),hparams={m:configuration(m,EXPERIMENT/'runs/L38'/m) for m in ['OFF','ON']},target='L38',seed=0,conceptual_change='only common ERM checkpoint selection policy corrected; original residual/SIRM settings retained')
    if (DEST/'plan.json').exists():assert read(DEST/'plan.json')==plan
    else:
        write(DEST/'plan.json',plan)
        for name in {**plan['base_code_hashes'],**plan['variant_code_hashes']}:
            p=DEST/'code_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    def check():
        assert digest(ANCHOR)==plan['anchor_sha256'] and digest(MANIFEST)==plan['manifest_sha256'] and hashes()==plan['base_code_hashes']
        assert all(digest(ROOT/p)==h for p,h in plan['variant_code_hashes'].items())
    # Verify the prior run is terminal, rather than assuming its state file suffices.
    prior=ROOT/'results/sirm_research_loop/iteration_03';pid=read(prior/'worker_pid.json')['pid']
    assert not live(pid) and read(prior/'progress.json')['status']=='paired_training_complete'
    check();EXPERIMENT.mkdir(exist_ok=True);manifest=read(MANIFEST);write(EXPERIMENT/'subset_manifest.json',manifest)
    trio=EXPERIMENT/'runs/L38';b0=trio/'ERM_H2';(b0/'posthoc').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ANCHOR,b0/'source_best.pkl');write(b0/'posthoc/complete.json',report)
    write(b0/'posthoc/selection_lock.json',dict(checkpoint_sha256=digest(ANCHOR),selection=selection,completed_training_updates=501,retraining=False,target_used_for_selection=False))
    for method in ['OFF','ON']:
        check();bank=trio/method;bank.mkdir(exist_ok=True)
        if not (bank/'done').exists():
            assert not (bank/'diagnostics.jsonl').exists(),'refusing overwrite of partial run'
            with (bank/'console.log').open('w') as f:
                proc=subprocess.Popen(invocation(method,bank),cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    p=bank/'diagnostics.jsonl';step=None
                    if p.exists():
                        for line in reversed(p.read_text().splitlines()[-2:]):
                            try:step=json.loads(line)['step'];break
                            except ValueError:pass
                    write(DEST/'progress.json',dict(status='training',method=method,pid=proc.pid,step=step,observed_unix=time.time()));time.sleep(10)
                assert proc.returncode==0,f'{method} failed; inspect console.log'
        from experiments.terra_10pct_factor_diagnosis.report import audit_training,analyze
        audit_training(bank,1);audit_anchor(bank,ANCHOR)
        if method=='ON':audit_pair(trio)
        write(DEST/'progress.json',dict(status='posthoc',method=method));analyze(bank,1,method,manifest)
        score=read(bank/'posthoc/complete.json')
        assert all(report['splits'][s]['source_mean']==score['splits'][s]['global_mean'] for s in ['source_out','target_in','target_out'])
    check();write(DEST/'progress.json',dict(status='paired_training_complete',remaining='case EDA, bias/shrinkage controls, utility reliability, required report and audit'))

def read(p):return json.loads(p.read_text())

if __name__=='__main__':
    try:main()
    except Exception as exc:write(DEST/'progress.json',dict(status='failed',error=str(exc)));raise
