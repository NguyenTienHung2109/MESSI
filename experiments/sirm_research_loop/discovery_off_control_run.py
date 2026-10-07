"""One source-only auxiliary ablation; does not replace discovery/SIRM OFF-ON pair."""
import json,os,subprocess,time,copy
from pathlib import Path
from experiments.terra_10pct_factor_diagnosis.run import ROOT,command,digest,write
from .frozen_erm_audit import audit_anchor

DEST=ROOT/'results/sirm_research_loop/iteration_11/discovery_off_control'
PRIOR=DEST.parent

def read(path):return json.loads(path.read_text())

def main():
    assert (DEST/'preregistration.md').exists()
    prior=read(PRIOR/'plan.json')
    assert read(PRIOR/'completion_audit.json')['analysis_complete']
    assert read(PRIOR/'source_followup_completion_audit.json')['discovery_error_gradient_complete']
    for name,h in {**prior['base_code_hashes'],**prior['variant_code_hashes']}.items():assert digest(ROOT/name)==h
    bank=DEST/'run';bank.mkdir(exist_ok=True)
    hp=copy.deepcopy(prior['hparams']['OFF']);hp['sirm_output_dir']=str(bank)
    hp['sirm_discover_config']['lambda_discover']=0.
    assert hp['diagnostic_defer_target_eval'] and hp['sirm_discover_config']['lambda_sirm']==0
    old=prior['hparams']['OFF']['sirm_discover_config'];new=hp['sirm_discover_config']
    assert {k:v for k,v in old.items() if k!='lambda_discover'}=={k:v for k,v in new.items() if k!='lambda_discover'}
    plan=dict(driver_sha256=digest(Path(__file__)),reference_plan_sha256=digest(PRIOR/'plan.json'),
        hparams=hp,anchor_sha256=prior['anchor_sha256'],manifest_sha256=prior['manifest_sha256'],
        code_hashes={**prior['base_code_hashes'],**prior['variant_code_hashes']},
        training_updates=400,change='lambda_discover0.1->0 only; router, balance, centering and schedules retained',
        target_access=False,primary_method=False,scope='Auxiliary control for standalone discovery contribution; SIRM OFF.')
    if (DEST/'plan.json').exists():assert read(DEST/'plan.json')==plan
    else:write(DEST/'plan.json',plan)
    cmd=command(1,'OFF',bank);cmd[cmd.index('domainbed.scripts.train')]='experiments.sirm_research_loop.hard_router_cap_train'
    cmd[cmd.index('--steps')+1]='400';cmd[cmd.index('--hparams')+1]=json.dumps(hp)
    if not (bank/'done').exists():
        assert not (bank/'diagnostics.jsonl').exists(),'Do not restart an existing partial run'
        with (bank/'console.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            write(DEST/'training_pid.json',dict(pid=proc.pid,started_unix=time.time()))
            while proc.poll() is None:
                step=None;p=bank/'diagnostics.jsonl'
                if p.exists():
                    for line in reversed(p.read_text().splitlines()[-2:]):
                        try:step=json.loads(line)['step'];break
                        except ValueError:pass
                write(DEST/'progress.json',dict(status='training',pid=proc.pid,step=step,observed_unix=time.time()));time.sleep(10)
            assert proc.returncode==0,'inspect control console.log'
    rows=[json.loads(l) for l in (bank/'diagnostics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(400))
    assert all(r['loss_sirm']==0 for r in rows)
    reference=PRIOR/'experiment/runs/L38/OFF'
    assert read(bank/'initialization.json')==read(reference/'initialization.json')
    assert read(bank/'source_schedules.json')==read(reference/'source_schedules.json')
    for line in (bank/'results.jsonl').read_text().splitlines():
        assert not any(k.startswith('env1_') for k in json.loads(line))
    audit_anchor(bank,Path(new['strong_anchor_checkpoint']))
    assert digest(Path(__file__))==plan['driver_sha256']
    for name,h in plan['code_hashes'].items():assert digest(ROOT/name)==h
    write(DEST/'training_audit.json',dict(passed=True,updates=400,initialization_identical=True,schedules_identical=True,
        target_training_rows=0,global_frozen=True,loss_change_only='lambda_discover',selected_step=read(bank/'selection.json')['selected_step']))
    write(DEST/'progress.json',dict(status='training_complete',remaining='source-only replay and original error-case analysis'))


if __name__=='__main__':
    try:main()
    except Exception as exc:write(DEST/'progress.json',dict(status='failed',error=str(exc)));raise
