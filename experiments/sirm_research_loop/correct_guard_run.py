"""Locked matched400-step source-only trajectory test of native discovery guard."""
import copy,json,os,shutil,subprocess,time
from pathlib import Path
from .routed_discovery_run import ROOT,DEST as PRIOR,ANCHOR,CACHE,MANIFEST,command
from .shrinkage import digest,write
from .frozen_erm_audit import audit_anchor,audit_pair
DEST=PRIOR.parent/'iteration_16';TRIO=DEST/'experiment/runs/L38'
def configuration(arm,bank):
 p=json.loads((PRIOR/'plan.json').read_text());hp=copy.deepcopy(p['hparams'][arm]);hp['sirm_output_dir']=str(bank);hp['sirm_discover_config']['protocol']='sirm_iter16_native_correct_CE_guard';return hp
def invocation(arm,bank,steps=400):
 cmd=command(1,arm,bank);cmd[cmd.index('domainbed.scripts.train')]='experiments.sirm_research_loop.correct_guard_train';cmd[cmd.index('--steps')+1]=str(steps);cmd[cmd.index('--hparams')+1]=json.dumps(configuration(arm,bank));return cmd
def main():
 assert json.loads((DEST/'prelaunch_gradient_audit.json').read_text())['passed'];assert json.loads((DEST/'solver_validation.json').read_text())['passed']
 previous=json.loads((PRIOR/'final_analysis_audit.json').read_text())
 for f,h in previous['sha256'].items():assert digest(PRIOR/f)==h,f
 old=json.loads((PRIOR/'plan.json').read_text());assert digest(ANCHOR)==old['anchor_sha256'] and digest(MANIFEST)==old['manifest_sha256'] and digest(CACHE)==old['feature_cache_sha256']
 for f,h in {**old['base_code_hashes'],**old['variant_code_hashes']}.items():assert digest(ROOT/f)==h,f
 variants=['correct_guard_model.py','native_correct_guard.py','correct_guard_train.py','correct_guard_run.py','repair_margin_guard_helpers.py','native_guard_projection.py','discovery_error_gradient_diagnosis.py']
 locked={**old['base_code_hashes'],**old['variant_code_hashes'],**{str(Path(__file__).with_name(f).relative_to(ROOT)):digest(Path(__file__).with_name(f)) for f in variants}}
 plan=dict(anchor_sha256=digest(ANCHOR),manifest_sha256=digest(MANIFEST),feature_cache_sha256=digest(CACHE),code_sha256=locked,hparams={a:configuration(a,TRIO/a) for a in ['OFF','ON']},target=1,seed=0,source_only_training=True,target_selection=False,steps=400,discovery_steps=200,invariance_steps=200,guard_minimum_samples=2,guard_refresh='every optimizer update',hypothesis='Source-correct domain/class mean-CE gradient guard separates harmful and useful native discovery directions, allowing complementary source gains and shared Q. A local EDA signal may fail over an Adam trajectory.',changed='Only residual discovery gradient is projected onto source-in global-correct cohort CE halfspaces then restored to original gradient norm. Scalar loss preserved with explicit gradient surrogate. Classification/router/balance/SIRM and configuration otherwise unchanged. Guards use current residual and detached current native router, differentiate full-source equal-domain centering.',source_evidence_sha256=digest(PRIOR/'native_discovery_correct_guard/interpretation.md'),target_promotion_rule='Inspect source utility/Q/held-domain errors before any new target replay. No accuracy threshold tuning.',scientific_goal_achieved=False)
 assert not (DEST/'plan.json').exists();write(DEST/'plan.json',plan)
 for f in locked:
  target=DEST/'code_snapshot'/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/f).read_bytes())
 (DEST/'preregistration.md').write_text('Native correct-cohort CE guard from completed source-only EDA. Matched OFF/ON400-step schedule, fixed10%manifest/seed0/strongERM anchor; source-only earliest-best checkpoint cadence unchanged. Residual discovery gradient projection and norm restoration are the only intervention; scalar loss value is unchanged but its gradient intentionally changes. Minimum cohort2. No target tuning or promotion before source analysis. All code/config hashes locked in plan.json. This is a trajectory falsification test, not a claim of invariant-factor benefit.\n')
 manifest=json.loads(MANIFEST.read_text());(DEST/'experiment').mkdir(exist_ok=True);write(DEST/'experiment/subset_manifest.json',manifest)
 base=TRIO/'ERM_H2';(base/'posthoc').mkdir(parents=True,exist_ok=True);shutil.copyfile(ANCHOR,base/'source_best.pkl');shutil.copyfile(ANCHOR.parent/'posthoc/complete.json',base/'posthoc/complete.json')
 env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 for arm in ['OFF','ON']:
  for f,h in locked.items():assert digest(ROOT/f)==h
  bank=TRIO/arm;bank.mkdir(parents=True,exist_ok=True);assert not (bank/'diagnostics.jsonl').exists()
  with (bank/'console.log').open('w') as log:
   proc=subprocess.Popen(invocation(arm,bank),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   while proc.poll() is None:
    rows=bank/'diagnostics.jsonl';step=None
    if rows.exists():
     for line in reversed(rows.read_text().splitlines()[-2:]):
      try:step=json.loads(line)['step'];break
      except ValueError:pass
    write(DEST/'progress.json',dict(status='training',arm=arm,pid=proc.pid,step=step,observed_unix=time.time()));time.sleep(5)
   assert proc.returncode==0,f'{arm}failed; inspect console.log'
  audit_anchor(bank,ANCHOR)
  rows=[json.loads(l) for l in (bank/'diagnostics.jsonl').read_text().splitlines()];assert [r['step'] for r in rows]==list(range(400))
 audit_pair(TRIO)
 initials=[json.loads((TRIO/a/'spatial_initialization.json').read_text()) for a in ['OFF','ON']];assert len({r['parameter_sha256'] for r in initials})==1;assert initials[0]['parameter_sha256']==json.loads((PRIOR/'experiment/runs/L38/OFF/spatial_initialization.json').read_text())['parameter_sha256']
 schedules=[json.loads((TRIO/a/'source_schedules.json').read_text()) for a in ['OFF','ON']];assert schedules[0]==schedules[1]==json.loads((PRIOR/'experiment/runs/L38/OFF/source_schedules.json').read_text())
 write(DEST/'training_audit.json',dict(passed=True,exact_anchor=True,exact_initialization=True,exact_source_schedules=True,contiguous_400_updates=True,target_reads=False))
 write(DEST/'progress.json',dict(status='training_complete',remaining='Source utility/Q/guard trajectory and failed-case analysis before target promotion'))
if __name__=='__main__':
 write(DEST/'worker_pid.json',dict(pid=os.getpid(),started_unix=time.time()))
 try:main()
 except Exception as e:write(DEST/'progress.json',dict(status='failed',error=str(e)));raise
