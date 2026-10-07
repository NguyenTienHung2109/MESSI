"""Matched400-update objective-only actual-mixture router falsification."""
import copy,json,os,shutil,subprocess,time
from pathlib import Path
from .routed_discovery_run import ROOT,ANCHOR,CACHE,MANIFEST,command
from .shrinkage import digest,write
from .frozen_erm_audit import audit_anchor,audit_pair
PRIOR=ROOT/'results/sirm_research_loop/iteration_16';DEST=PRIOR.parent/'iteration_17';TRIO=DEST/'experiment/runs/L38'
def configuration(arm,bank):
 hp=copy.deepcopy(json.loads((PRIOR/'plan.json').read_text())['hparams'][arm]);hp['sirm_output_dir']=str(bank);hp['sirm_discover_config']['protocol']='sirm_iter17_actual_mixture_router_risk';return hp
def invocation(arm,bank):
 cmd=command(1,arm,bank);cmd[cmd.index('domainbed.scripts.train')]='experiments.sirm_research_loop.direct_router_risk_train';cmd[cmd.index('--steps')+1]='400';cmd[cmd.index('--hparams')+1]=json.dumps(configuration(arm,bank));return cmd

def main():
 assert json.loads((DEST/'prelaunch_integration_audit.json').read_text())['passed']
 old=json.loads((PRIOR/'plan.json').read_text());audit=json.loads((PRIOR/'final_analysis_audit.json').read_text())
 for f,h in audit['sha256'].items():assert digest(PRIOR/f)==h,f
 assert digest(ANCHOR)==old['anchor_sha256'] and digest(MANIFEST)==old['manifest_sha256'] and digest(CACHE)==old['feature_cache_sha256']
 for f,h in old['code_sha256'].items():assert digest(ROOT/f)==h,f
 variants=['direct_router_risk_model.py','direct_router_risk_train.py','direct_router_risk_prelaunch.py','direct_router_risk_run.py'];locked={**old['code_sha256'],**{str(Path(__file__).with_name(f).relative_to(ROOT)):digest(Path(__file__).with_name(f)) for f in variants}}
 hp={a:configuration(a,TRIO/a) for a in ['OFF','ON']}
 for a in hp:
  before=old['hparams'][a]['sirm_discover_config'];after=hp[a]['sirm_discover_config'];assert {k:v for k,v in before.items() if k!='protocol'}=={k:v for k,v in after.items() if k!='protocol'}
  assert after['global_steps']==0 and after['discover_steps']==200 and after['utility_interval']==5 and after['sirm_ramp_steps']==50 and after['sirm_weighting']=='binary'
 plan=dict(anchor_sha256=digest(ANCHOR),manifest_sha256=digest(MANIFEST),feature_cache_sha256=digest(CACHE),code_sha256=locked,hparams=hp,target=1,seed=0,source_only_training=True,target_selection=False,steps=400,discovery_steps=200,invariance_steps=200,guard_minimum_samples=2,hypothesis='Replacing raw-single-expert teacher KL with actual-mixture CE removes teacher/objective mismatch and may improve useful source corrections and shared Q within the existing400-update trajectory.',changed='Auxiliary router objective only: CE(global_detached + sum pi*delta_detached), original lambda_router/cap retained. Global/base classification, guarded discovery, balance, architecture, Adam, source schedules, calibration, Q, ESS and lambda1/ramp unchanged. This auxiliary router gradient is parallel to classification-router gradient; it adds no new supervision. No shared low-dimensional router or extra fitting budget.',source_evidence_sha256=digest(PRIOR/'source_direct_mixture_reliability_router/interpretation.md'),target_promotion_rule='Complete source utility/Q/repair-harm analysis before locked target replay; no target selection.',scientific_goal_achieved=False)
 assert not (DEST/'plan.json').exists();write(DEST/'plan.json',plan)
 for f in locked:
  dest=DEST/'code_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
 (DEST/'preregistration.md').write_text('Objective-only actual-mixture router risk test from completed source EDA. Same strong frozen ERM anchor, fixed10%manifest and seed0. OFF/ON400 updates: discovery0–199, invariance200–399, Q refresh5 detached, binary eligibility, lambda1 ramp50. Original router architecture, auxiliary coefficient/cap and discovery guard retained. New auxiliary gradient is parallel to classification-router CE, not a new information source. Source-only selection; no target read before source analysis. Improvement requires useful correction and harm control, not confidence or residual contraction alone. No post-hoc400-step router fitting in the primary budget.\n')
 (DEST/'experiment').mkdir(exist_ok=True);shutil.copyfile(MANIFEST,DEST/'experiment/subset_manifest.json');base=TRIO/'ERM_H2';(base/'posthoc').mkdir(parents=True,exist_ok=True);shutil.copyfile(ANCHOR,base/'source_best.pkl');shutil.copyfile(ANCHOR.parent/'posthoc/complete.json',base/'posthoc/complete.json')
 env=os.environ.copy();env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONWARNINGS='ignore',OPENBLAS_NUM_THREADS='4')
 for arm in ['OFF','ON']:
  for f,h in locked.items():assert digest(ROOT/f)==h,f
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
   assert proc.returncode==0,f'{arm} failed; inspect console.log'
  audit_anchor(bank,ANCHOR);rows=[json.loads(l) for l in (bank/'diagnostics.jsonl').read_text().splitlines()];assert [r['step'] for r in rows]==list(range(400));assert all(r['router_objective']=='actual_joint_mixture_CE_detached_experts' for r in rows)
 audit_pair(TRIO)
 initials=[json.loads((TRIO/a/'spatial_initialization.json').read_text()) for a in ['OFF','ON']];assert len({r['parameter_sha256'] for r in initials})==1;assert initials[0]['parameter_sha256']==json.loads((PRIOR/'experiment/runs/L38/OFF/spatial_initialization.json').read_text())['parameter_sha256']
 schedules=[json.loads((TRIO/a/'source_schedules.json').read_text()) for a in ['OFF','ON']];assert schedules[0]==schedules[1]==json.loads((PRIOR/'experiment/runs/L38/OFF/source_schedules.json').read_text())
 write(DEST/'training_audit.json',dict(passed=True,exact_anchor=True,exact_initialization=True,exact_source_schedules=True,contiguous_400_updates=True,target_reads=False))
 write(DEST/'progress.json',dict(status='training_complete',remaining='Source utility/Q/repair-harm trajectory before target promotion'))
if __name__=='__main__':
 write(DEST/'worker_pid.json',dict(pid=os.getpid(),started_unix=time.time()))
 try:main()
 except Exception as e:write(DEST/'progress.json',dict(status='failed',error=str(e)));raise
