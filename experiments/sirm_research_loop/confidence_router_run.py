"""Matched400-update objective-only actual-mixture router falsification."""
import copy,json,os,shutil,subprocess,time
from pathlib import Path
from .routed_discovery_run import ROOT,ANCHOR,CACHE,MANIFEST,command
from .shrinkage import digest,write
from .frozen_erm_audit import audit_anchor,audit_pair
PRIOR=ROOT/'results/sirm_research_loop/iteration_17';DEST=PRIOR.parent/'iteration_18';TRIO=DEST/'experiment/runs/L38'
def configuration(arm,bank):
 hp=copy.deepcopy(json.loads((PRIOR/'plan.json').read_text())['hparams'][arm]);hp['sirm_output_dir']=str(bank);hp['sirm_discover_config']['protocol']='sirm_iter18_shared_confidence_router';return hp
def invocation(arm,bank):
 cmd=command(1,arm,bank);cmd[cmd.index('domainbed.scripts.train')]='experiments.sirm_research_loop.confidence_router_train';cmd[cmd.index('--steps')+1]='400';cmd[cmd.index('--hparams')+1]=json.dumps(configuration(arm,bank));return cmd

def main():
 assert json.loads((DEST/'prelaunch_integration_audit.json').read_text())['passed']
 old=json.loads((PRIOR/'plan.json').read_text());audit=json.loads((PRIOR/'final_analysis_audit.json').read_text())
 for f,h in audit['sha256'].items():assert digest(PRIOR/f)==h,f
 assert digest(ANCHOR)==old['anchor_sha256'] and digest(MANIFEST)==old['manifest_sha256'] and digest(CACHE)==old['feature_cache_sha256']
 for f,h in old['code_sha256'].items():assert digest(ROOT/f)==h,f
 variants=['confidence_router_model.py','confidence_router_train.py','confidence_router_prelaunch.py','confidence_router_run.py','source_direct_mixture_reliability_router.py'];locked={**old['code_sha256'],**{str(Path(__file__).with_name(f).relative_to(ROOT)):digest(Path(__file__).with_name(f)) for f in variants}}
 hp={a:configuration(a,TRIO/a) for a in ['OFF','ON']}
 for a in hp:
  before=old['hparams'][a]['sirm_discover_config'];after=hp[a]['sirm_discover_config'];assert {k:v for k,v in before.items() if k!='protocol'}=={k:v for k,v in after.items() if k!='protocol'}
  assert after['global_steps']==0 and after['discover_steps']==200 and after['utility_interval']==5 and after['sirm_ramp_steps']==50 and after['sirm_weighting']=='binary'
 plan=dict(anchor_sha256=digest(ANCHOR),manifest_sha256=digest(MANIFEST),feature_cache_sha256=digest(CACHE),code_sha256=locked,hparams=hp,target=1,seed=0,source_only_training=True,target_selection=False,steps=400,discovery_steps=200,invariance_steps=200,guard_minimum_samples=2,hypothesis='Shared expert-confidence routing may expose conditional useful corrections during the existing400-update bank trajectory, unlike objective-only original router. Falsify against prior17 actual-risk routing and matched OFF/ON.',changed='Router architecture/readout only: original768->7 replaced by shared11->1 expert confidence score with NULL0 and zero initialweights/bias. Label-free descriptors of frozen global and each centered residual candidate, detached from experts. Entropy/log10 and frozen-anchor source-in class-centered global logitRMS define fixed units, no online normalization or learned source-out stats. Auxiliary actual-mixture CE, lambda_router/cap, guarded discovery, centering, balance, Adam, source schedules, calibration, Q/ESS and lambda1/ramp retained. Inverse bank normalization reconstructs raw CLS for descriptor logits within1e-4; deployed global still uses exact CLS. No diagnostic weights reused or extra Stage2 fits.',source_evidence_sha256=digest(PRIOR/'source_magnitude_aware_reliability_router/interpretation.md'),target_promotion_rule='Complete source utility/Q/repair-harm analysis before locked target replay; no target selection.',scientific_goal_achieved=False)
 assert not (DEST/'plan.json').exists();write(DEST/'plan.json',plan)
 for f in locked:
  dest=DEST/'code_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
 (DEST/'preregistration.md').write_text('Shared-confidence architecture test following completed iteration17 source-only EDA. Same immutable strong ERM,10%manifest/seed0, OFF/ON400updates0–199discovery,200–399invariance, detachedQ refresh5,binarylambda1ramp50. No expert or diagnostic router weights reused. Shared11confidence head with NULL0 and zero initialization; fixed units from anchor source-in logits, descriptors detached from residuals. Retain actual-mixture auxiliary CE, oldcoefficients/cap/discoveryguard. This is architecture/readout+fixed-normalization family change, not an isolated objective change or identical diagnostic standardization. Freeze all global/backbone; source-only selection and source case/Q analysis before target replay. No threshold/lambda tuning based on target; lack of support remains a failed mechanism test. Prior architecture initialization differs; OFF/ON exact paired initialization/schedules required.\n')
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
 initials=[json.loads((TRIO/a/'spatial_initialization.json').read_text()) for a in ['OFF','ON']];assert len({r['parameter_sha256'] for r in initials})==1
 schedules=[json.loads((TRIO/a/'source_schedules.json').read_text()) for a in ['OFF','ON']];assert schedules[0]==schedules[1]==json.loads((PRIOR/'experiment/runs/L38/OFF/source_schedules.json').read_text())
 write(DEST/'training_audit.json',dict(passed=True,exact_anchor=True,exact_initialization=True,exact_source_schedules=True,contiguous_400_updates=True,target_reads=False))
 write(DEST/'progress.json',dict(status='training_complete',remaining='Source utility/Q/repair-harm trajectory before target promotion'))
if __name__=='__main__':
 write(DEST/'worker_pid.json',dict(pid=os.getpid(),started_unix=time.time()))
 try:main()
 except Exception as e:write(DEST/'progress.json',dict(status='failed',error=str(e)));raise
