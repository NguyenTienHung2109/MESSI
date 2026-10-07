"""Read-only live prefix audit; no training or selection modifications."""
import json,math,time
from pathlib import Path
from .shrinkage import ROOT,write,digest
DEST=ROOT/'results/sirm_research_loop/iteration_17'
def rows(p):
 if not p.exists():return []
 raw=p.read_text().splitlines();out=[]
 for i,line in enumerate(raw):
  try:out.append(json.loads(line))
  except json.JSONDecodeError:
   assert i==len(raw)-1,'Malformed completed log row'
 return out

def main():
 plan=json.loads((DEST/'plan.json').read_text());assert all(digest(ROOT/f)==h for f,h in plan['code_sha256'].items());arms={}
 for arm in ['OFF','ON']:
  root=DEST/'experiment/runs/L38'/arm;history=rows(root/'diagnostics.jsonl')
  if not history:continue
  frozen=rows(root/'frozen_anchor_checks.jsonl');refresh=rows(root/'q_refresh.jsonl');validation=rows(root/'results.jsonl')
  checks=dict(contiguous_prefix=[r['step'] for r in history]==list(range(len(history))),stage_boundaries=all(r['stage']==('discover' if r['step']<200 else 'invariance') for r in history),losses_finite=all(math.isfinite(r[k]) for r in history for k in ['loss','loss_cls','loss_discover','loss_router','loss_balance','loss_sirm']),preactivation_penalty_zero=all(r['loss_sirm']==0 and not r['sirm_active'] and r['effective_lambda_sirm']==0 for r in history if r['step']<200),router_objective_exact=all(r['router_objective']=='actual_joint_mixture_CE_detached_experts' and abs(r['router_aux_raw_loss']-r['router_objective_raw_CE'])<1e-6 for r in history),aux_residual_gradient_absent=all(r['router_objective_residual_gradient_absent'] for r in history),aux_cap_preserved=all(r['router_aux_post_cap_parameter_grad_norm']<=r['router_cls_parameter_grad_norm']*(1+1e-5)+1e-10 for r in history),Q_refresh_cadence=all(r['q_refreshed']==(r['step']%5==0) for r in history),logged_frozen_checks_pass=all(r['requires_grad_false'] and r['gradient_absent'] and r['parameter_versions_unchanged'] for r in frozen),target_validation_absent=all(not any(k.startswith('env1_') for k in r) for r in validation))
  init=json.loads((root/'spatial_initialization.json').read_text());prior=json.loads((DEST.parent/'iteration_16/experiment/runs/L38/OFF/spatial_initialization.json').read_text());checks['original_initialization_exact']=init['parameter_sha256']==prior['parameter_sha256'];checks['actual_anchor_lock']=json.loads((root/'anchor_loaded.json').read_text())['checkpoint_sha256']==plan['anchor_sha256']
  assert all(checks.values()),checks
  stage3=[r for r in history if r['step']>=200];arms[arm]=dict(last_step=history[-1]['step'],completed_rows=len(history),checks=checks,frozen_check_rows=len(frozen),active_steps=sum(r['sirm_active'] for r in stage3),stage3_steps_so_far=len(stage3),shared_refreshes_so_far=sum(r.get('shared_experts',0)>0 for r in refresh if r['step']>=200),latest_losses={k:history[-1][k] for k in ['loss_cls','loss_router','loss_discover','residual_rms','router_null_mass']},latest_source_validation=[{k:r[k] for k in ['step','source_val_mean'] if k in r} for r in validation[-1:]])
 write(DEST/'live_behavior_audit.json',dict(observed_unix=time.time(),arms=arms,all_checks_pass=True,scope='Available contiguous completed log prefix; frozen checks are version/requires_grad assertions, not final tensor equality audit. Target keys absent from logged validation. Full400-update/paired audit pending.',source_only=True,target_read=False,scientific_goal_achieved=False,code_sha256=digest(Path(__file__))))
 print(json.dumps(arms,indent=2))
if __name__=='__main__':main()
