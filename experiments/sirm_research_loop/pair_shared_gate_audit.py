"""Read-only gate and discovery-scale trajectory audit, partial or complete."""
import argparse,json,math
from pathlib import Path
import numpy as np
from .shrinkage import digest,write

def run(dest,arm):
 bank=dest/'experiment/runs/L38'/arm;plan=json.loads((dest/'plan.json').read_text());c=plan['hparams'][arm]['sirm_discover_config'];rows=[json.loads(l) for l in (bank/'diagnostics.jsonl').read_text().splitlines()];refs=[r for r in rows if r['q_refreshed'] and r['step']>=200];counts={'invalid_U_refreshes':0,'no_U_floor_pass_refreshes':0,'U_pass_but_no_joint_U_V_refreshes':0,'joint_pass_but_no_shared_refreshes':0,'shared_but_ESS_ineligible_refreshes':0,'eligible_refreshes':0};u_max=None;v_max=None;records=[]
 for r in refs:
  u=np.array(r['utility_ema']);v=np.array(r['utility_interaction']);valid=np.array(r['utility_valid']);up=u>c['utility_floor'];vp=v>c['interaction_floor'];joint=up&vp;u_max=float(u.max()) if u_max is None else max(u_max,float(u.max()));v_max=float(v.max()) if v_max is None else max(v_max,float(v.max()));
  if not valid.all():blocker='invalid_U_refreshes'
  elif not up.any():blocker='no_U_floor_pass_refreshes'
  elif not joint.any():blocker='U_pass_but_no_joint_U_V_refreshes'
  elif not r['shared_experts']:blocker='joint_pass_but_no_shared_refreshes'
  elif not r['eligible_pair_expert_count']:blocker='shared_but_ESS_ineligible_refreshes'
  else:blocker='eligible_refreshes'
  counts[blocker]+=1;records.append({'step':r['step'],'U_floor_pass':int(up.sum()),'V_floor_pass':int(vp.sum()),'joint_pass':int(joint.sum()),'selected_cells':r['selected_cells'],'shared_experts':r['shared_experts'],'eligible_pairs':r['eligible_pair_expert_count'],'blocker':blocker})
 errors=[abs(r['discovery_matched_parameter_grad_norm']-r['discovery_hard_parameter_grad_norm'])/max(r['discovery_hard_parameter_grad_norm'],1e-20) for r in rows];assert max(errors)<1e-5;assert all(math.isfinite(r['discovery_detached_scale']) for r in rows);assert [r['step'] for r in rows]==list(range(len(rows)))
 result={'code_sha256':digest(Path(__file__)),'arm':arm,'through_step':rows[-1]['step'],'complete_400_steps':len(rows)==400 and (bank/'done').exists(),'scope':'Observed logged invariance refreshes only; thresholds unchanged. Marginal utility is routed/contextual, whereas pair discovery uses unscaled single corrections. Do not substitute pair utility for Q utility.','refreshes':len(refs),'blockers':counts,'max_positive_U':u_max,'max_positive_V':v_max,'floors':{'U':c['utility_floor'],'V':c['interaction_floor']},'discovery_norm_match_max_relative_error':max(errors),'no_pair_fallback_steps':sum(r['discovery_no_pair_fallback'] for r in rows),'zero_pair_gradient_fallback_steps':sum(r['discovery_zero_pair_gradient_fallback'] for r in rows),'active_steps':sum(r['sirm_active'] for r in rows),'records':records,'scientific_goal_achieved':False};write(dest/f'{arm}_gate_audit.json',result);print({k:v for k,v in result.items() if k not in ['records','scope']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('destination',type=Path);p.add_argument('arm',choices=['OFF','ON']);a=p.parse_args();run(a.destination,a.arm)
