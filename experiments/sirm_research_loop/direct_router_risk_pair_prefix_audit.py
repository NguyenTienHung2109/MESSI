"""Actual OFF/ON log-prefix control checks, no tensor-history claims."""
import json,time,math
from pathlib import Path
from .shrinkage import ROOT,write,digest
DEST=ROOT/'results/sirm_research_loop/iteration_17'
def read(p):
 a=[];ls=p.read_text().splitlines()
 for i,l in enumerate(ls):
  try:a.append(json.loads(l))
  except json.JSONDecodeError:assert i==len(ls)-1
 return a

def main():
 banks={arm:DEST/'experiment/runs/L38'/arm for arm in ['OFF','ON']};hist={a:read(b/'diagnostics.jsonl') for a,b in banks.items()};n=min(map(len,hist.values()));init=[json.loads((b/'spatial_initialization.json').read_text()) for b in banks.values()];schedules=[json.loads((b/'source_schedules.json').read_text()) for b in banks.values()];assert init[0]['parameter_sha256']==init[1]['parameter_sha256'] and schedules[0]==schedules[1]
 fields=['stage','loss','loss_cls','loss_discover','loss_router','loss_balance','loss_sirm','router_null_mass','residual_rms','q','q_null','utility_ema','utility_interaction','G','q_refreshed','sirm_active','sirm_pair_count'];first={k:next((i for i in range(n) if hist['OFF'][i][k]!=hist['ON'][i][k]),None) for k in fields}
 assert all(v is None or v>=200 for v in first.values()),first
 for a in hist:
  assert [r['step'] for r in hist[a]]==list(range(len(hist[a])))
  for r in hist[a]:
   expected=0 if a=='OFF' or r['step']<200 else min((r['step']-200)/50,1)
   assert r['effective_lambda_sirm']==expected
   assert r['step']>=200 or (r['loss_sirm']==0 and not r['sirm_active'])
   assert all(math.isfinite(r[k]) for k in ['loss','loss_cls','loss_discover','loss_router','loss_balance','loss_sirm'])
 shared=[r for r in hist['ON'] if r['step']>=200 and r.get('shared_experts',0)>0];lastq=next((r for r in reversed(hist['ON']) if r.get('q_refreshed')),None)
 result=dict(through_step=n-1,OFF_last_step=hist['OFF'][-1]['step'],ON_last_step=hist['ON'][-1]['step'],same_initial_parameters=True,same_source_schedules=True,preactivation_prefix_matches=True,all_compared_trajectory_fields_equal=all(v is None for v in first.values()),first_difference_by_field=first,ON_shared_refreshes_so_far=len(shared),ON_active_steps_so_far=sum(r['sirm_active'] for r in hist['ON'][200:]),latest_ON_Q={'step':lastq['step'],'selected_cells':lastq.get('selected_cells'),'shared_experts':lastq.get('shared_experts'),'eligible_pairs':lastq.get('eligible_pair_expert_count')} if lastq else None,observed_unix=time.time(),scope='Exact reported loss/output-statistic/Q prefix comparison; initialization named-parameter hash and source schedules. Not direct intermediate checkpoint tensor comparisons or completed-run audit.',source_only=True,target_read=False,training=False,scientific_goal_achieved=False,code_sha256=digest(Path(__file__)))
 write(DEST/'pair_prefix_audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
