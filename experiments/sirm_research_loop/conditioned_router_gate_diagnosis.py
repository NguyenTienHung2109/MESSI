"""Read-only live/final source EMA support blockers, never change Q."""
import csv,json,time
from pathlib import Path
import numpy as np
import torch
from .shrinkage import ROOT,write,digest
from experiments.sirm_discover.math import soft_null_q
DEST=ROOT/'results/sirm_research_loop/iteration_19'
def main():
 plan=json.loads((DEST/'plan.json').read_text());out={};allrows=[]
 for arm in ['OFF','ON']:
  bank=DEST/'experiment/runs/L38'/arm;p=bank/'q_refresh.jsonl'
  if not p.exists():continue
  config=plan['hparams'][arm]['sirm_discover_config'];rows=[]
  for line in p.read_text().splitlines():
   try:a=json.loads(line)
   except ValueError:continue
   if a['step']<200:continue
   u=torch.tensor(a['utility_ema']);valid=torch.tensor(a['utility_valid']);q,_,_,v=soft_null_q(u,valid,config['q_temperature'],config['utility_floor'],config['interaction_floor']);saved=torch.tensor(a['q']);assert float((q-saved).abs().max())<1e-7
   uonly=(u>config['utility_floor'])&valid;both=uonly&(v>config['interaction_floor']);actual=saved>0
   rows.append(dict(arm=arm,step=a['step'],complete_utility_matrix=bool(valid.all()),max_utility=float(u.max()),max_positive_interaction=float(v.max()),U_floor_passing_cells=int(uonly.sum()),U_floor_potential_shared_experts=int((uonly.sum(0)>=2).sum()),both_floors_passing_cells=int(both.sum()),both_floors_potential_shared_experts=int((both.sum(0)>=2).sum()),actual_selected_cells=int(actual.sum()),actual_shared_experts=int((actual.sum(0)>=2).sum()),source_validation_accuracy=a.get('source_validation_accuracy')))
  if not rows:continue
  out[arm]=dict(through_step=rows[-1]['step'],refreshes=len(rows),U_only_shared_candidate_refreshes=sum(x['U_floor_potential_shared_experts']>0 for x in rows),joint_shared_candidate_refreshes=sum(x['both_floors_potential_shared_experts']>0 for x in rows),actual_shared_refreshes=sum(x['actual_shared_experts']>0 for x in rows),all_U_candidates_blocked_by_V_refreshes=sum(x['U_floor_potential_shared_experts']>0 and x['both_floors_potential_shared_experts']==0 for x in rows),latest=rows[-1],complete_training=(bank/'done').exists());allrows.extend(rows)
 write(DEST/'source_Q_floor_diagnosis.json',dict(source_only=True,target_read=False,parameter_updates=False,config_changes=False,observed_unix=time.time(),arms=out,interpretation='U-only support is a counterfactual threshold count, not a trained alternative Q or proof of transferable utility. Original augmented-minibatch EMA is distinct from source-out diagnostic utility. Exact current Q replay verified. Double-centering can suppress domain-common utility; causal usefulness and any new selector require completed analysis.',code_sha256=digest(Path(__file__))))
 if allrows:
  with (DEST/'source_Q_floor_history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
