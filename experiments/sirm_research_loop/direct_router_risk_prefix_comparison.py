"""Matched source-only trajectory comparison; live snapshots, no intervention."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from .shrinkage import ROOT,write
BASE=ROOT/'results/sirm_research_loop'
def snap(p):
 raw=p.read_bytes();lines=raw.splitlines();rows=[]
 for i,line in enumerate(lines):
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:assert i==len(lines)-1
 return rows,hashlib.sha256(raw).hexdigest()
def main():
 data={};hashes={}
 for it in [16,17]:
  root=BASE/f'iteration_{it}/experiment/runs/L38/OFF';data[it]={}
  for kind,file in [('history','diagnostics.jsonl'),('refresh','q_refresh.jsonl'),('validation','results.jsonl')]:
   data[it][kind],hashes[str(root/file)]=snap(root/file)
  data[it]['init']=json.loads((root/'spatial_initialization.json').read_text());data[it]['schedule']=json.loads((root/'source_schedules.json').read_text())
 assert data[16]['init']['parameter_sha256']==data[17]['init']['parameter_sha256'] and data[16]['schedule']==data[17]['schedule']
 through=min(data[i]['history'][-1]['step'] for i in [16,17]);traces={};pairs=[]
 for it in [16,17]:
  h={r['step']:r for r in data[it]['history'] if r['step']<=through};v={r['step']:r for r in data[it]['validation'] if r['step']<=through};q={r['step']:r for r in data[it]['refresh'] if 200<=r['step']<=through};traces[it]=(h,v,q)
 for step in sorted(set(traces[16][1])&set(traces[17][1])):
  if 'source_val_mean' not in traces[16][1][step] or 'source_val_mean' not in traces[17][1][step]:continue
  record={'step':step,'states':{}}
  for it in [16,17]:
   h,v,q=traces[it];a=h[step];s=dict(source_accuracy=v[step]['source_val_mean'],router_NULL=a['router_null_mass'],residual_RMS=a['residual_rms'],train_minibatch_CE=a['loss_cls'])
   if step in q:
    b=q[step];u=np.array(b['utility_ema']);vv=np.array(b['utility_interaction']);s.update(utility_max=float(u.max()),interaction_max=float(vv.max()),utility_floor_pass=int((u>.002).sum()),interaction_floor_pass=int((vv>.002).sum()),joint_floor_pass=int(((u>.002)&(vv>.002)).sum()),selected_cells=b['selected_cells'],shared_experts=b['shared_experts'],eligible_pairs=b['eligible_pair_expert_count'])
   record['states'][str(it)]=s
  record['accuracy_delta_17_minus_16']=record['states']['17']['source_accuracy']-record['states']['16']['source_accuracy'];pairs.append(record)
 write(BASE/'iteration_17/live_objective_comparison.json',dict(observed_unix=time.time(),through_step=through,source_only=True,target_read=False,scope='Same-step OFF trajectory log snapshots; source selection unchanged, partial training result. Minibatch CE is a single scheduled H2 batch, not population CE.',initial_parameters_exact=True,source_schedules_exact=True,training_complete=False,source_rows=pairs,stage3_refresh_support={str(i):dict(refreshes=len(traces[i][2]),shared=sum(r['shared_experts']>0 for r in traces[i][2].values()),selected_cells_max=max([r['selected_cells'] for r in traces[i][2].values()] or [0])) for i in [16,17]},snapshot_sha256=hashes,scientific_goal_achieved=False))
 print(json.dumps(dict(through_step=through,latest_pair=pairs[-1]),indent=2))
if __name__=='__main__':main()
