"""Read-only fixed-bank utility stability, not cross-fitted model evaluation."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from .shrinkage import ROOT,write,digest
DEST=ROOT/'results/sirm_research_loop/iteration_19/source_utility_stability'
def main():
 DEST.mkdir(exist_ok=True)
 root=DEST.parent;inputs=[root/f'source_analysis/OFF_{state}_{split}_replay.npz' for state in ['selected','last'] for split in ['source_in','source_out']]
 plan=dict(source_only=True,target_read=False,fitting=False,model_crossfit=False,partition='Within domain/class SHA256(image_id) sorted round-robin4; fixed model trained on allsource-in. Stability partitions do not exclude images from model fitting.',bootstrap='600 class-stratified image resamples per domain seed19001; descriptive, not correlation-adjusted confidence intervals',minimum_samples=2,utility_floor=.002,code_sha256=digest(Path(__file__)),inputs={str(p.relative_to(ROOT)):digest(p) for p in inputs})
 write(DEST/'plan.json',plan);rows=[];classrows=[];summary={};rng=np.random.default_rng(19001)
 for state in ['selected','last']:
  stats={}
  for split in ['source_in','source_out']:
   p=np.load(root/f'source_analysis/OFF_{state}_{split}_replay.npz');ids=p['ids'];y=p['y'];dom=p['domains'];u=p['utility'].astype(float);correct=p['global_logits'].argmax(1)==y;fold=np.full(len(y),-1)
   for d in np.unique(dom):
    for c in np.unique(y[dom==d]):
     ix=np.flatnonzero((dom==d)&(y==c));ix=sorted(ix,key=lambda i:hashlib.sha256(str(ids[i]).encode()).hexdigest())
     for j,i in enumerate(ix):fold[i]=j%4
   for d in np.unique(dom):
    ix=np.flatnonzero(dom==d);groups=[np.flatnonzero((dom==d)&(y==c)) for c in np.unique(y[ix])];draws=np.stack([u[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])].mean(0) for _ in range(600)])
    for e in range(6):
     val=u[ix,e];neg=np.maximum(-val,0);pos=np.maximum(val,0);er=~correct[ix];halves=[float(u[(dom==d)&(fold==k),e].mean()) for k in range(4)];order=np.sort(neg)[::-1]
     row=dict(state=state,split=split,domain=int(d),expert=e,n=len(ix),mean=float(val.mean()),median=float(np.median(val)),positive_fraction=float((val>0).mean()),negative_mass_on_global_errors=float(neg[er].sum()/neg.sum()) if neg.sum()>0 else None,negative_mass_top3_fraction=float(order[:3].sum()/neg.sum()) if neg.sum()>0 else None,positive_mass_on_global_correct=float(pos[~er].sum()/pos.sum()) if pos.sum()>0 else None,global_error_U_mean=float(val[er].mean()) if er.any() else None,global_correct_U_mean=float(val[~er].mean()) if (~er).any() else None,partition_means=halves,all_partition_positive=all(v>0 for v in halves),bootstrap_positive_fraction=float((draws[:,e]>0).mean()),bootstrap_floor_fraction=float((draws[:,e]>.002).mean()))
     rows.append(row)
    for c in np.unique(y[ix]):
     jj=np.flatnonzero((dom==d)&(y==c))
     for e in range(6):classrows.append(dict(state=state,split=split,domain=int(d),label=int(c),expert=e,n=len(jj),mean=float(u[jj,e].mean()),global_wrong_n=int((~correct[jj]).sum())))
   here=[r for r in rows if r['state']==state and r['split']==split]
   stats[split]=dict(positive_mean_cells=sum(r['mean']>0 for r in here),all4_partition_positive_cells=sum(r['all_partition_positive'] for r in here),U_floor_cells=sum(r['mean']>.002 for r in here),bootstrap_floor_probability_ge_095_cells=sum(r['bootstrap_floor_fraction']>=.95 for r in here))
  train=[r for r in rows if r['state']==state and r['split']=='source_in'];val=[r for r in rows if r['state']==state and r['split']=='source_out']
  summary[state]=dict(partition_summary=stats,stable_train_positive_cells=sum(r['all_partition_positive'] for r in train),stable_train_positive_but_validation_negative=sum(a['all_partition_positive'] and b['mean']<0 for a,b in zip(train,val)),validation_positive_cells=[r for r in val if r['mean']>0],validation_error_negative_cells=sum(r['global_error_U_mean'] is not None and r['global_error_U_mean']<0 for r in val))
 for name,data in [('cells.csv',rows),('class_cells.csv',classrows)]:
  with (DEST/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader()
   for r in data:w.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()})
 write(DEST/'results.json',dict(plan=plan,summary=summary,cells=rows))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
