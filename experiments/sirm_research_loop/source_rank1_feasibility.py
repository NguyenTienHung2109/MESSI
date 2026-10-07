"""Label-aware rank1 output-line feasibility bound, not deployable accuracy."""
import json
from pathlib import Path
import numpy as np,torch
from .source_low_rank_readout import CACHE
from .source_rank1_class_payoff import DEST
from .shrinkage import write,digest

def interval(g,b,y):
 lo=-np.inf;hi=np.inf
 for j in range(len(b)):
  if j==y:continue
  slope=float(b[y]-b[j]);rhs=float(g[j]-g[y])
  if abs(slope)<1e-12:
   if rhs>1e-10:return False,lo,hi
  elif slope>0:lo=max(lo,rhs/slope)
  else:hi=min(hi,rhs/slope)
 return lo<=hi+1e-10,lo,hi

def main():
 root=DEST.parent;data=dict(np.load(CACHE/'source_out_tokens.npz'));families={'risk_learned':('source_rank1_risk_axis','learned_risk_axis'),'risk_fixed':('source_rank1_risk_axis','fixed_risk_axis'),'agreement_learned':('source_rank1_domain_agreement','learned_minimum_norm_convex'),'random0':('source_low_rank_readout','rank1'),'random1':('source_rank1_initialization','learned_seed1'),'random2':('source_rank1_initialization','learned_seed2')};records={};locks={};raw=torch.tensor(np.concatenate([data['cls'],data['patch_max']],1));g=data['global_logits'].astype(float);y=data['y'];dom=data['domains'];gc=g.argmax(1)==y
 for name,(folder,suffix) in families.items():
  p=root/folder/f'pooled_{suffix}.pt';locks[str(p)]=digest(p);a=torch.load(p,map_location='cpu',weights_only=False);b=a['b'][:,0].numpy().astype(float);b-=b.mean();norm=np.linalg.norm(b);b/=norm;z=(raw-a['mean'])/a['std']/2**.5;scalar=(z@a['a'][0]*norm).numpy();items=[];feasible=[]
  for i in range(len(g)):
   possible,lo,hi=interval(g[i],b,int(y[i]));feasible.append(possible);actual=g[i]+scalar[i]*b;pred=int(actual.argmax());items.append(dict(id=str(data['ids'][i]),label=int(y[i]),global_prediction=int(g[i].argmax()),prediction=pred,reachable_with_any_scalar=bool(possible),scalar_interval_lower=float(lo) if np.isfinite(lo) else None,scalar_interval_upper=float(hi) if np.isfinite(hi) else None,actual_scalar=float(scalar[i]),GT_margin=float(actual[y[i]]-max(np.delete(actual,y[i])))))
  f=np.array(feasible);assert f[gc].all();records[name]=dict(class_axis=b.tolist(),global_errors=int((~gc).sum()),reachable_errors=int((~gc&f).sum()),unreachable_errors=int((~gc&~f).sum()),label_oracle_equal_domain_upper_bound=float(np.mean([f[dom==d].mean() for d in np.unique(dom)])),per_class={str(c):dict(n=int((y==c).sum()),global_errors=int(((y==c)&~gc).sum()),reachable_errors=int(((y==c)&~gc&f).sum()),unreachable_errors=int(((y==c)&~gc&~f).sum())) for c in range(10)},cases=items)
 for p,h in locks.items():assert digest(Path(p))==h
 write(DEST/'rank1_feasibility.json',dict(source_only=True,target_read=False,fitting=False,label_aware_oracle_not_deployable=True,method='Solve all inequalities g_y+s*b_y>=g_j+s*b_j at fixed learned class axis; unbounded scalar s, ties allowed. This upper bound grants GT-dependent arbitrary scalar and excludes input readout capacity. It cannot demonstrate achievable performance or causal visual factors.',families=records,input_sha256=locks))
 for k,v in records.items():print(k,v['reachable_errors'],v['unreachable_errors'],v['per_class']['0'])
if __name__=='__main__':main()
