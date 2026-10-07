"""Convex per-image source mixture oracle; no target, no primary parameter updates."""
import json
from pathlib import Path
import numpy as np
import torch
from .shrinkage import digest,write
from .fixed_teacher_probe import score
B=Path('results/sirm_research_loop/iteration_08')
def project(v):
 u=v.sort(1,descending=True).values;css=u.cumsum(1)-1;r=torch.arange(1,v.shape[1]+1,dtype=v.dtype);valid=u-css/r>0;rho=valid.sum(1).clamp_min(1);theta=css.gather(1,(rho-1)[:,None])/rho[:,None];return (v-theta).clamp_min(0)
def solve(a,max_iter=20000,tolerance=1e-5):
 g=a['g'].double();delta=a['delta'].double();y=a['y'];d=torch.cat([delta,torch.zeros(len(g),1,g.shape[1],dtype=torch.double)],1);p=a['teacher'].double().clone();p/=p.sum(1,keepdim=True)
 # softmax CE Hessian has spectral norm <=1/2. Frobenius D bound is conservative.
 L=(.5*d.square().sum((1,2))).clamp_min(1e-8);one=torch.nn.functional.one_hot(y,g.shape[1]).double();gap=None
 for step in range(max_iter):
  logits=g+(p[:,:,None]*d).sum(1);grad=((logits.softmax(1)-one)[:,None,:]*d).sum(2);gap=(p*grad).sum(1)-grad.min(1).values
  if gap.max()<=tolerance:break
  p=project(p-grad/L[:,None])
 return p.float(),dict(iterations=step+1,max_FW_gap=float(gap.max()),mean_FW_gap=float(gap.mean()),converged=bool(gap.max()<=tolerance),tolerance=tolerance)
def run():
 torch.set_num_threads(4);out=B/'mixture_teacher_diagnosis';out.mkdir(exist_ok=True);paths={s:B/f'source_router_diagnosis/{s}_original.npz' for s in ['source_in','source_out']};locks={s:digest(p) for s,p in paths.items()};plan={'code_sha256':digest(Path(__file__)),'cache_hashes':locks,'scope':'Separate label-dependent per-image simplex CE oracle on fixed source bank; no target or bank training. Source-out oracle is retrospective capacity only and will not be used to train/select a policy.','hypothesis':'Single-expert teacher utilities may not align with its composed mixture; quantify objective gap before changing distillation.','solver':'Projected gradient using per-image conservative Hessian bound, initial existing teacher, max20000, stop max Frank-Wolfe dual gap<=1e-5','primary_method_changed':False};write(out/'plan.json',plan);results={}
 for s,path in paths.items():
  raw=dict(np.load(path));a={k:torch.tensor(raw[k]) for k in ['g','delta','y','domains','pi','teacher']};p,solver=solve(a);full=a['g']+(p[:,:6,None]*a['delta']).sum(1);old=a['g']+(a['teacher'][:,:6,None]*a['delta']).sum(1);oldce=torch.nn.functional.cross_entropy(old,a['y'],reduction='none');newce=torch.nn.functional.cross_entropy(full,a['y'],reduction='none');assert (newce-oldce).max()<1e-5
  # local linear mixture gradient at old teacher, its simplex FW gap diagnostic.
  d=torch.cat([a['delta'].double(),torch.zeros(len(p),1,10,dtype=torch.double)],1);prob=old.double().softmax(1);one=torch.nn.functional.one_hot(a['y'],10).double();grad=((prob-one)[:,None]*d).sum(2);gap=(a['teacher'].double()*grad).sum(1)-grad.min(1).values
  results[s]={'solver':solver,'old_teacher':score(a,a['teacher']),'mixture_oracle':score(a,p),'mean_CE_improvement':float((oldce-newce).mean()),'old_teacher_FW_gap_mean':float(gap.mean()),'old_teacher_FW_gap_max':float(gap.max()),'p_L1_mean_change':float((p-a['teacher']).abs().sum(1).mean())};np.savez_compressed(out/f'{s}_oracle.npz',pi=p.numpy(),y=raw['y'],domains=raw['domains'],ids=raw['ids'])
  print(s,solver,results[s]['mixture_oracle']['pooled'],flush=True)
 assert all(digest(p)==locks[s] for s,p in paths.items());write(out/'results.json',{'plan':plan,'splits':results})
if __name__=='__main__':run()
