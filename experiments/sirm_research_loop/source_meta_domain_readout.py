"""Source-only virtual domain-transfer readout, with matched objective-scale control."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .routed_discovery_run import ROOT, DEST
from .shrinkage import digest, write
from .source_structured_contrast_readout import score
OUT=DEST/'source_meta_domain_readout'
L2=.01
ALPHA=.1

def fit(x,g,y,dom,basis,kind):
 ds=sorted(set(dom)); weights=np.array([1/(len(ds)*sum(dom==d)) for d in dom]);mean=weights@x;std=np.sqrt(weights@((x-mean)**2)).clip(1e-8)
 z=torch.tensor((x-mean)/std,dtype=torch.float64);gg=torch.tensor(g,dtype=torch.float64);yy=torch.tensor(y);bb=torch.tensor(basis,dtype=torch.float64);pg=gg.softmax(1);coef=torch.zeros((x.shape[1],basis.shape[1]),dtype=torch.float64,requires_grad=True)
 ix={d:torch.tensor(np.flatnonzero(dom==d)) for d in ds}
 def risk(c,domains):
  logits=gg+z@c@bb.T;v=F.cross_entropy(logits,yy,reduction='none')+(pg*(pg.log()-logits.log_softmax(1))).sum(1)
  return torch.stack([v[ix[d]].mean() for d in domains]).mean()
 def objective():
  base=risk(coef,ds)
  if kind=='matched_double_risk':meta=base
  else:
   episodes=[]
   for held in ds:
    adapt=risk(coef,[d for d in ds if d!=held])+L2/2*coef.square().sum()
    grad=torch.autograd.grad(adapt,coef,create_graph=True)[0]
    episodes.append(risk(coef-ALPHA*grad,[held]))
   meta=torch.stack(episodes).mean()
  return base+meta+L2/2*coef.square().sum(),base,meta
 opt=torch.optim.LBFGS([coef],max_iter=500,tolerance_grad=1e-8,tolerance_change=1e-12,line_search_fn='strong_wolfe');calls=0
 def closure():
  nonlocal calls
  opt.zero_grad();loss,_,_=objective();loss.backward();calls+=1;return loss
 opt.step(closure);obj=float(closure().detach());grad=float(coef.grad.abs().max());assert grad<1e-5,(kind,grad,calls)
 _,base,meta=objective()
 return dict(mean=mean,std=std,coef=coef.detach().numpy(),basis=basis,objective=obj,gradient_max=grad,closure_calls=calls,base_risk=float(base.detach()),meta_risk=float(meta.detach()))

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();parent=DEST/'source_structured_contrast_readout';ref=json.loads((parent/'results.json').read_text());locks={str(parent/'results.json'):digest(parent/'results.json')};source={}
 for split in ['source_in','source_out']:
  f=ROOT/f'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/{split}_tokens.npz';assert digest(f)==ref['input_sha256'][str(f)];locks[str(f)]=digest(f);source[split]=dict(np.load(f))
 plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,diagnostic_readout_fitting=True,held_domains=[None,0,2,3],families=['matched_double_risk','virtual_domain_transfer'],L2=L2,virtual_step=ALPHA,meta_weight=1.,base='Equal-domain original-source-in CE + KL(global||composed), fixed coefficient1; frozen global unchanged. Same centered rank9 output and zero-intercept linear CLS+patch-max correction.',meta='For each fitting domain, take one differentiable gradient step on the remaining fitting domains CE+KL+L2. Evaluate CE+KL on the held fitting domain; average episodes. Outer objective base+meta+L2. Full second derivatives, not first-order MAML.',control='matched_double_risk substitutes base for meta: 2*(CE+KL)+L2. This controls risk scale/effective regularization, with same fitting geometry/optimizer. Alpha0.1 and meta weight1 chosen before evaluation; no grid or source-out selection.',limits='Readout diagnostic, not residual expert discovery or SIRM. Held domain excludes diagnostic fitting only, not backbone training. Original source images only. Source-out has already been inspected. No target confirmation or invariant-factor claim.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());si=source['source_in'];x=np.concatenate([si['cls'],si['patch_max']],1).astype(float);rows=[];cases=[]
 for held in plan['held_domains']:
  rr=next(r for r in ref['rows'] if r['held_domain']==held and r['input']=='CLS_plus_patch_max' and r['family']=='unconstrained_centered');f=parent/rr['fit_file'];assert digest(f)==rr['fit_sha256'];locks[str(f)]=digest(f);basis=np.load(f)['basis'];mask=np.ones(len(si['y']),bool) if held is None else si['domains']!=held
  assert min(rr['fit_domain_class_counts'].values())>=2
  for kind in plan['families']:
   m=fit(x[mask],si['global_logits'][mask].astype(float),si['y'][mask],si['domains'][mask],basis,kind);name=f'{kind}_held{held}.npz';np.savez_compressed(OUT/name,**m);row=dict(family=kind,held_domain=held,fit_file=name,fit_sha256=digest(OUT/name),gradient_max=m['gradient_max'],closure_calls=m['closure_calls'],base_risk=m['base_risk'],meta_risk=m['meta_risk'],scores={})
   for split,b in source.items():
    k=mask if split=='source_in' else (np.ones(len(b['y']),bool) if held is None else b['domains']==held);xx=np.concatenate([b['cls'],b['patch_max']],1).astype(float);g=b['global_logits'][k].astype(float);corr=((xx[k]-m['mean'])/m['std'])@m['coef']@basis.T;s,pred=score(g,corr,b['y'][k],b['domains'][k],b['ids'][k]);row['scores'][split]=dict(composed=s,global_control=rr['scores'][split]['global_control'])
    if split=='source_out':
     for i,id in enumerate(b['ids'][k]):
      if pred[i]!=g[i].argmax():cases.append(dict(family=kind,held_domain=held,image_id=str(id),label=int(b['y'][k][i]),global_prediction=int(g[i].argmax()),prediction=int(pred[i])))
   rows.append(row);s=row['scores']['source_out']['composed'];print(held,kind,round(100*s['equal_domain_accuracy'],6),s['repairs'],s['harms'],m['gradient_max'],flush=True)
 for f,h in locks.items():assert digest(Path(f))==h
 assert len(rows)==8;write(OUT/'results.json',dict(plan=plan,rows=rows,input_sha256=locks,source_only=True,target_read=False,bank_updates=False));write(OUT/'cases.json',cases)
if __name__=='__main__':run()
