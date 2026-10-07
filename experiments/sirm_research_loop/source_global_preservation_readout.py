"""Convex source-only global-preservation residual calibration controls."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
from .source_structured_contrast_readout import score
OUT=DEST/'source_global_preservation_readout';L2=.01

def fit(x,g,y,dom,basis,kind):
 w=np.array([1/(len(set(dom))*sum(dom==d)) for d in dom]);correct=g.argmax(1)==y;wc=np.array([1/(len(set(dom))*sum((dom==d)&correct)) if correct[i] else 0 for i,d in enumerate(dom)]);mean=w@x;std=np.sqrt(w@((x-mean)**2)).clip(1e-8);z=torch.tensor((x-mean)/std,dtype=torch.float64);gg=torch.tensor(g,dtype=torch.float64);yy=torch.tensor(y);ww=torch.tensor(w,dtype=torch.float64);wcorrect=torch.tensor(wc,dtype=torch.float64);bb=torch.tensor(basis,dtype=torch.float64);pg=gg.softmax(1);coef=torch.zeros((x.shape[1],basis.shape[1]),dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([coef],max_iter=500,tolerance_grad=1e-8,tolerance_change=1e-12,line_search_fn='strong_wolfe');calls=0
 def losses():
  corr=z@coef@bb.T;full=gg+corr;ce=(F.cross_entropy(full,yy,reduction='none')*ww).sum();kl=(pg*(pg.log()-full.log_softmax(1))).sum(1)
  if kind=='KL_all':penalty=(kl*ww).sum()
  elif kind=='KL_global_correct':penalty=(kl*wcorrect).sum()
  elif kind=='squared_margin_global_correct':
   # Preserve all correct-global y-versus-c competitor contrasts; soft penalty, not exact guarantee.
   contrast=corr[torch.arange(len(y)),yy,None]-corr;violation=F.relu(-contrast).square().sum(1)/9;penalty=(violation*wcorrect).sum()
  else:raise ValueError(kind)
  return ce,penalty,corr
 def closure():
  nonlocal calls
  opt.zero_grad();ce,penalty,_=losses();loss=ce+penalty+L2/2*coef.square().sum();loss.backward();calls+=1;return loss
 opt.step(closure);objective=float(closure().detach());grad=float(coef.grad.abs().max());assert grad<1e-5,(kind,grad,calls);ce,penalty,corr=losses();assert torch.allclose((corr*ww[:,None]).sum(0),torch.zeros(10,dtype=torch.float64),atol=1e-10)
 # KL_all equivalence checks correct implementation of the distillation direction.
 if kind=='KL_all':assert torch.allclose(ce+penalty,(-(F.one_hot(yy,10)+pg)*(gg+corr).log_softmax(1)).sum(1).mul(ww).sum()+(pg*pg.log()).sum(1).mul(ww).sum(),atol=1e-10)
 return dict(mean=mean,std=std,coef=coef.detach().numpy(),basis=basis,objective=objective,gradient_max=grad,closure_calls=calls,source_in_CE=float(ce.detach()),source_in_penalty=float(penalty.detach()))

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();parent=DEST/'source_structured_contrast_readout';reference=json.loads((parent/'results.json').read_text());locks={str(parent/'results.json'):digest(parent/'results.json')};source={}
 for split in ['source_in','source_out']:
  f=ROOT/f'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/{split}_tokens.npz';assert digest(f)==reference['input_sha256'][str(f)];locks[str(f)]=digest(f);source[split]=dict(np.load(f))
 plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,diagnostic_readout_fitting=True,input='CLS_plus_patch_max',output_basis='Same unconstrained rank9 centered class basis as CE reference.',families=['KL_all','KL_global_correct','squared_margin_global_correct'],held_domains=[None,0,2,3],L2=L2,preservation_weight=1.,fit='Fixed source-in equal-domain CE plus preservation penalty, source-in feature mean/std, zero-intercept additive linear correction, LBFGS500 gradient<1e-5. No source-out selection. Correct-global subsets use source labels and are equally normalized within each fitting domain.',control='KL_all versus correct-global KL distinguishes global distillation from label-selected preservation. Squared margin penalty averages nine true-class/competitor negative correction-contrast violations; soft penalty, not a hard constraint. Weight1 fixed for these local readout controls, not a SIRM lambda update.',limits='Linear classifier controls, not400-step factor discovery/router or subset SIRM. Lower accuracy changes are not independent replications. Held calibration excludes fitting only, not frozen backbone pretraining. Source-out already inspected. Minimum2 fitting images per present domain/class verified.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[];cases=[];si=source['source_in'];so=source['source_out'];x=np.concatenate([si['cls'],si['patch_max']],1).astype(float)
 for held in plan['held_domains']:
  ref=next(r for r in reference['rows'] if r['held_domain']==held and r['input']==plan['input'] and r['family']=='unconstrained_centered');file=parent/ref['fit_file'];assert digest(file)==ref['fit_sha256'];locks[str(file)]=digest(file);basis=np.load(file)['basis'];mask=np.ones(len(si['y']),bool) if held is None else si['domains']!=held;assert min(ref['fit_domain_class_counts'].values())>=2
  for kind in plan['families']:
   model=fit(x[mask],si['global_logits'][mask].astype(float),si['y'][mask],si['domains'][mask],basis,kind);name=f'{kind}_held{held}.npz';np.savez_compressed(OUT/name,**model);row=dict(family=kind,held_domain=held,fit_file=name,fit_sha256=digest(OUT/name),gradient_max=model['gradient_max'],objective=model['objective'],source_in_penalty=model['source_in_penalty'],reference_CE_fit_sha256=ref['fit_sha256'],scores={})
   for split,b in source.items():
    k=mask if split=='source_in' else (np.ones(len(b['y']),bool) if held is None else b['domains']==held);xx=np.concatenate([b['cls'],b['patch_max']],1).astype(float);g=b['global_logits'][k].astype(float);corr=((xx[k]-model['mean'])/model['std'])@model['coef']@basis.T;metrics,pred=score(g,corr,b['y'][k],b['domains'][k],b['ids'][k]);yy=b['y'][k];gc=g.argmax(1)==yy;contrast=corr[np.arange(len(corr)),yy,None]-corr;metrics['correct_global_any_negative_correction_contrast_fraction']=float((contrast[gc].min(1)<-1e-8).mean());metrics['correct_global_squared_contrast_violation']=float((np.maximum(-contrast[gc],0)**2).sum(1).mean()/9);row['scores'][split]=dict(composed=metrics,global_control=ref['scores'][split]['global_control'],CE_fit_reference=ref['scores'][split]['composed'])
    if split=='source_out':
     for i,id in enumerate(b['ids'][k]):
      if pred[i]!=g[i].argmax():cases.append(dict(family=kind,held_domain=held,image_id=str(id),label=int(yy[i]),global_prediction=int(g[i].argmax()),prediction=int(pred[i])))
   rows.append(row);a=row['scores']['source_out']['composed'];print(held,kind,round(a['equal_domain_accuracy']*100,5),a['repairs'],a['harms'],flush=True)
 for f,h in locks.items():assert digest(Path(f))==h
 assert len(rows)==12
 write(OUT/'results.json',dict(plan=plan,rows=rows,input_sha256=locks,source_only=True,target_read=False,bank_updates=False));write(OUT/'cases.json',cases)
if __name__=='__main__':run()
