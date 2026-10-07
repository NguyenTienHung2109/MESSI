"""Isolate domain/class risk weighting from frozen additive readout capacity."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
from .source_structured_contrast_readout import score
OUT=DEST/'source_domain_class_weighting';L2=.01

def fit(x,g,y,dom,basis,kind):
 domains=sorted(set(dom));labels=sorted(set(y));assert len(labels)==10
 wd=np.array([1/(len(domains)*sum(dom==d)) for d in dom]);wc=np.array([1/(len(domains)*len(labels)*sum((dom==d)&(y==c))) for d,c in zip(dom,y)]);assert np.isclose(wd.sum(),1) and np.isclose(wc.sum(),1)
 # Preserve input geometry/centering from the reference: only risk weights change.
 mean=wd@x;std=np.sqrt(wd@((x-mean)**2)).clip(1e-8);z=torch.tensor((x-mean)/std,dtype=torch.float64);gg=torch.tensor(g,dtype=torch.float64);yy=torch.tensor(y);ww=torch.tensor(wc,dtype=torch.float64);kw=torch.tensor(wd if kind=='CE_class_KL_domain' else wc,dtype=torch.float64);bb=torch.tensor(basis,dtype=torch.float64);pg=gg.softmax(1);coef=torch.zeros((x.shape[1],basis.shape[1]),dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([coef],max_iter=500,tolerance_grad=1e-8,tolerance_change=1e-12,line_search_fn='strong_wolfe');calls=0
 def closure():
  nonlocal calls
  opt.zero_grad();full=gg+z@coef@bb.T;ce=(F.cross_entropy(full,yy,reduction='none')*ww).sum();kl=(pg*(pg.log()-full.log_softmax(1))).sum(1);penalty=(kl*kw).sum() if kind!='CE_class' else ce*0;loss=ce+penalty+L2/2*coef.square().sum();loss.backward();calls+=1;return loss
 opt.step(closure);objective=float(closure().detach());grad=float(coef.grad.abs().max());assert grad<1e-5,(kind,grad,calls);corr=z@coef@bb.T;assert torch.allclose((corr*torch.tensor(wd)[:,None]).sum(0),torch.zeros(10,dtype=torch.float64),atol=1e-10)
 return dict(mean=mean,std=std,coef=coef.detach().numpy(),basis=basis,objective=objective,gradient_max=grad,closure_calls=calls),dict(class_weight_to_domain_weight_min=float((wc/wd).min()),class_weight_to_domain_weight_max=float((wc/wd).max()),domain_class_counts={f'{d},{c}':int(((dom==d)&(y==c)).sum()) for d in domains for c in labels})

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();base=DEST/'source_structured_contrast_readout';klbase=DEST/'source_global_preservation_readout';reference=json.loads((base/'results.json').read_text());klreference=json.loads((klbase/'results.json').read_text());locks={str(base/'results.json'):digest(base/'results.json'),str(klbase/'results.json'):digest(klbase/'results.json')};source={}
 for split in ['source_in','source_out']:
  f=ROOT/f'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/{split}_tokens.npz';assert digest(f)==reference['input_sha256'][str(f)];locks[str(f)]=digest(f);source[split]=dict(np.load(f))
 plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,diagnostic_readout_fitting=True,input='CLS_plus_patch_max',families=['CE_class','CE_class_KL_domain','CE_class_KL_class'],reference_families=['CE_domain','CE_domain_KL_domain'],held_domains=[None,0,2,3],L2=L2,KL_weight=1.,weighting='Class weight1/(D*10*N_domain,class), domain weight1/(D*N_domain). All fitting domain/class cells minimum2. Feature standardization, equal-domain input centering, rank9 class-centered basis, zero intercept and effective map L2 remain identical to prior reference. No source-out choice.',fit='Original source-in convex CE(+KL) objective, LBFGS500 gradient<1e-5. CE_class_KL_domain changes supervised risk only; CE_class_KL_class also reweights global preservation. Reuse locked prior equal-domain CE/KL fits.',limits='This is classifier risk calibration, not neural factor discovery or subset SIRM. Primary source score remains equal-domain image accuracy; domain/class-balanced accuracy is a supplementary different metric. Source-out already inspected; held fit excludes calibration only, not backbone learning.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());si=source['source_in'];x=np.concatenate([si['cls'],si['patch_max']],1).astype(float);rows=[];cases=[]
 for held in plan['held_domains']:
  ref=next(r for r in reference['rows'] if r['held_domain']==held and r['input']==plan['input'] and r['family']=='unconstrained_centered');refkl=next(r for r in klreference['rows'] if r['held_domain']==held and r['family']=='KL_all');f=base/ref['fit_file'];assert digest(f)==ref['fit_sha256'];locks[str(f)]=digest(f);basis=np.load(f)['basis'];mask=np.ones(len(si['y']),bool) if held is None else si['domains']!=held;dom=si['domains'][mask];y=si['y'][mask];g=si['global_logits'][mask].astype(float);fits={}
  for family in plan['families']:
   model,coverage=fit(x[mask],g,y,dom,basis,family);assert min(coverage['domain_class_counts'].values())>=2;name=f'{family}_held{held}.npz';np.savez_compressed(OUT/name,**model);fits[family]=(model,OUT/name,coverage,False)
  for family,parent,rr in [('CE_domain',base,ref),('CE_domain_KL_domain',klbase,refkl)]:
   f=parent/rr['fit_file'];assert digest(f)==rr['fit_sha256'];locks[str(f)]=digest(f);fits[family]=(dict(np.load(f)),f,dict(domain_class_counts=ref['fit_domain_class_counts']),True)
  for family,(model,f,coverage,reused) in fits.items():
   row=dict(held_domain=held,family=family,reused_reference=reused,fit_file=str(f),fit_sha256=digest(f),gradient_max=float(model['gradient_max']),coverage=coverage,scores={})
   for split,b in source.items():
    k=mask if split=='source_in' else (np.ones(len(b['y']),bool) if held is None else b['domains']==held);xx=np.concatenate([b['cls'],b['patch_max']],1).astype(float);gg=b['global_logits'][k].astype(float);corr=((xx[k]-model['mean'])/model['std'])@model['coef']@basis.T;yy=b['y'][k];dd=b['domains'][k];ids=b['ids'][k];metrics,pred=score(gg,corr,yy,dd,ids);control,_=score(gg,np.zeros_like(gg),yy,dd,ids)
    def balanced(pp):return float(np.mean([np.mean([np.mean(pp[(dd==domain)&(yy==label)]==label) for label in sorted(set(yy[dd==domain]))]) for domain in sorted(set(dd))]))
    metrics['equal_domain_present_class_accuracy']=balanced(pred);control['equal_domain_present_class_accuracy']=balanced(gg.argmax(1));metrics['per_class']={str(label):dict(n=int((yy==label).sum()),correct=int(((yy==label)&(pred==yy)).sum()),global_correct=int(((yy==label)&(gg.argmax(1)==yy)).sum())) for label in sorted(set(yy))};row['scores'][split]=dict(composed=metrics,global_control=control)
    if split=='source_out':
     for i,id in enumerate(ids):
      if pred[i]!=gg[i].argmax():cases.append(dict(family=family,held_domain=held,image_id=str(id),label=int(yy[i]),global_prediction=int(gg[i].argmax()),prediction=int(pred[i])))
   rows.append(row);a=row['scores']['source_out']['composed'];print(held,family,round(a['equal_domain_accuracy']*100,5),a['repairs'],a['harms'],'balanced',round(a['equal_domain_present_class_accuracy']*100,4),flush=True)
 for f,h in locks.items():assert digest(Path(f))==h
 assert len(rows)==20 and sum(not r['reused_reference'] for r in rows)==12
 write(OUT/'results.json',dict(plan=plan,rows=rows,input_sha256=locks,source_only=True,target_read=False,bank_updates=False));write(OUT/'cases.json',cases)
if __name__=='__main__':run()
