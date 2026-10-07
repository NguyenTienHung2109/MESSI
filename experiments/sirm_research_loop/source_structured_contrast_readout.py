"""Source-only centered additive linear readouts with matched output-subspace controls."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import log_softmax
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
OUT=DEST/'source_structured_contrast_readout';L2=.01

def choose_basis(g,y,dom):
 pred=g.argmax(1);counts={}
 for a,b in zip(y,pred):
  if a==b:continue
  key=tuple(sorted((int(a),int(b))));counts[key]=counts.get(key,0)+1
 pairs=sorted(counts,key=lambda k:(-counts[k],k))[:6];assert len(pairs)==6
 axes=np.zeros((6,10))
 for i,(a,b) in enumerate(pairs):axes[i,a]=1/np.sqrt(2);axes[i,b]=-1/np.sqrt(2)
 _,sv,vh=np.linalg.svd(axes,full_matrices=False);rank=int(sum(sv>1e-10));basis=vh[:rank].T
 for i in range(rank):
  if basis[np.abs(basis[:,i]).argmax(),i]<0:basis[:,i]*=-1
 assert np.allclose(basis.T@basis,np.eye(rank)) and np.allclose(basis.sum(0),0)
 return basis,[dict(classes=list(pair),count=counts[pair],domain_counts={str(d):int(((dom==d)&(pred!=y)&(((y==pair[0])&(pred==pair[1]))|((y==pair[1])&(pred==pair[0])))).sum()) for d in sorted(set(dom))}) for pair in pairs]

def fit(x,g,y,dom,basis):
 w=np.array([1/(len(set(dom))*sum(dom==d)) for d in dom]);mean=w@x;std=np.sqrt(w@((x-mean)**2)).clip(1e-8);z=torch.tensor((x-mean)/std,dtype=torch.float64);gg=torch.tensor(g,dtype=torch.float64);yy=torch.tensor(y);ww=torch.tensor(w,dtype=torch.float64);bb=torch.tensor(basis,dtype=torch.float64);coef=torch.zeros((x.shape[1],basis.shape[1]),dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([coef],max_iter=500,tolerance_grad=1e-8,tolerance_change=1e-12,line_search_fn='strong_wolfe');calls=0
 def closure():
  nonlocal calls
  opt.zero_grad();loss=(F.cross_entropy(gg+z@coef@bb.T,yy,reduction='none')*ww).sum()+L2/2*coef.square().sum();loss.backward();calls+=1;return loss
 opt.step(closure);objective=float(closure().detach());grad=float(coef.grad.abs().max());assert grad<1e-5,(grad,calls)
 assert torch.allclose((z@coef@bb.T).mul(ww[:,None]).sum(0),torch.zeros(10,dtype=torch.float64),atol=1e-10)
 effective=coef.detach().numpy()@basis.T;assert np.isclose(np.linalg.norm(effective),float(coef.detach().norm()),rtol=1e-10)
 return dict(mean=mean,std=std,coef=coef.detach().numpy(),basis=basis,objective=objective,gradient_max=grad,closure_calls=calls)

def score(g,corr,y,dom,ids):
 f=g+corr;pred=f.argmax(1);gc=g.argmax(1)==y;correct=pred==y;ce=-log_softmax(f,1)[np.arange(len(y)),y];out=dict(n=len(y),equal_domain_accuracy=float(np.mean([correct[dom==d].mean() for d in sorted(set(dom))])),equal_domain_CE=float(np.mean([ce[dom==d].mean() for d in sorted(set(dom))])),repairs=int((~gc&correct).sum()),harms=int((gc&~correct).sum()),repair_ids=ids[~gc&correct].tolist(),harm_ids=ids[gc&~correct].tolist(),correction_RMS=float(np.sqrt(np.mean(corr**2))),per_domain={str(d):dict(n=int((dom==d).sum()),accuracy=float(correct[dom==d].mean()),CE=float(ce[dom==d].mean())) for d in sorted(set(dom))})
 return out,pred

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();rank=json.loads((DEST/'source_residual_rank_diagnosis/results.json').read_text());source={};locks={}
 for split in ['source_in','source_out']:
  path=ROOT/f'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/{split}_tokens.npz';assert digest(path)==rank['input_sha256'][str(path)];source[split]=dict(np.load(path));locks[str(path)]=digest(path)
 plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,diagnostic_readout_fitting=True,axis_rule='Top6 unordered frozen-global error pairs by source-in occurrence count, lexicographic ties; recompute on fitting domains only for each held control. SVD orthonormal span, not six independently identified factors.',families=['confusion_span','random_matched_rank','unconstrained_centered'],inputs=['CLS','CLS_plus_patch_max'],held_domains=[None,0,2,3],regularization=L2,fit='Original source-in equal-domain CE of frozen_global+linear correction; no intercept. Equal-domain source-in feature centering/std. LBFGS500 stationarity<1e-5; fixed L2/no source-out selection. Orthonormal output bases make coefficient L2 equal effective logit-map L2 across all families. Random rank control seeded0, class-centered QR.',limits='This is an additive linear classifier control, not400-step neural discovery/router training or ON/OFF SIRM. Axis labels do not establish semantic visual factors. Held fitting excludes output-axis selection/calibration only; frozen backbone already learned all sources. Source-out was previously inspected. Minimum2 source-in images per present domain/class cell verified; error-pair counts are not SIRM eligibility.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[];cases=[];si=source['source_in'];so=source['source_out'];assert not set(si['ids'])&set(so['ids']);_,_,vh=np.linalg.svd(np.eye(10)-np.ones((10,10))/10);fullbasis=vh[:9].T;assert np.allclose(fullbasis.sum(0),0)
 for held in plan['held_domains']:
  mask=np.ones(len(si['y']),bool) if held is None else si['domains']!=held;y=si['y'][mask];dom=si['domains'][mask];g=si['global_logits'][mask].astype(float);counts={f'{d},{c}':int(((dom==d)&(y==c)).sum()) for d in sorted(set(dom)) for c in sorted(set(y)) if ((dom==d)&(y==c)).any()};assert min(counts.values())>=2;basis,pairs=choose_basis(g,y,dom);rng=np.random.default_rng(0);rr=rng.normal(size=(10,basis.shape[1]));rr-=rr.mean(0);random=np.linalg.qr(rr)[0];assert np.allclose(random.sum(0),0) and np.allclose(random.T@random,np.eye(basis.shape[1]));bases=dict(confusion_span=basis,random_matched_rank=random,unconstrained_centered=fullbasis)
  for kind in plan['inputs']:
   xx=np.concatenate([si['cls'],si['patch_max']],1).astype(float) if kind=='CLS_plus_patch_max' else si['cls'].astype(float)
   for family,bb in bases.items():
    model=fit(xx[mask],g,y,dom,bb);name=f'{kind}_{family}_held{held}.npz';np.savez_compressed(OUT/name,**model);row=dict(held_domain=held,input=kind,family=family,fit_rows=int(mask.sum()),fit_domain_class_counts=counts,output_rank=bb.shape[1],source_in_confusion_pairs=pairs,fit_file=name,fit_sha256=digest(OUT/name),gradient_max=model['gradient_max'],objective=model['objective'],closure_calls=model['closure_calls'],coefficient_count=int(model['coef'].size),scores={})
    for split,b in source.items():
     k=mask if split=='source_in' else (np.ones(len(b['y']),bool) if held is None else b['domains']==held);x=np.concatenate([b['cls'],b['patch_max']],1).astype(float) if kind=='CLS_plus_patch_max' else b['cls'].astype(float);gg=b['global_logits'][k].astype(float);corr=((x[k]-model['mean'])/model['std'])@model['coef']@bb.T;metrics,pred=score(gg,corr,b['y'][k],b['domains'][k],b['ids'][k]);control,_=score(gg,np.zeros_like(gg),b['y'][k],b['domains'][k],b['ids'][k]);row['scores'][split]=dict(composed=metrics,global_control=control)
     if split=='source_out':
      for i,id in enumerate(b['ids'][k]):
       if pred[i]!=gg[i].argmax():cases.append(dict(held_domain=held,input=kind,family=family,image_id=str(id),label=int(b['y'][k][i]),global_prediction=int(gg[i].argmax()),prediction=int(pred[i])))
    rows.append(row);print(held,kind,family,'rank',bb.shape[1],'val',round(row['scores']['source_out']['composed']['equal_domain_accuracy']*100,5),flush=True)
 for f,h in locks.items():assert digest(Path(f))==h
 assert len(rows)==24
 write(OUT/'results.json',dict(plan=plan,rows=rows,input_sha256=locks,source_only=True,target_read=False,bank_updates=False));write(OUT/'cases.json',cases)
if __name__=='__main__':run()
