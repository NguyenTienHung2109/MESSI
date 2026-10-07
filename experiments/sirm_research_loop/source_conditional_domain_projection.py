"""Source-only residual-input domain-centroid projection with matched controls."""
import json,itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
from .source_structured_contrast_readout import score
OUT=DEST/'source_conditional_domain_projection';L2=.01

def project(z,axes):return z-(z@axes.T)@axes

def geometry(z,y,dom):
 cells={(int(d),int(c)):z[(dom==d)&(y==c)].mean(0) for d in sorted(set(dom)) for c in sorted(set(y)) if ((dom==d)&(y==c)).sum()>=2};within=[];means={}
 for c in sorted(set(y)):
  vs=[v for (d,cl),v in cells.items() if cl==c]
  if len(vs):means[int(c)]=np.mean(vs,0)
  within.extend(float(np.linalg.norm(a-b)) for a,b in itertools.combinations(vs,2))
 between=[float(np.linalg.norm(a-b)) for a,b in itertools.combinations(means.values(),2)]
 return dict(class_conditioned_domain_centroid_distance=float(np.mean(within)) if within else None,valid_domain_centroid_pairs=len(within),between_class_centroid_distance=float(np.mean(between)) if between else None,valid_class_pairs=len(between),feature_RMS=float(np.sqrt(np.mean(z**2))))

def fit(z,g,y,dom,axes,basis,kind):
 projected=project(z,axes);zz=torch.tensor(projected,dtype=torch.float64);gg=torch.tensor(g,dtype=torch.float64);yy=torch.tensor(y);w=np.array([1/(len(set(dom))*sum(dom==d)) for d in dom]);ww=torch.tensor(w,dtype=torch.float64);bb=torch.tensor(basis,dtype=torch.float64);pg=gg.softmax(1);coef=torch.zeros((z.shape[1],basis.shape[1]),dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([coef],max_iter=500,tolerance_grad=1e-8,tolerance_change=1e-12,line_search_fn='strong_wolfe');calls=0
 def closure():
  nonlocal calls
  opt.zero_grad();full=gg+zz@coef@bb.T;ce=(F.cross_entropy(full,yy,reduction='none')*ww).sum();kl=(pg*(pg.log()-full.log_softmax(1))).sum(1);loss=ce+(kl*ww).sum()*(kind=='CE_KL')+L2/2*coef.square().sum();loss.backward();calls+=1;return loss
 opt.step(closure);objective=float(closure().detach());grad=float(coef.grad.abs().max());assert grad<1e-5,(kind,grad,calls);co=coef.detach().numpy();assert np.abs(axes@co).max(initial=0)<1e-8
 return dict(coef=co,basis=basis,removed_axes=axes,objective=objective,gradient_max=grad,closure_calls=calls)

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();cp=DEST/'source_structured_contrast_readout';kp=DEST/'source_global_preservation_readout';cr=json.loads((cp/'results.json').read_text());kr=json.loads((kp/'results.json').read_text());locks={str(cp/'results.json'):digest(cp/'results.json'),str(kp/'results.json'):digest(kp/'results.json')};source={}
 for split in ['source_in','source_out']:
  f=ROOT/f'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/{split}_tokens.npz';assert digest(f)==cr['input_sha256'][str(f)];locks[str(f)]=digest(f);source[split]=dict(np.load(f))
 plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,diagnostic_readout_fitting=True,input='CLS_plus_patch_max',projectors=['conditional_domain','random_matched_rank','PCA_matched_rank'],objectives=['CE','CE_KL'],held_domains=[None,0,2,3],regularization=L2,KL_weight=1.,projection='Span of within-class source-domain centroid minus equal-domain class mean; all10 fitting classes/domain cells min2. SVD numerical rank >1e-10; full-source rank20, two-domain controls at most10. Projection/basis/standardization fit on allowed source-in only.',controls='Random seeded0 orthonormal rank match; unlabelled fitting-data top PCA directions same rank. All preserve frozen global logits; only additive residual inputs projected. Effective input-map L2 is matched in original standardized coordinates. No amplitude rescaling. Record removed variance and class separation to distinguish generic feature/capacity loss.',fit='Reuse locked original-source-in mean/std, rank9 centered output basis, zero intercept. Source-only equal-domain CE or CE+globalKL1, LBFGS500 stationarity<1e-5. No source-out selection; reused no-projection references.',limits='Centroid equality on fitting data is by construction; not proof of domain-information removal or invariant visual factors. Sparse cell means are noisy. Projection labels are descriptive domain contrast, not established nuisance/causal axes. Held fitting does not hold out backbone learning. Source-out already inspected; no400-step discovery/router or SIRM.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[];cases=[];si=source['source_in'];sx=np.concatenate([si['cls'],si['patch_max']],1).astype(float)
 for held in plan['held_domains']:
  ref=next(r for r in cr['rows'] if r['held_domain']==held and r['input']==plan['input'] and r['family']=='unconstrained_centered');refkl=next(r for r in kr['rows'] if r['held_domain']==held and r['family']=='KL_all');f=cp/ref['fit_file'];assert digest(f)==ref['fit_sha256'];locks[str(f)]=digest(f);normal=dict(np.load(f));mask=np.ones(len(si['y']),bool) if held is None else si['domains']!=held;y=si['y'][mask];dom=si['domains'][mask];g=si['global_logits'][mask].astype(float);z=(sx[mask]-normal['mean'])/normal['std'];counts={f'{d},{c}':int(((dom==d)&(y==c)).sum()) for d in sorted(set(dom)) for c in range(10)};assert min(counts.values())>=2
  centroids=[]
  for c in range(10):
   means=np.stack([z[(dom==d)&(y==c)].mean(0) for d in sorted(set(dom))]);centroids.extend(means-means.mean(0))
  _,sv,vh=np.linalg.svd(np.array(centroids),full_matrices=False);rank=int((sv>1e-10).sum());axes=vh[:rank];assert rank==10*(len(set(dom))-1);rng=np.random.default_rng(0);random=np.linalg.qr(rng.normal(size=(768,rank)))[0].T;w=np.array([1/(len(set(dom))*sum(dom==d)) for d in dom]);_,_,pv=np.linalg.svd(z*np.sqrt(w[:,None]),full_matrices=False);projectors=dict(conditional_domain=axes,random_matched_rank=random,PCA_matched_rank=pv[:rank],none=np.zeros((0,768)));original_geometry=geometry(z,y,dom);assert geometry(project(z,axes),y,dom)['class_conditioned_domain_centroid_distance']<1e-8
  for name,aa in projectors.items():
   assert np.allclose(aa@aa.T,np.eye(len(aa)),atol=1e-10);assert np.allclose(project(project(z,aa),aa),project(z,aa),atol=1e-10)
   for kind in plan['objectives']:
    if name=='none':
     parent,rr=(cp,ref) if kind=='CE' else (kp,refkl);f=parent/rr['fit_file'];assert digest(f)==rr['fit_sha256'];locks[str(f)]=digest(f);model=dict(np.load(f));model['removed_axes']=aa;reused=True
    else:
     model=fit(z,g,y,dom,aa,normal['basis'],kind);model.update(mean=normal['mean'],std=normal['std']);f=OUT/f'{name}_{kind}_held{held}.npz';np.savez_compressed(f,**model);reused=False
    pg=geometry(project(z,aa),y,dom);row=dict(held_domain=held,projection=name,objective=kind,reused_reference=reused,removed_rank=len(aa),fit_file=str(f),fit_sha256=digest(f),gradient_max=float(model['gradient_max']),fitting_domain_class_counts=counts,source_in_original_geometry=original_geometry,source_in_projected_geometry=pg,removed_source_in_RMS_energy_fraction=1-(pg['feature_RMS']/original_geometry['feature_RMS'])**2,scores={})
    for split,b in source.items():
     k=mask if split=='source_in' else (np.ones(len(b['y']),bool) if held is None else b['domains']==held);xx=np.concatenate([b['cls'],b['patch_max']],1).astype(float);zz=(xx[k]-normal['mean'])/normal['std'];zp=project(zz,aa);gg=b['global_logits'][k].astype(float);corr=zp@model['coef']@normal['basis'].T;metrics,pred=score(gg,corr,b['y'][k],b['domains'][k],b['ids'][k]);globalmetrics,_=score(gg,np.zeros_like(gg),b['y'][k],b['domains'][k],b['ids'][k]);row['scores'][split]=dict(composed=metrics,global_control=globalmetrics,original_geometry=geometry(zz,b['y'][k],b['domains'][k]),projected_geometry=geometry(zp,b['y'][k],b['domains'][k]))
     if split=='source_out':
      for i,id in enumerate(b['ids'][k]):
       if pred[i]!=gg[i].argmax():cases.append(dict(projection=name,objective=kind,held_domain=held,image_id=str(id),label=int(b['y'][k][i]),global_prediction=int(gg[i].argmax()),prediction=int(pred[i])))
    rows.append(row);v=row['scores']['source_out']['composed'];print(held,name,kind,'rank',len(aa),'acc',round(v['equal_domain_accuracy']*100,5),'rep/harm',v['repairs'],v['harms'],flush=True)
 for f,h in locks.items():assert digest(Path(f))==h
 assert len(rows)==32 and sum(not r['reused_reference'] for r in rows)==24
 write(OUT/'results.json',dict(plan=plan,rows=rows,input_sha256=locks,source_only=True,target_read=False,bank_updates=False));write(OUT/'cases.json',cases)
if __name__=='__main__':run()
