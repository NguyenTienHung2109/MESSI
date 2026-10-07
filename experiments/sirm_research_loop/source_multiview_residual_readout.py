"""Source-only additive readouts trained on existing matched H2 views."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .routed_discovery_run import ROOT,DEST
from .routed_conditional_utility import load
from .shrinkage import digest,write
from .source_structured_contrast_readout import score
OUT=DEST/'source_multiview_residual_readout';L2=.01

def weights(dom):return np.array([1/(len(set(dom))*sum(dom==d)) for d in dom])
def fit(x,g,y,w,geometry,kind):
 mean,std,basis=[geometry[k] for k in ['mean','std','basis']];z=torch.tensor((x-mean)/std,dtype=torch.float64);gg=torch.tensor(g,dtype=torch.float64);yy=torch.tensor(y);ww=torch.tensor(w,dtype=torch.float64);bb=torch.tensor(basis,dtype=torch.float64);pg=gg.softmax(1);coef=torch.zeros((x.shape[1],basis.shape[1]),dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([coef],max_iter=500,tolerance_grad=1e-8,tolerance_change=1e-12,line_search_fn='strong_wolfe');calls=0
 def closure():
  nonlocal calls
  opt.zero_grad();full=gg+z@coef@bb.T;ce=(F.cross_entropy(full,yy,reduction='none')*ww).sum();kl=(pg*(pg.log()-full.log_softmax(1))).sum(1);loss=ce+(kl*ww).sum()*(kind=='CE_KL')+L2/2*coef.square().sum();loss.backward();calls+=1;return loss
 opt.step(closure);objective=float(closure().detach());grad=float(coef.grad.abs().max());assert grad<1e-5,(kind,grad,calls)
 return dict(mean=mean,std=std,basis=basis,coef=coef.detach().numpy(),objective=objective,gradient_max=grad,closure_calls=calls)

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();ceparent=DEST/'source_structured_contrast_readout';klparent=DEST/'source_global_preservation_readout';ceref=json.loads((ceparent/'results.json').read_text());klref=json.loads((klparent/'results.json').read_text());locks={str(ceparent/'results.json'):digest(ceparent/'results.json'),str(klparent/'results.json'):digest(klparent/'results.json')};source={}
 for split in ['source_in','source_out']:
  f=ROOT/f'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/{split}_tokens.npz';assert digest(f)==ceref['input_sha256'][str(f)];locks[str(f)]=digest(f);source[split]=dict(np.load(f))
 m,p,path=load(DEST,'selected');locks[str(path)]=digest(path);before={k:v.clone() for k,v in m.state_dict().items()};assert all(not v.requires_grad for module in [m.featurizer,m.global_head] for v in module.parameters());proofpath=DEST/'source_conditional_H2/extraction_audit.json';proof=json.loads(proofpath.read_text());locks[str(proofpath)]=digest(proofpath);raws=[];gs=[];ys=[];doms=[];ids=[]
 with torch.no_grad():
  for rec in proof['rows']:
   f=DEST/'source_conditional_H2'/f"H2_step{rec['step']}.pt";assert digest(f)==rec['sha256'];locks[str(f)]=digest(f);b=torch.load(f,map_location='cpu',weights_only=False);raws.append(b['raw'].numpy());gs.append(m.global_head(b['raw'][:,:384]).numpy());ys.append(b['y'].numpy());doms.append(b['domains'].numpy());ids+=b['ids']
  original=np.concatenate([source['source_in']['cls'],source['source_in']['patch_max']],1);err=float(np.abs(m.global_head(torch.tensor(original[:,:384])).numpy()-source['source_in']['global_logits']).max());assert err<1e-4
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and all(v.grad is None for v in m.parameters());del m,before
 h2=dict(x=np.concatenate(raws).astype(float),g=np.concatenate(gs).astype(float),y=np.concatenate(ys),dom=np.concatenate(doms),ids=np.array(ids));si=source['source_in'];sx=np.concatenate([si['cls'],si['patch_max']],1).astype(float);assert set(ids)<=set(si['ids']) and not set(ids)&set(source['source_out']['ids'])
 plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,diagnostic_readout_fitting=True,input='CLS_plus_patch_max',cohorts=['original_reference','H2','joint50_50'],objectives=['CE','CE_KL'],held_domains=[None,0,2,3],regularization=L2,KL_weight=1.,risk='Equal-domain means within each cohort; joint half original/half H2. All40 actual saved H2 batches,3840 occurrences. Verify minimum2 distinct source-in IDs per present fitting domain/class. Repeated views not independent.',geometry='Reuse each held calibration original-source-in mean/std and rank9 orthonormal class-centered basis, no intercept; only fitting risk/cohort changes. This deliberately isolates augmented supervision from changed normalization and preserves zero mean correction on original source-in.',fit='Fixed CE(+global KL_all), LBFGS500 gradient<1e-5, no validation selection. Global predictions on H2 recomputed from exactly frozen global head, original-logit replay verified.',limits='Additive linear classifier controls, not400-step factor discovery/router or ON/OFF SIRM. H2 original cache means differ by view; held fitting does not hold out backbone learning. Source-out already inspected. No target inference or accuracy record.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[];cases=[]
 for held in plan['held_domains']:
  ref=next(r for r in ceref['rows'] if r['held_domain']==held and r['input']==plan['input'] and r['family']=='unconstrained_centered');kr=next(r for r in klref['rows'] if r['held_domain']==held and r['family']=='KL_all');f=ceparent/ref['fit_file'];assert digest(f)==ref['fit_sha256'];locks[str(f)]=digest(f);geometry=dict(np.load(f));sm=np.ones(len(si['y']),bool) if held is None else si['domains']!=held;hm=np.ones(len(h2['y']),bool) if held is None else h2['dom']!=held;hc={f'{d},{c}':len(set(h2['ids'][hm&(h2['dom']==d)&(h2['y']==c)])) for d in sorted(set(h2['dom'][hm])) for c in range(10)};assert min(hc.values())>=2
  for cohort in plan['cohorts']:
   for kind in plan['objectives']:
    if cohort=='original_reference':
     parent,rr=(ceparent,ref) if kind=='CE' else (klparent,kr);f=parent/rr['fit_file'];assert digest(f)==rr['fit_sha256'];locks[str(f)]=digest(f);model=dict(np.load(f));reused=True
    else:
     if cohort=='H2':x,g,y,w=h2['x'][hm],h2['g'][hm],h2['y'][hm],weights(h2['dom'][hm])
     else:x=np.concatenate([sx[sm],h2['x'][hm]]);g=np.concatenate([si['global_logits'][sm].astype(float),h2['g'][hm]]);y=np.concatenate([si['y'][sm],h2['y'][hm]]);w=np.concatenate([weights(si['domains'][sm])*.5,weights(h2['dom'][hm])*.5])
     assert np.isclose(w.sum(),1);model=fit(x,g,y,w,geometry,kind);f=OUT/f'{cohort}_{kind}_held{held}.npz';np.savez_compressed(f,**model);reused=False
    row=dict(cohort=cohort,objective=kind,held_domain=held,reused_reference=reused,fit_file=str(f),fit_sha256=digest(f),gradient_max=float(model['gradient_max']),H2_distinct_image_domain_class_counts=hc,scores={})
    for split,b in source.items():
     mask=sm if split=='source_in' else (np.ones(len(b['y']),bool) if held is None else b['domains']!=held if False else b['domains']==held);xx=np.concatenate([b['cls'],b['patch_max']],1).astype(float);gg=b['global_logits'][mask].astype(float);corr=((xx[mask]-model['mean'])/model['std'])@model['coef']@model['basis'].T;metric,pred=score(gg,corr,b['y'][mask],b['domains'][mask],b['ids'][mask]);control,_=score(gg,np.zeros_like(gg),b['y'][mask],b['domains'][mask],b['ids'][mask]);row['scores'][split]=dict(composed=metric,global_control=control)
     if split=='source_out':
      for i,id in enumerate(b['ids'][mask]):
       if pred[i]!=gg[i].argmax():cases.append(dict(cohort=cohort,objective=kind,held_domain=held,image_id=str(id),label=int(b['y'][mask][i]),global_prediction=int(gg[i].argmax()),prediction=int(pred[i])))
    rows.append(row);v=row['scores']['source_out']['composed'];print(held,cohort,kind,round(v['equal_domain_accuracy']*100,5),v['repairs'],v['harms'],flush=True)
 for f,h in locks.items():assert digest(Path(f))==h
 assert len(rows)==24 and sum(not r['reused_reference'] for r in rows)==16
 write(OUT/'results.json',dict(plan=plan,rows=rows,input_sha256=locks,source_only=True,target_read=False,bank_updates=False,original_global_logit_replay_max_error=err));write(OUT/'cases.json',cases)
if __name__=='__main__':run()
