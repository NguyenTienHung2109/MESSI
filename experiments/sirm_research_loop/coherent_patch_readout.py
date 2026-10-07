"""Matched source-only intact-patch additive readouts versus max-pooling references."""
import json
from pathlib import Path
import numpy as np
import torch
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
from .source_structured_contrast_readout import fit as fit_ce,score
from .source_global_preservation_readout import fit as fit_kl
OUT=DEST/'coherent_patch_readout'

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();cache=DEST/'coherent_patch_source_cache';audit=json.loads((cache/'extraction_audit.json').read_text());assert audit['global_backbone_tensors_exactly_unchanged'] and not audit['target_read'];locks={str(cache/'extraction_audit.json'):digest(cache/'extraction_audit.json')};source={}
 for split in ['source_in','source_out']:
  f=cache/(split+'_tokens.npz');assert digest(f)==next(r['cache_sha256'] for r in audit['rows'] if r['split']==split);locks[str(f)]=digest(f);source[split]=dict(np.load(f))
 parent=DEST/'source_structured_contrast_readout';reference=json.loads((parent/'results.json').read_text());preserve=DEST/'source_global_preservation_readout';klref=json.loads((preserve/'results.json').read_text());locks[str(parent/'results.json')]=digest(parent/'results.json');locks[str(preserve/'results.json')]=digest(preserve/'results.json')
 plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,diagnostic_readout_fitting=True,inputs=['predicted_class_top49','token_norm_top49','image_seeded_random49','patch_mean','patch_max_reference'],held_domains=[None,0,2,3],objectives=['CE','CE_KL'],L2=.01,KL_weight=1.,fit='CLS concatenated with384-dimensional intact-patch pooled vector. Equal-domain source-in feature mean/std, no intercept, centered rank9 output, fixed LBFGS500 stationarity<1e-5. Pooling scores use no true labels/domain.32 new fits+8 reused max references. No source-out choice of pooling or regularization.',limits='Linear readout availability controls, not400-step factor discovery/router/SIRM. Global-predicted-class patch scoring not calibrated localization. Same-dimensional pooling does not isolate semantic causality. Held fitting excludes readout fitting only. Source-out already inspected; no target evaluation.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[];cases=[];si=source['source_in']
 for held in plan['held_domains']:
  rr=next(r for r in reference['rows'] if r['held_domain']==held and r['input']=='CLS_plus_patch_max' and r['family']=='unconstrained_centered');f=parent/rr['fit_file'];assert digest(f)==rr['fit_sha256'];locks[str(f)]=digest(f);basis=np.load(f)['basis'];mask=np.ones(len(si['y']),bool) if held is None else si['domains']!=held
  assert min(rr['fit_domain_class_counts'].values())>=2
  for kind in plan['inputs']:
   for objective in plan['objectives']:
    if kind=='patch_max_reference':
     ref=rr if objective=='CE' else next(r for r in klref['rows'] if r['held_domain']==held and r['family']=='KL_all');f=(parent if objective=='CE' else preserve)/ref['fit_file'];assert digest(f)==ref['fit_sha256'];locks[str(f)]=digest(f);m=dict(np.load(f));pool='patch_max';fitpath=str(f)
    else:
     pool=kind;x=np.concatenate([si['cls'],si[pool]],1).astype(float);args=(x[mask],si['global_logits'][mask].astype(float),si['y'][mask],si['domains'][mask],basis);m=fit_ce(*args) if objective=='CE' else fit_kl(*args,'KL_all');f=OUT/f'{kind}_{objective}_held{held}.npz';np.savez_compressed(f,**m);fitpath=str(f)
    row=dict(pooling=kind,objective=objective,held_domain=held,reused_reference=kind=='patch_max_reference',fit_file=fitpath,fit_sha256=digest(f),gradient_max=float(m['gradient_max']),scores={})
    for split,b in source.items():
     k=mask if split=='source_in' else (np.ones(len(b['y']),bool) if held is None else b['domains']==held);x=np.concatenate([b['cls'],b[pool]],1).astype(float);g=b['global_logits'][k].astype(float);corr=((x[k]-m['mean'])/m['std'])@m['coef']@basis.T;s,pred=score(g,corr,b['y'][k],b['domains'][k],b['ids'][k]);control,_=score(g,np.zeros_like(g),b['y'][k],b['domains'][k],b['ids'][k]);row['scores'][split]=dict(composed=s,global_control=control)
     if kind=='patch_max_reference':
      expected=ref['scores'][split]['composed'];assert abs(s['equal_domain_accuracy']-expected['equal_domain_accuracy'])<1e-12;assert abs(s['equal_domain_CE']-expected['equal_domain_CE'])<1e-10
     if split=='source_out':
      for i,id in enumerate(b['ids'][k]):
       if pred[i]!=g[i].argmax():cases.append(dict(pooling=kind,objective=objective,held_domain=held,image_id=str(id),label=int(b['y'][k][i]),global_prediction=int(g[i].argmax()),prediction=int(pred[i])))
    rows.append(row);s=row['scores']['source_out']['composed'];print(held,kind,objective,round(s['equal_domain_accuracy']*100,6),s['repairs'],s['harms'],flush=True)
 for f,h in locks.items():assert digest(Path(f))==h
 assert len(rows)==40;write(OUT/'results.json',dict(plan=plan,rows=rows,input_sha256=locks,source_only=True,target_read=False,bank_updates=False));write(OUT/'cases.json',cases)
if __name__=='__main__':run()
