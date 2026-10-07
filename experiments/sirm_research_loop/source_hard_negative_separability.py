"""Source-only conditional label separability and within-location neighbors."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
from .label_expected_utility_helpers import fit_label,predict_label
OUT=DEST/'source_hard_negative_separability';LABELS=[1,6,8]

def metrics(p,y,dom,labels):
 pred=np.array(labels)[p.argmax(1)];ce=-np.log(p[np.arange(len(y)),np.array([labels.index(int(i)) for i in y])].clip(1e-300));cells=[]
 for domain in sorted(set(dom)):
  for label in labels:
   mask=(dom==domain)&(y==label)
   if not mask.any():continue
   cells.append(dict(domain=int(domain),label=label,n=int(mask.sum()),accuracy=float((pred[mask]==y[mask]).mean()),CE=float(ce[mask].mean()),confusion={str(c):int((pred[mask]==c).sum()) for c in labels}))
 return dict(n=len(y),equal_present_domain_class_accuracy=float(np.mean([c['accuracy'] for c in cells])),equal_domain_accuracy=float(np.mean([(pred[dom==d]==y[dom==d]).mean() for d in sorted(set(dom))])),image_accuracy=float((pred==y).mean()),equal_present_domain_class_CE=float(np.mean([c['CE'] for c in cells])),cells=cells)

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();locks={};data={};token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';proof=json.loads((token/'extraction_audit.json').read_text())
 for split in ['source_in','source_out']:
  f=token/f'{split}_tokens.npz';assert digest(f)==next(r['cache_sha256'] for r in proof['rows'] if r['split']==split);locks[str(f)]=digest(f);b=dict(np.load(f));b['X']={'CLS':b['cls'].astype(float),'patch_max':b['patch_max'].astype(float),'CLS_plus_patch_max':np.concatenate([b['cls'],b['patch_max']],1).astype(float)};data[split]=b
 assert not set(data['source_in']['ids'])&set(data['source_out']['ids'])
 plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,diagnostic_probe_fitting=True,L2=.01,optimizer='Existing source-only label helper; LBFGS500 stationarity<1e-5, no validation stopping or regularization selection.',labels=LABELS,class_names={1:'bobcat',6:'opossum',8:'raccoon'},tasks=[LABELS]+[list(x) for x in itertools.combinations(LABELS,2)],inputs=list(data['source_in']['X']),calibration_holds=[None,0,2,3],balance='Equal present source-location/class cells; minimum2 fitting images percell verified. Standardize using fitting rows only.',interpretation='Conditional classifiers evaluated only on true specified labels are nondeployable cohort diagnostics, not full10-class accuracy or a new factor model. Holds exclude probe fitting only; backbone/banks saw allsources. Location domains are not asserted to be verified camera IDs.',neighbors='Top3 same-label and top3 each competing focused label within source location; standardized Euclidean using source-in whole-location moments, normalized by sqrt dimensions. Also nearest across all10 labels. No neighbor count/metric selected from outcomes.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[];cases=[];neighbors=[]
 for labels in plan['tasks']:
  for kind in plan['inputs']:
   for held in plan['calibration_holds']:
    a=data['source_in'];keep=np.isin(a['y'],labels)&(True if held is None else a['domains']!=held);x=a['X'][kind][keep];y=a['y'][keep];domains=a['domains'][keep];groups=domains*10+y
    assert min(np.sum(groups==g) for g in set(groups))>=2
    target=np.eye(len(labels))[np.array([labels.index(int(i)) for i in y])];fit=fit_label(x,target,groups);file=OUT/f"{'_'.join(map(str,labels))}_{kind}_held{held}.npz";np.savez_compressed(file,**fit);rec=dict(labels=labels,input=kind,held_calibration_domain=held,fit_file=file.name,fit_sha256=digest(file),fit_rows=len(x),gradient_max=fit['gradient_max'],fit_class_domain_counts={str(g):int(sum(groups==g)) for g in sorted(set(groups))})
    for split,b in data.items():
     mask=np.isin(b['y'],labels)&(True if held is None else (b['domains']!=held if split=='source_in' else b['domains']==held));p=predict_label(fit,b['X'][kind][mask]);g=b['global_logits'][mask][:,labels];g=np.exp(g-g.max(1,keepdims=True));g/=g.sum(1,keepdims=True);rec[split]=metrics(p,b['y'][mask],b['domains'][mask],labels);rec[split+'_restricted_global']=metrics(g,b['y'][mask],b['domains'][mask],labels)
     for i,index in enumerate(np.flatnonzero(mask)):
      cases.append(dict(labels=labels,input=kind,held_calibration_domain=held,split=split,image_id=str(b['ids'][index]),domain=int(b['domains'][index]),label=int(b['y'][index]),predicted_label=int(labels[p[i].argmax()]),posterior=p[i].tolist(),restricted_global_prediction=int(labels[g[i].argmax()]),unrestricted_global_prediction=int(b['global_logits'][index].argmax())))
    rows.append(rec)
   print(labels,kind,'complete',flush=True)
 a=data['source_in'];v=data['source_out']
 for kind in plan['inputs']:
  for index in np.flatnonzero(np.isin(v['y'],LABELS)):
   domain=v['domains'][index];trainmask=a['domains']==domain;xx=a['X'][kind][trainmask];center=xx.mean(0);scale=xx.std(0).clip(1e-8);distance=np.sqrt(np.mean(((xx-v['X'][kind][index])/scale)**2,1));sourceidx=np.flatnonzero(trainmask)
   buckets={'all10':np.argsort(distance,kind='stable')[:3]}
   for label in LABELS:
    eligible=np.flatnonzero(a['y'][sourceidx]==label);buckets[str(label)]=eligible[np.argsort(distance[eligible],kind='stable')[:3]]
   neighbors.append(dict(input=kind,image_id=str(v['ids'][index]),domain=int(domain),label=int(v['y'][index]),neighbors={key:[dict(image_id=str(a['ids'][sourceidx[i]]),label=int(a['y'][sourceidx[i]]),distance=float(distance[i])) for i in indices] for key,indices in buckets.items()}))
 for f,h in locks.items():assert digest(Path(f))==h
 assert len(rows)==48
 write(OUT/'results.json',dict(plan=plan,rows=rows,case_file='cases.json',neighbors_file='neighbors.json',source_only=True,target_read=False,input_sha256=locks,distinct_fits=len(rows)));write(OUT/'cases.json',cases);write(OUT/'neighbors.json',neighbors)
if __name__=='__main__':run()
