"""Immutable source-only E1 object/background interventions at selected250."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import gaussian_blur
from .conditioned_router_source_analysis import load,DEST as BANK
from .terra_attention_eda import OUT,box_mask
from .terra_object_mask_population import bg_masks
from experiments.sirm_h1_h6.run import ViewTransform
from .shrinkage import ROOT,write,digest
DEST=BANK/'source_specialist_masks'
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;DEST.mkdir(exist_ok=True);(DEST/'cases').mkdir(exist_ok=True)
 roi=OUT/'native_source_roi/results.json';records=[r for r in json.loads(roi.read_text())['records'] if r['image_id'].startswith('location_100/')];official=OUT/'caltech_bboxes_20200316.json';a=json.loads(official.read_text());cats={r['id']:r['name'] for r in a['categories']};sizes={r['id']:(r['width'],r['height']) for r in a['images']};boxes={}
 for b in a['annotations']:
  if b.get('bbox'):boxes.setdefault(b['image_id'],[]).append((cats[b['category_id']],b['bbox']))
 m,p,path=load(BANK,'selected');m=m.cuda();cache=torch.load(Path(m.config['center_feature_cache']),weights_only=False,map_location='cpu');m.calibrate_features(cache['features'],cache['domains']);m.eval();assert all(not q.requires_grad for q in m.featurizer.parameters()) and all(not q.requires_grad for q in m.global_head.parameters());before={k:v.cpu().clone() for k,v in m.state_dict().items()};ref=np.load(BANK/'source_analysis/OFF_selected_source_out_replay.npz');lookup={str(v):i for i,v in enumerate(ref['ids'])}
 plan=dict(source_only=True,target_read=False,training=False,selection=False,checkpoint_sha256=digest(path),checkpoint_step=250,cohort='All79 official class-matched boxed location100 source-out images; missingboxes/empty excluded, not interpreted as object absence',selected_ids=[r['image_id'] for r in records],replacement='Exact normalized tensor meanfill0 and Gaussian blur31/sigma7. Three translated equal-area same-shape nonoverlapping background masks when feasible. Positions fixed geometrically, not by model.',specialist='E1; squirrel-logit minus GT for nonsquirrel; squirrel minus original global runner-up for true squirrel. Report raw/frozen-route/recomputed-route effects separately.',roi_sha256=digest(roi),annotations_sha256=digest(official),code_sha256=digest(Path(__file__)))
 if (DEST/'plan.json').exists():assert json.loads((DEST/'plan.json').read_text())==plan
 else:write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 transform=ViewTransform(m.config,'original',False);results=[]
 for j,r in enumerate(records):
  fp=DEST/'cases'/f'{j:03d}.json'
  if fp.exists():results.append(json.loads(fp.read_text()));continue
  ident=Path(r['image_id']).stem;ow,oh=sizes[ident];bb=[[x/ow,y/oh,w/ow,h/oh] for name,(x,y,w,h) in boxes[ident] if name==r['class_name']];mask=box_mask(bb);bg=bg_masks(mask);assert all(v.sum()==mask.sum() and not (v&mask).any() for v in bg)
  with Image.open(ROOT/'domainbed/data/terra_incognita'/r['image_id']) as im:x=transform(im.convert('RGB'))[None].cuda()
  blur=gaussian_blur(x,[31,31],[7.,7.]);views={'original':x}
  for region,ma in {'object':mask,**{f'background{k}':v for k,v in enumerate(bg)}}.items():
   t=torch.tensor(ma,device=x.device)[None,None];views[region+'_mean']=torch.where(t,torch.zeros_like(x),x);views[region+'_blur']=torch.where(t,blur,x)
  with torch.no_grad():out=m(torch.cat(list(views.values())))
  f=out['logits'].cpu();g=out['global_logits'].cpu();d=out['delta'].cpu();pi=out['pi'].cpu();i=lookup[r['image_id']];err=float((f[0]-torch.tensor(ref['full_logits'][i])).abs().max());assert err<5e-5,(r['image_id'],err)
  y=int(r['label']);other=g[0].clone();other[9]=-torch.inf;contrast=y if y!=9 else int(other.argmax());vals={}
  for k,key in enumerate(views):
   gg=g[k].clone();ff=f[k].clone();gg[y]=-torch.inf;ff[y]=-torch.inf;raw=float(d[k,1,9]-d[k,1,contrast]);vals[key]=dict(global_prediction=int(g[k].argmax()),full_prediction=int(f[k].argmax()),global_GT_margin=float(g[k,y]-gg.max()),full_GT_margin=float(f[k,y]-ff.max()),E1_mass=float(pi[k,1]),E1_raw_squirrel_contrast=raw,E1_fixed_original_route_contrast=float(pi[0,1])*raw,E1_recomputed_route_contrast=float(pi[k,1])*raw)
  row=dict(image_id=r['image_id'],label=y,class_name=r['class_name'],area=float(mask.mean()),background_controls=len(bg),contrast_label=contrast,original_replay_error=err,views=vals);write(fp,row);results.append(row);write(DEST/'progress.json',dict(status='running',completed=len(results),total=len(records)));print(j+1,flush=True)
 assert all(torch.equal(v,m.state_dict()[k].cpu()) for k,v in before.items());assert all(q.grad is None for q in m.parameters());assert digest(path)==plan['checkpoint_sha256']
 summary={}
 for group in ['all','true_squirrel','nonsquirrel','global_wrong','squirrel_confusion']:
  rr=[r for r in results if group=='all' or group=='true_squirrel' and r['label']==9 or group=='nonsquirrel' and r['label']!=9 or group=='global_wrong' and r['views']['original']['global_prediction']!=r['label'] or group=='squirrel_confusion' and r['label']!=9 and r['views']['original']['global_prediction']==9];summ={}
  for kind in ['mean','blur']:
   paired=[r for r in rr if r['background_controls']];metrics={}
   for metric in ['global_GT_margin','full_GT_margin','E1_raw_squirrel_contrast','E1_fixed_original_route_contrast','E1_recomputed_route_contrast','E1_mass']:
    od=[r['views']['original'][metric]-r['views']['object_'+kind][metric] for r in paired];bd=[r['views']['original'][metric]-np.mean([r['views'][f'background{k}_{kind}'][metric] for k in range(r['background_controls'])]) for r in paired]
    metrics[metric]=dict(median_object_drop=float(np.median(od)) if od else None,median_background_drop=float(np.median(bd)) if bd else None,object_drop_larger_fraction=float(np.mean(np.array(od)>bd)) if od else None)
   summ[kind]=dict(n=len(rr),paired_n=len(paired),metrics=metrics)
  summary[group]=summ
 write(DEST/'results.json',dict(plan=plan,summary=summary,records=results,parameters_unchanged=True,target_read=False));write(DEST/'progress.json',dict(status='complete',completed=len(results),total=len(records)));print('Complete source specialist masks',flush=True)
if __name__=='__main__':main()
