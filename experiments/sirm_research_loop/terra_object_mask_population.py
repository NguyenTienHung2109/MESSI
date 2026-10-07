"""Population source-out localization + matched-shape masking, fixed checkpoints."""
import json,copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from PIL import Image
from .terra_attention_eda import ROOT,OUT,load_models,explain,box_mask,localization,draw_grid
from experiments.sirm_h1_h6.run import ViewTransform
from .shrinkage import digest,write

DEST=OUT/'source_object_population'

def bg_masks(mask):
 ys,xs=np.nonzero(mask);bh=int(ys.max()-ys.min()+1);bw=int(xs.max()-xs.min()+1);shape=mask[ys.min():ys.max()+1,xs.min():xs.max()+1];candidates=[]
 for yy in range(0,224-bh+1,8):
  for xx in range(0,224-bw+1,8):
   z=np.zeros_like(mask);z[yy:yy+bh,xx:xx+bw]=shape
   if not (z&mask).any():candidates.append((yy,xx,z))
 if not candidates:return []
 # Prespecified farthest-first centers; no model scores choose positions.
 chosen=[candidates[0]]
 while len(chosen)<min(3,len(candidates)):
  remaining=[z for z in candidates if not any(z[:2]==q[:2] for q in chosen)]
  chosen.append(max(remaining,key=lambda z:min((z[0]-q[0])**2+(z[1]-q[1])**2 for q in chosen)))
 return [z[2] for z in chosen]

def measure(logits,label,original_pred):
 scores=logits.detach().cpu();others=scores.clone();others[label]=-torch.inf
 return dict(prediction=int(scores.argmax()),correct=bool(scores.argmax()==label),GT_logit=float(scores[label]),GT_margin=float(scores[label]-others.max()),original_pred_logit=float(scores[original_pred]),logits=scores.tolist())

def run():
 torch.set_num_threads(4);DEST.mkdir(exist_ok=True);(DEST/'cases').mkdir(exist_ok=True)
 roi_path=OUT/'native_source_roi/results.json';source=json.loads(roi_path.read_text());records=source['records'];names=json.loads((OUT/'results.json').read_text())['class_names']
 official=json.loads((OUT/'caltech_bboxes_20200316.json').read_text());cats={a['id']:a['name'] for a in official['categories']};sizes={a['id']:(a['width'],a['height']) for a in official['images']};boxes={}
 for a in official['annotations']:
  if a.get('bbox'):boxes.setdefault(a['image_id'],[]).append((cats[a['category_id']],a['bbox']))
 plan=dict(scope='All206 source-out class-matched boxed images, fixed original view; target excluded',ROI_input_sha256=digest(roi_path),
  interventions='Exact normalized input tensor. Object mean-fill(0), three translated same-shape equal-area nonoverlapping background fills when feasible; repeat with image Gaussian blur kernel31 sigma7 as replacement.',
  controls='Background positions determined geometrically by fixed raster/farthest-first, never by attribution/logit. Missing feasible backgrounds remain missing.',
  maps='True-class/predicted-class positive Grad-CAM and raw attention rollout. Current strong ERM and historical lambda1 SIRM only.',
  inference_models=['strong_ERM','current_SIRM_inactive','historical_ERM_global','historical_SIRM_IRMv1_ON'],
  metric='GT-versus-max-other margin, true logit and original predicted class logit; repair/harm; object impact versus mean background impact separately by correctness/class/area.',
  training=False,target_read=False,selection=False,code_sha256=digest(Path(__file__)))
 if (DEST/'plan.json').exists():assert json.loads((DEST/'plan.json').read_text())==plan
 else:write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 current,historical,forwards,locks=load_models();versions={name:[q._version for q in m.parameters()] for name,m in [('current',current),('historical',historical)]};transform=ViewTransform(current.config,'original',False)
 all_records=[]
 for i,r in enumerate(records):
  casefile=DEST/'cases'/f'{i:03d}.json'
  if casefile.exists():all_records.append(json.loads(casefile.read_text()));continue
  path=ROOT/'domainbed/data/terra_incognita'/r['image_id'];img=Image.open(path).convert('RGB');x=transform(img)[None].cuda();id=Path(r['image_id']).stem;ow,oh=sizes[id];bb=[[a/ow,b/oh,w/ow,h/oh] for label,(a,b,w,h) in boxes[id] if label==r['class_name']];mask=box_mask(bb);bg=bg_masks(mask);label=r['label'];masks={'object':mask,**{f'background{k}':v for k,v in enumerate(bg)}}
  assert all(v.sum()==mask.sum() and not (v&mask).any() for v in bg)
  blur=gaussian_blur(x,[31,31],[7.,7.]);views={'original':x}
  for region,m in masks.items():
   m=torch.tensor(m,device=x.device)[None,None]
   views[f'{region}_mean']=torch.where(m,torch.zeros_like(x),x)
   views[f'{region}_blur']=torch.where(m,blur,x)
  batch=torch.cat(list(views.values()));result=dict(image_id=r['image_id'],split='source_out',class_name=r['class_name'],label=label,boxes=bb,box_area=float(mask.mean()),background_controls=len(bg),image_sha256=digest(path),models={})
  for name,forward in forwards.items():
   with torch.no_grad():logits=forward(batch)
   p=int(logits[0].argmax());values=dict(views={key:measure(z,label,p) for key,z in zip(views,logits)},localization={})
   if name in ['strong_ERM','historical_SIRM_IRMv1_ON']:
    model=current if name=='strong_ERM' else historical
    original,roll,gt,pred=explain(model,forward,x,label)
    assert np.allclose(original,logits[0].cpu().numpy(),rtol=1e-5,atol=1e-5)
    heat=dict(rollout=roll,GT_GradCAM=gt,predicted_GradCAM=pred);values['localization']={key:localization(v,mask) for key,v in heat.items()}
    np.savez_compressed(DEST/'cases'/f'{i:03d}_{name}.npz',**heat)
   result['models'][name]=values
  write(casefile,result);all_records.append(result);write(DEST/'progress.json',dict(status='running',complete=len(all_records),total=len(records)));print(i+1,'/',len(records),flush=True)
 for name,m in [('current',current),('historical',historical)]:assert versions[name]==[q._version for q in m.parameters()] and all(q.grad is None for q in m.parameters())
 for path,h in locks.items():assert digest(Path(path))==h
 groups={'all':all_records,'original_wrong':None,'original_correct':None}
 summary={}
 for name in forwards:
  summary[name]={}
  for group in ['all','original_wrong','original_correct']+names+['area_lt_0.05','area_ge_0.05']:
   rr=[r for r in all_records if (group=='all' or (group=='original_wrong' and not r['models'][name]['views']['original']['correct']) or (group=='original_correct' and r['models'][name]['views']['original']['correct']) or r['class_name']==group or (group=='area_lt_0.05' and r['box_area']<.05) or (group=='area_ge_0.05' and r['box_area']>=.05))]
   if not rr:continue
   g=dict(n=len(rr),original_correct=sum(r['models'][name]['views']['original']['correct'] for r in rr),matched_background_n=sum(r['background_controls']>0 for r in rr),interventions={})
   for kind in ['mean','blur']:
    paired=[];flips=0;repairs=0;harms=0;bgflips=[]
    for r in rr:
     v=r['models'][name]['views'];a=v['original'];b=v[f'object_{kind}'];flips+=a['prediction']!=b['prediction'];repairs+=not a['correct'] and b['correct'];harms+=a['correct'] and not b['correct']
     if r['background_controls']:
      controls=[v[f'background{k}_{kind}'] for k in range(r['background_controls'])];od=a['GT_margin']-b['GT_margin'];bd=np.mean([a['GT_margin']-c['GT_margin'] for c in controls]);paired.append((od,bd));bgflips.extend([a['prediction']!=c['prediction'] for c in controls])
    p=np.array(paired);g['interventions'][kind]=dict(object_prediction_flips=flips,object_repairs=repairs,object_harms=harms,background_prediction_flip_fraction=float(np.mean(bgflips)) if bgflips else None,
     matched_n=len(paired),median_object_GT_margin_drop=float(np.median(p[:,0])) if len(p) else None,median_mean_background_GT_margin_drop=float(np.median(p[:,1])) if len(p) else None,median_paired_object_minus_background_drop=float(np.median(p[:,0]-p[:,1])) if len(p) else None,object_drop_greater_fraction=float(np.mean(p[:,0]>p[:,1])) if len(p) else None)
   summary[name][group]=g
 write(DEST/'results.json',dict(summary=summary,records=all_records,checkpoints_sha256=locks,parameters_unchanged=True,target_read=False,training=False))
 write(DEST/'progress.json',dict(status='complete',complete=len(all_records),total=len(records)));print(json.dumps({name:v['all'] for name,v in summary.items()},indent=2))
if __name__=='__main__':run()
