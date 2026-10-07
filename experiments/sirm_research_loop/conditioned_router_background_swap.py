"""Source-only object-retained background donor controls, no fitting."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import gaussian_blur
from .conditioned_router_source_analysis import load,DEST as BANK
from .terra_attention_eda import OUT,box_mask
from experiments.sirm_h1_h6.run import ViewTransform
from .shrinkage import ROOT,write,digest
DEST=BANK/'source_background_swap'
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;DEST.mkdir(exist_ok=True);(DEST/'cases').mkdir(exist_ok=True)
 manifest=json.loads((BANK/'experiment/subset_manifest.json').read_text());records=json.loads((BANK/'source_specialist_masks/results.json').read_text())['records'];ann=OUT/'caltech_bboxes_20200316.json';a=json.loads(ann.read_text());cats={r['id']:r['name'] for r in a['categories']};sizes={r['id']:(r['width'],r['height']) for r in a['images']};boxes={}
 for b in a['annotations']:
  if b.get('bbox'):boxes.setdefault(b['image_id'],[]).append((cats[b['category_id']],b['bbox']))
 donorids={}
 for domain in [0,2,3]:
  s=manifest['sources'][str(domain)];donorids[str(domain)]=sorted(s['images'][i]['id'] for i in s['in'] if s['images'][i]['class']==5)[0]
 m,p,path=load(BANK,'selected');m=m.cuda();cache=torch.load(Path(m.config['center_feature_cache']),weights_only=False,map_location='cpu');m.calibrate_features(cache['features'],cache['domains']);m.eval();before={k:v.cpu().clone() for k,v in m.state_dict().items()};transform=ViewTransform(m.config,'original',False);donors={}
 for domain,ident in donorids.items():
  with Image.open(ROOT/'domainbed/data/terra_incognita'/ident) as im:donors[domain]=transform(im.convert('RGB'))[None].cuda()
 plan=dict(source_only=True,target_read=False,training=False,checkpoint_changes=False,checkpoint_sha256=digest(path),annotations_sha256=digest(ann),donors={d:dict(id=i,sha256=digest(ROOT/'domainbed/data/terra_incognita'/i)) for d,i in donorids.items()},donor_selection='Lexicographically first labeled-empty original source-in image per domain0/2/3; never chosen by prediction or source-out outcome',cohort='All79 previously locked boxed location100 source-out images',ids=[r['image_id'] for r in records],interventions='Preserve all normalized bbox pixels, position and scale exactly; replace outside with source-in empty donor. Hard rectangular paste and feather outside bbox only Gaussian15/sigma3. Exact sham self-donor control. Donor-only outputs saved separately.',limitations='Synthetic boundary/illumination/context shift; same object content, not natural counterfactual. Does not identify natural causal shortcut or provide deployable accuracy.',code_sha256=digest(Path(__file__)))
 if (DEST/'plan.json').exists():assert json.loads((DEST/'plan.json').read_text())==plan
 else:write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 ref=np.load(BANK/'source_analysis/OFF_selected_source_out_replay.npz');lookup={str(v):i for i,v in enumerate(ref['ids'])};results=[]
 for j,r in enumerate(records):
  fp=DEST/'cases'/f'{j:03d}.json'
  if fp.exists():results.append(json.loads(fp.read_text()));continue
  ident=Path(r['image_id']).stem;ow,oh=sizes[ident];bb=[[x/ow,y/oh,w/ow,h/oh] for name,(x,y,w,h) in boxes[ident] if name==r['class_name']];mask=torch.tensor(box_mask(bb),device='cuda')[None,None];alpha=gaussian_blur(mask.float(),[15,15],[3.,3.]);alpha=torch.where(mask,torch.ones_like(alpha),alpha)
  with Image.open(ROOT/'domainbed/data/terra_incognita'/r['image_id']) as im:x=transform(im.convert('RGB'))[None].cuda()
  views={'original':x,'sham':torch.where(mask,x,x)};assert torch.equal(views['sham'],x)
  for d,bg in donors.items():
   views[f'hard_{d}']=torch.where(mask,x,bg);views[f'feather_{d}']=alpha*x+(1-alpha)*bg
   for k in [f'hard_{d}',f'feather_{d}']:assert torch.equal(views[k].masked_select(mask.expand_as(x)),x.masked_select(mask.expand_as(x)))
   views[f'donor_{d}']=bg
  with torch.no_grad():out=m(torch.cat(list(views.values())))
  f=out['logits'].cpu();g=out['global_logits'].cpu();delta=out['delta'].cpu();pi=out['pi'].cpu();i=lookup[r['image_id']];err=float((f[0]-torch.tensor(ref['full_logits'][i])).abs().max());assert err<5e-5;assert torch.equal(f[0],f[1])
  y=int(r['label']);other=g[0].clone();other[9]=-torch.inf;contrast=y if y!=9 else int(other.argmax());values={}
  for k,key in enumerate(views):
   gg=g[k].clone();ff=f[k].clone();gg[y]=-torch.inf;ff[y]=-torch.inf;raw=float(delta[k,1,9]-delta[k,1,contrast]);values[key]=dict(global_prediction=int(g[k].argmax()),full_prediction=int(f[k].argmax()),global_GT_margin=float(g[k,y]-gg.max()),full_GT_margin=float(f[k,y]-ff.max()),E1_mass=float(pi[k,1]),E1_raw_squirrel_contrast=raw,E1_routed_squirrel_contrast=float(pi[k,1])*raw)
  row=dict(image_id=r['image_id'],label=y,class_name=r['class_name'],area=r['area'],contrast_label=contrast,original_replay_error=err,object_pixels_exact=True,sham_logits_exact=True,views=values);write(fp,row);results.append(row);write(DEST/'progress.json',dict(status='running',completed=len(results),total=len(records)));print(j+1,flush=True)
 assert all(torch.equal(v,m.state_dict()[k].cpu()) for k,v in before.items()) and all(q.grad is None for q in m.parameters());assert digest(path)==plan['checkpoint_sha256'];summary={}
 for group in ['all','true_squirrel','squirrel_confusion']:
  rr=[r for r in results if group=='all' or group=='true_squirrel' and r['label']==9 or group=='squirrel_confusion' and r['label']!=9 and r['views']['original']['global_prediction']==9];ss={}
  for style in ['hard','feather']:
   ss[style]=dict(n=len(rr),global_prediction_change_fraction=float(np.mean([r['views'][f'{style}_{d}']['global_prediction']!=r['views']['original']['global_prediction'] for r in rr for d in donorids])) if rr else None,full_prediction_change_fraction=float(np.mean([r['views'][f'{style}_{d}']['full_prediction']!=r['views']['original']['full_prediction'] for r in rr for d in donorids])) if rr else None,E1_route_range_median=float(np.median([np.ptp([r['views'][f'{style}_{d}']['E1_mass'] for d in donorids]) for r in rr])) if rr else None,E1_raw_contrast_range_median=float(np.median([np.ptp([r['views'][f'{style}_{d}']['E1_raw_squirrel_contrast'] for d in donorids]) for r in rr])) if rr else None,global_repairs=sum(r['views']['original']['global_prediction']!=r['label'] and r['views'][f'{style}_{d}']['global_prediction']==r['label'] for r in rr for d in donorids),global_harms=sum(r['views']['original']['global_prediction']==r['label'] and r['views'][f'{style}_{d}']['global_prediction']!=r['label'] for r in rr for d in donorids))
  summary[group]=ss
 write(DEST/'results.json',dict(plan=plan,records=results,summary=summary,parameters_unchanged=True,object_pixels_exact=True,sham_logits_exact=True,target_read=False));write(DEST/'progress.json',dict(status='complete',completed=len(results),total=len(records)));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
