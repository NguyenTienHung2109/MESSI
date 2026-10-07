"""Source donor figure from locked swaps; visual clipping is not inference clipping."""
import json
from pathlib import Path
import torch,numpy as np
from PIL import Image
from torchvision.transforms.functional import gaussian_blur
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .shrinkage import ROOT,write
from .terra_attention_eda import OUT,box_mask
from .conditioned_router_background_swap_photometric import DEST,BANK
from experiments.sirm_h1_h6.run import ViewTransform
def main():
 a=json.loads((DEST/'results.json').read_text());meta=json.loads((OUT/'caltech_bboxes_20200316.json').read_text());cats={x['id']:x['name'] for x in meta['categories']};sizes={x['id']:(x['width'],x['height']) for x in meta['images']};boxes={}
 for b in meta['annotations']:
  if b.get('bbox'):boxes.setdefault(b['image_id'],[]).append((cats[b['category_id']],b['bbox']))
 failed=[x for x in a['records'] if x['label']!=9 and x['views']['original']['global_prediction']==9];controls=sorted([x for x in a['records'] if x['label']==9],key=lambda x:x['image_id'])[:2];chosen=failed+controls;assert len(chosen)==4
 config=torch.load(BANK/'experiment/runs/L38/OFF/source_best.pkl',weights_only=False,map_location='cpu')['config'];tr=ViewTransform(config,'original',False);names=json.loads((BANK/'experiment/subset_manifest.json').read_text())['class_names'];donors={}
 for d,z in a['plan']['donors'].items():
  with Image.open(ROOT/'domainbed/data/terra_incognita'/z['id']) as im:donors[d]=tr(im.convert('RGB'))[None]
 fig,axes=plt.subplots(4,4,figsize=(16,16));shown=[]
 for ri,r in enumerate(chosen):
  ident=Path(r['image_id']).stem;ow,oh=sizes[ident];bb=[[x/ow,y/oh,w/ow,h/oh] for name,(x,y,w,h) in boxes[ident] if name==r['class_name']];mask=torch.tensor(box_mask(bb))[None,None];alpha=torch.where(mask,torch.ones_like(mask).float(),gaussian_blur(mask.float(),[15,15],[3.,3.]))
  with Image.open(ROOT/'domainbed/data/terra_incognita'/r['image_id']) as im:x=tr(im.convert('RGB'))[None]
  views={'original':x};outside=(~mask).expand_as(x);tm=x.masked_select(outside).view(3,-1).mean(1)[None,:,None,None];ts=x.masked_select(outside).view(3,-1).std(1,unbiased=False).clamp_min(1e-4)[None,:,None,None]
  for d,bg in donors.items():
   dm=bg.masked_select(outside).view(3,-1).mean(1)[None,:,None,None];ds=bg.masked_select(outside).view(3,-1).std(1,unbiased=False).clamp_min(1e-4)[None,:,None,None];nb=(bg-dm)/ds*ts+tm;views['feather_'+d]=alpha*x+(1-alpha)*nb
  for ci,(key,z) in enumerate(views.items()):
   rgb=z[0]*torch.tensor([.229,.224,.225])[:,None,None]+torch.tensor([.485,.456,.406])[:,None,None];display=rgb.permute(1,2,0).numpy();clipped=float(((display<0)|(display>1)).mean());ax=axes[ri,ci];ax.imshow(display.clip(0,1));ax.axis('off');v=r['views'][key];ax.set_title(f"GT {r['class_name']} {ident[:8]} | {key}\nERM/full {names[v['global_prediction']]}/{names[v['full_prediction']]}\nE1 {v['E1_mass']:.3f}, contrast {v['E1_routed_squirrel_contrast']:+.3f}",fontsize=9);shown.append(dict(id=r['image_id'],view=key,display_clipped_channel_fraction=clipped))
 fig.suptitle('Object pixels exact; source-in empty donor backgrounds, channel mean/std matched; display clipped, inference not clipped');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(DEST/'paired_contexts_4x4.png',dpi=120);plt.close(fig);write(DEST/'figure_manifest.json',shown)
if __name__=='__main__':main()
