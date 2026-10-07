"""4x4 paired source mask display; exact metrics saved by mask runner."""
import json
import numpy as np
from PIL import Image,ImageFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .terra_attention_eda import box_mask,OUT
from .terra_object_mask_population import bg_masks
from .conditioned_router_object_mask import DEST
from .shrinkage import ROOT,write
from pathlib import Path
def main():
 a=json.loads((DEST/'results.json').read_text());meta=json.loads((OUT/'caltech_bboxes_20200316.json').read_text());cats={x['id']:x['name'] for x in meta['categories']};sizes={x['id']:(x['width'],x['height']) for x in meta['images']};boxes={}
 for b in meta['annotations']:
  if b.get('bbox'):boxes.setdefault(b['image_id'],[]).append((cats[b['category_id']],b['bbox']))
 failed=[x for x in a['records'] if x['label']!=9 and x['views']['original']['global_prediction']==9];controls=sorted([x for x in a['records'] if x['label']==9],key=lambda x:x['image_id'])[:2];chosen=failed+controls;assert len(chosen)==4
 fig,axes=plt.subplots(4,4,figsize=(16,16));keys=['original','object_mean','background0_mean','object_blur']
 for ri,x in enumerate(chosen):
  ident=Path(x['image_id']).stem;ow,oh=sizes[ident];bb=[[xx/ow,yy/oh,w/ow,h/oh] for name,(xx,yy,w,h) in boxes[ident] if name==x['class_name']];mask=box_mask(bb);bg=bg_masks(mask)
  with Image.open(ROOT/'domainbed/data/terra_incognita'/x['image_id']) as im:img=np.array(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR))/255
  for ci,key in enumerate(keys):
   ax=axes[ri,ci];im=img.copy()
   if key.endswith('_mean'):im[mask if key=='object_mean' else bg[0]]=np.array([.485,.456,.406])
   if key=='object_blur':
    blur=np.array(Image.fromarray((img*255).astype('uint8')).filter(ImageFilter.GaussianBlur(7)))/255;im[mask]=blur[mask]
   ax.imshow(im);ax.axis('off');v=x['views'][key];ax.set_title(f"{x['class_name']} {ident[:8]} | {key}\nE1 mass {v['E1_mass']:.3f}, contrast {v['E1_recomputed_route_contrast']:+.3f}",fontsize=10)
 fig.suptitle('2 boxed squirrel confusions +2 true-squirrel controls; display blur approximate, metrics use exact normalized tensors');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(DEST/'paired_interventions_4x4.png',dpi=120);plt.close(fig);write(DEST/'paired_interventions_manifest.json',[x['image_id'] for x in chosen])
if __name__=='__main__':main()
