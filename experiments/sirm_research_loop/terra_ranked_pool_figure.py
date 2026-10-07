"""Paired top8 selection overlays for harms and repairs, source only."""
import json
from pathlib import Path
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .terra_ranked_pool_control import OUTPUT
from .terra_attention_eda import ROOT,OUT
from .shrinkage import write,digest

def main():
 eda=json.loads((OUTPUT/'eda.json').read_text());cases=eda['states']['foreground_top8_max']['cases'];harms=sorted([c for c in cases if c['event']=='harm'],key=lambda x:x['id']);repairs=sorted([c for c in cases if c['event']=='repair'],key=lambda x:x['id']);selected=harms+repairs[:16-len(harms)];assert len(selected)==16
 data=torch.load(OUTPUT/'source_out_features.pt',map_location='cpu');byid={id:ix for ix,id in enumerate(data['ids'])};pop=json.loads((OUT/'source_object_population/results.json').read_text());boxes={r['image_id']:r['boxes'] for r in pop['records']};fig,axes=plt.subplots(4,4,figsize=(14,15))
 for number,(ax,c) in enumerate(zip(axes.flat,selected),1):
  img=Image.open(ROOT/'domainbed/data/terra_incognita'/c['id']).convert('RGB').resize((224,224),Image.Resampling.BILINEAR);ax.imshow(img)
  for ix in data['top8_indices'][byid[c['id']]].tolist():
   row,col=divmod(ix,14);ax.add_patch(Rectangle((col*16,row*16),16,16,fill=False,edgecolor='cyan',linewidth=1.2))
  for x,y,w,h in boxes.get(c['id'],[]):ax.add_patch(Rectangle((x*224,y*224),w*224,h*224,fill=False,edgecolor='lime',linewidth=1.6))
  ax.set_title(f"{number:02d} {c['event']} | GT:{c['label']}\nERM:{c['global_prediction']} → ranked:{c['prediction']}",fontsize=8);ax.axis('off')
 fig.suptitle('Foreground top8-max reader: all10 harms + first6 of9 repairs',fontsize=13);fig.text(.5,.015,'Cyan: eight source-trained foreground-ranked patches. Green: available official class boxes.\nPatch selection is class-agnostic, not Grad-CAM. Missing box does not imply no object. Source-out only.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.97));fig.savefig(OUTPUT/'repair_harm_4x4.png',dpi=130);plt.close(fig);write(OUTPUT/'grid_manifest.json',dict(selection='All10 harms + first6 of9 repairs sorted by imageID',cases=selected,source_only=True,code_sha256=digest(Path(__file__))))
if __name__=='__main__':main()
