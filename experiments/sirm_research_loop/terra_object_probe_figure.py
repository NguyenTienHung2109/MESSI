"""Original ERM-error images with source-trained objectness score overlays."""
import json
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch.nn.functional as F
from .terra_attention_eda import ROOT,OUT,box_mask,localization

def run():
 r=json.loads((OUT/'source_object_population/results.json').read_text());names=json.loads((OUT/'results.json').read_text())['class_names'];selected=json.loads((OUT/'source_object_population/failed_grid_ids.json').read_text())['ids'];byid={x['image_id']:x for x in r['records']};probe=dict(np.load(OUT/'source_object_probe/pooled_source_out_predictions.npz'));heatmap={str(k):v.reshape(14,14) for k,v in zip(probe['ids'],probe['probabilities'])};fig,axes=plt.subplots(4,4,figsize=(14,15))
 for i,(ax,id) in enumerate(zip(axes.flat,selected),1):
  x=byid[id];v=x['models']['strong_ERM']['views']['original'];img=Image.open(ROOT/'domainbed/data/terra_incognita'/id).convert('RGB').resize((224,224),Image.Resampling.BILINEAR);ax.imshow(img);h=heatmap[id];normalized=h/h.sum();expanded=F.interpolate(torch.tensor(normalized)[None,None],size=(224,224),mode='bilinear',align_corners=False)[0,0].numpy();ax.imshow(expanded,cmap='jet',alpha=.45,vmin=0,vmax=expanded.max());loc=localization(normalized,box_mask(x['boxes']))
  for a,b,w,height in x['boxes']:ax.add_patch(Rectangle((a*224,b*224),w*224,height*224,fill=False,edgecolor='lime',linewidth=1.5))
  ax.set_title(f"{i:02d} GT:{x['class_name']} | ERM:{names[v['prediction']]}\nobject-score enrichment={loc['area_normalized_enrichment']:.2f}",fontsize=8);ax.axis('off')
 fig.suptitle('Source-trained objectness probe on frozen patch features | 16 ERM errors',fontsize=13);fig.text(.5,.015,'Linear foreground-score probe, not Grad-CAM or transformer attention. Each overlay normalized independently.\nCaptions are original ERM predictions; this probe has not repaired their class labels. Green: official boxes.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.97));fig.savefig(OUT/'source_failed_objectness_4x4.png',dpi=130);plt.close(fig)
if __name__=='__main__':run()
