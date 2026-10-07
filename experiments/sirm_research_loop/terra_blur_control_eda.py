"""Source-only case and utility audit of paired blur residual controls."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .terra_blur_residual_control import DEST
from .terra_roi_fixed_reader_views import DEST as ROI
from .terra_roi_species_probe import DEST as PROBE
from .terra_attention_eda import ROOT,OUT
from .shrinkage import write,digest

def main():
 names=json.loads((OUT/'results.json').read_text())['class_names'];states={}
 for name in ['original_only','paired_CE','paired_CE_consistency']:
  p=torch.load(DEST/f'{name}.pt',map_location='cpu',weights_only=False)['replays']['source_out'];y=p['labels'];d=p['domains'];g=p['global_logits'];f=p['logits'];u=F.cross_entropy(g,y,reduction='none')-F.cross_entropy(f,y,reduction='none');classes={};domains={};cases=[]
  for c,label in enumerate(names):
   m=y.eq(c);classes[label]=dict(n=int(m.sum()),global_correct=int(g.argmax(1)[m].eq(c).sum()),full_correct=int(f.argmax(1)[m].eq(c).sum()),CE_gain=float(u[m].mean()))
  for dom in sorted(d.unique().tolist()):
   m=d.eq(dom);domains[str(dom)]=dict(n=int(m.sum()),global_correct=int(g.argmax(1)[m].eq(y[m]).sum()),full_correct=int(f.argmax(1)[m].eq(y[m]).sum()),CE_gain=float(u[m].mean()))
  for i,id in enumerate(p['ids']):
   gp=g[i].argmax();fp=f[i].argmax()
   if gp!=fp:cases.append(dict(id=id,label=names[y[i]],global_prediction=names[gp],prediction=names[fp],event='repair' if fp==y[i] else 'harm' if gp==y[i] else 'wrong_to_wrong',CE_gain=float(u[i])))
  states[name]=dict(classes=classes,domains=domains,cases=cases)
 write(DEST/'eda.json',dict(states=states,source_only=True,target_read=False))
 # Fixed-reader failed group, then first3 new blur harms, all same held-domain0 reader.
 rr=torch.load(ROI/'0_replay.pt',map_location='cpu',weights_only=False);native=rr['native'];blur=rr['blur'];resized=rr['resize_first'];byid={id:i for i,id in enumerate(native['ids'])};fail=json.loads((ROI/'location100_failed_cases.json').read_text())['cases'];selected=[dict(id=c['id'],event='prespecified_failure') for c in sorted(fail,key=lambda c:c['id'])];seen={c['id'] for c in selected}
 for i,id in sorted(enumerate(native['ids']),key=lambda x:x[1]):
  if id not in seen and len(selected)<16 and native['logits'][i].argmax()==native['labels'][i] and blur['logits'][i].argmax()!=native['labels'][i]:selected.append(dict(id=id,event='new_blur_harm'))
 assert len(selected)==16;cache=torch.load(PROBE/'source_out_features.pt',map_location='cpu',weights_only=False);cindex={id:i for i,id in enumerate(cache['ids'])};boxes={r['image_id']:r['boxes'] for r in json.loads((OUT/'source_object_population/results.json').read_text())['records']};fig,axes=plt.subplots(4,4,figsize=(14,15))
 for j,(ax,c) in enumerate(zip(axes.flat,selected),1):
  i=byid[c['id']];y=int(native['labels'][i]);ax.imshow(Image.open(ROOT/'domainbed/data/terra_incognita'/c['id']).convert('RGB').resize((224,224),Image.Resampling.BILINEAR));rect=cache['rectangles'][cindex[c['id']]]
  if rect:
   x,z,x1,z1=rect;ax.add_patch(Rectangle((x*224,z*224),(x1-x)*224,(z1-z)*224,fill=False,edgecolor='cyan',linewidth=1.4))
  for x,z,w,h in boxes.get(c['id'],[]):ax.add_patch(Rectangle((x*224,z*224),w*224,h*224,fill=False,edgecolor='lime',linewidth=1.5))
  ax.set_title(f"{j:02d} {c['event']} | GT:{names[y]}\nFixed native:{names[native['logits'][i].argmax()]} → blur:{names[blur['logits'][i].argmax()]}\nResize-first, same head:{names[resized['logits'][i].argmax()]}",fontsize=8);ax.axis('off')
 fig.suptitle('Fixed ROI reader:13 prespecified failures +3 new blur harms',fontsize=13);fig.text(.5,.015,'Cyan: oracle crop, green: class-matched boxes. Same reader and frozen backbone for all input views.\nSource-only, dependent camera frames possible; not attention or deployable accuracy.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.045,1,.97));fig.savefig(ROI/'fixed_reader_cases_4x4.png',dpi=130);plt.close(fig);write(ROI/'grid_manifest.json',dict(cases=selected,selection='All13 prior location100 failures, then first3 new blur harms sorted imageID',source_only=True))
 print({name:v['domains'] for name,v in states.items()})
if __name__=='__main__':main()
