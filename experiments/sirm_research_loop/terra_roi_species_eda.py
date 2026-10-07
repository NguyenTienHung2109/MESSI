"""Paired source ROI probe EDA and declared oracle limitations."""
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .terra_roi_species_probe import DEST
from .terra_attention_eda import ROOT,OUT
from .terra_object_pool_readout import score
from .shrinkage import write,digest

def main():
 names=json.loads((OUT/'results.json').read_text())['class_names'];cache=torch.load(DEST/'source_out_features.pt',map_location='cpu',weights_only=False);byid={id:i for i,id in enumerate(cache['ids'])};boxes={r['image_id']:r for r in json.loads((OUT/'source_object_population/results.json').read_text())['records']};result={};combined={v:[] for v in ['original_CLS','native_ROI_pad1_CLS','resize_first_ROI_pad1_CLS']}
 for held in [None,0,2,3]:
  rows={v:torch.load(DEST/f'{held}_{v}.pt',map_location='cpu',weights_only=False)['replays']['source_out'] for v in combined};a=rows['original_CLS'];b=rows['native_ROI_pad1_CLS'];c=rows['resize_first_ROI_pad1_CLS'];y=a['labels'];cases=[];classes={}
  for k,name in enumerate(names):
   m=y.eq(k);classes[name]=dict(n=int(m.sum()),**{v:int(r['logits'].argmax(1)[m].eq(k).sum()) for v,r in rows.items()})
  for i,id in enumerate(a['ids']):
   ap=a['logits'][i].argmax();bp=b['logits'][i].argmax();cp=c['logits'][i].argmax();g=a['global_logits'][i].argmax()
   if ap!=bp or bp!=cp:cases.append(dict(id=id,label=names[y[i]],global_prediction=names[g],original_prediction=names[ap],native_prediction=names[bp],resize_first_prediction=names[cp],event='repair_vs_original' if ap!=y[i] and bp==y[i] else 'harm_vs_original' if ap==y[i] and bp!=y[i] else 'other_change',area=boxes.get(id,{}).get('box_area'),held=held))
  result[str(held)]=dict(classes=classes,cases=cases,native_vs_original=score(a['logits'],b['logits'],y,a['domains']),native_vs_resize=score(c['logits'],b['logits'],y,a['domains']))
  if held is not None:
   for v,r in rows.items():combined[v].append(r)
 aggregate={}
 for v,rr in combined.items():
  a={k:sum([r[k] for r in rr],[]) if k=='ids' else torch.cat([r[k] for r in rr]) for k in rr[0]};aggregate[v]=dict(score=score(a['global_logits'],a['logits'],a['labels'],a['domains']),boxed_correct=int(a['logits'].argmax(1)[a['is_boxed']].eq(a['labels'][a['is_boxed']]).sum()),boxed_n=int(a['is_boxed'].sum()),empty_correct=int(a['logits'].argmax(1)[~a['is_boxed']].eq(5).sum()),empty_n=int((~a['is_boxed']).sum()))
 write(DEST/'eda.json',dict(states=result,aggregate_auxiliary_held_domain=aggregate,oracle_box_diagnostic=True,deployment_accuracy=False,source_only=True,target_read=False,code_sha256=digest(Path(__file__))))
 # Show domain0 failure explicitly, then paired native-detail wins across folds.
 harms=sorted([x for x in result['0']['cases'] if x['event']=='harm_vs_original'],key=lambda x:x['id']);repairs=sorted([x for d in ['0','2','3'] for x in result[d]['cases'] if x['event']=='repair_vs_original'],key=lambda x:x['id']);selected=(harms+repairs)[:16];assert len(selected)==16;fig,axes=plt.subplots(4,4,figsize=(14,15))
 for j,(ax,c) in enumerate(zip(axes.flat,selected),1):
  i=byid[c['id']];ax.imshow(Image.open(ROOT/'domainbed/data/terra_incognita'/c['id']).convert('RGB').resize((224,224),Image.Resampling.BILINEAR));rect=cache['rectangles'][i]
  if rect:
   x,z,x1,z1=rect;ax.add_patch(Rectangle((x*224,z*224),(x1-x)*224,(z1-z)*224,fill=False,edgecolor='cyan',linewidth=1.5))
  for x,z,w,h in boxes.get(c['id'],{}).get('boxes',[]):ax.add_patch(Rectangle((x*224,z*224),w*224,h*224,fill=False,edgecolor='lime',linewidth=1.4))
  ax.set_title(f"{j:02d} {c['event']} | GT:{c['label']}\nOrigCLS:{c['original_prediction']} → native:{c['native_prediction']}\nResize-first:{c['resize_first_prediction']} | held:{c['held']}",fontsize=8);ax.axis('off')
 fig.suptitle('Oracle ROI probe: held-domain failures and repairs',fontsize=13);fig.text(.5,.015,'Cyan: oracle padded crop; green: class-matched official boxes.\nHeld out only from auxiliary fitting. Extra label-aware boxes; not deployable accuracy or attention.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.045,1,.97));fig.savefig(DEST/'held_domain_cases_4x4.png',dpi=130);plt.close(fig);write(DEST/'grid_manifest.json',dict(cases=selected,selection='All domain0 native-versus-original harms sorted ID first, then held-domain repairs until16',source_only=True,oracle_box_diagnostic=True));print(json.dumps(aggregate,indent=2)[:2000]);print('Bird:',{d:x['classes']['bird'] for d,x in result.items()})
if __name__=='__main__':main()
