"""Paired source utility and cases for matched fusion controls."""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .terra_local_fusion_control import DEST,SOURCE
from .terra_attention_eda import ROOT,OUT
from .shrinkage import write,digest

def main():
 torch.set_num_threads(4);names=['bird','bobcat','cat','coyote','dog','empty','opossum','rabbit','raccoon','squirrel'];results={};states={}
 for name in ['global_local','global_repeat','global_permuted_local']:
  p=torch.load(DEST/f'{name}.pt',map_location='cpu',weights_only=False)['replays']['source_out'];states[name]=p;y=p['labels'];d=p['domains'];g=p['global_logits'];f=p['logits'];u=F.cross_entropy(g,y,reduction='none')-F.cross_entropy(f,y,reduction='none');classes={};domains={};cases=[]
  for c,label in enumerate(names):
   m=y.eq(c);classes[label]=dict(n=int(m.sum()),global_correct=int(g.argmax(1)[m].eq(c).sum()),full_correct=int(f.argmax(1)[m].eq(c).sum()),mean_CE_gain=float(u[m].mean()))
  for dom in sorted(d.unique().tolist()):
   m=d.eq(dom);domains[str(dom)]=dict(n=int(m.sum()),global_correct=int(g.argmax(1)[m].eq(y[m]).sum()),full_correct=int(f.argmax(1)[m].eq(y[m]).sum()),mean_CE_gain=float(u[m].mean()))
  for i,id in enumerate(p['ids']):
   gp=g[i].argmax();fp=f[i].argmax()
   if gp!=fp:cases.append(dict(index=i,id=id,label=names[y[i]],global_prediction=names[gp],prediction=names[fp],event='repair' if fp==y[i] else 'harm' if gp==y[i] else 'wrong_to_wrong',CE_gain=float(u[i])))
  # Descriptive paired image bootstrap within each source domain. Source-domain overlap
  # and camera dependence remain; not an independence-adjusted confidence interval.
  gen=np.random.default_rng(0);ds=sorted(d.unique().tolist());boot=[]
  for rep in range(5000):boot.append(np.mean([u[d.eq(dom)].numpy()[gen.integers(0,int(d.eq(dom).sum()),int(d.eq(dom).sum()))].mean() for dom in ds]))
  results[name]=dict(classes=classes,domains=domains,cases=cases,descriptive_image_bootstrap_CE_gain_95percentile=np.quantile(boot,[.025,.975]).tolist(),bootstrap_scope='Paired resampling within domains, descriptive only; camera dependence and repeated source overlap not adjusted.')
 p=states['global_local'];base=states['global_repeat'];perm=states['global_permuted_local'];y=p['labels'];d=p['domains'];results['pair_controls']=dict(local_vs_global_repeat_unique_repairs=int((p['logits'].argmax(1).eq(y)&base['logits'].argmax(1).ne(y)).sum()),global_repeat_vs_local_unique_repairs=int((base['logits'].argmax(1).eq(y)&p['logits'].argmax(1).ne(y)).sum()),local_vs_permuted_unique_repairs=int((p['logits'].argmax(1).eq(y)&perm['logits'].argmax(1).ne(y)).sum()),permuted_vs_local_unique_repairs=int((perm['logits'].argmax(1).eq(y)&p['logits'].argmax(1).ne(y)).sum()))
 write(DEST/'eda.json',dict(states=results,source_only=True,target_read=False,code_sha256=digest(Path(__file__))))
 # All changed local-fusion cases, followed by its largest-CE remaining errors.
 selected=sorted(results['global_local']['cases'],key=lambda c:c['id']);seen={c['index'] for c in selected};loss=F.cross_entropy(p['logits'],y,reduction='none')
 for i in sorted(torch.nonzero(p['logits'].argmax(1).ne(y)).flatten().tolist(),key=lambda i:-float(loss[i])):
  if i not in seen and len(selected)<16:selected.append(dict(index=i,id=p['ids'][i],label=names[y[i]],global_prediction=names[p['global_logits'][i].argmax()],prediction=names[p['logits'][i].argmax()],event='remaining_error'))
 selected=selected[:16];assert len(selected)==16;probe=torch.load(SOURCE/'all_image_replay/pooled_replay.pt',map_location='cpu',weights_only=False)['source_out'];boxes={r['image_id']:r['boxes'] for r in json.loads((OUT/'source_object_population/results.json').read_text())['records']};fig,axes=plt.subplots(4,4,figsize=(14,15))
 for j,(ax,c) in enumerate(zip(axes.flat,selected),1):
  i=c['index'];ax.imshow(Image.open(ROOT/'domainbed/data/terra_incognita'/c['id']).convert('RGB').resize((224,224),Image.Resampling.BILINEAR))
  for ix in probe['top8_indices'][i].tolist():
   r,col=divmod(ix,14);ax.add_patch(Rectangle((col*16,r*16),16,16,fill=False,edgecolor='cyan',linewidth=1))
  for x,z,w,h in boxes.get(c['id'],[]):ax.add_patch(Rectangle((x*224,z*224),w*224,h*224,fill=False,edgecolor='lime',linewidth=1.5))
  ax.set_title(f"{j:02d} {c['event']} | GT:{c['label']}\nERM:{c['global_prediction']} → fusion:{c['prediction']}\nGlobal-control:{names[base['logits'][i].argmax()]}",fontsize=8);ax.axis('off')
 fig.suptitle('Source-only learned fusion: all changed predictions + remaining errors',fontsize=13);fig.text(.5,.015,'Cyan: hierarchical local head top8 patches; green: available class boxes.\nSelection overlay, not Grad-CAM. No target data or SIRM penalty.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.97));fig.savefig(DEST/'cases_4x4.png',dpi=130);plt.close(fig);write(DEST/'grid_manifest.json',dict(cases=selected,source_only=True,selection='All local-fusion changes sorted by ID, then highest full-CE remaining errors until16'))
 print(json.dumps({k:v if k=='pair_controls' else dict(classes=v['classes'],domains=v['domains'],cases=v['cases'],CE_bootstrap=v['descriptive_image_bootstrap_CE_gain_95percentile']) for k,v in results.items()},indent=2))
if __name__=='__main__':main()
