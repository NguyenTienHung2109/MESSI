"""Compare learned axes against fixed axes, with train-only amplitude controls."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .source_low_rank_readout import Reader,CACHE,DEST
from .shrinkage import ROOT,write,digest

def main():
 torch.set_num_threads(4)
 fixed=DEST.parent/'source_low_rank_readout_fixed_axis'
 data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']}
 raw={s:torch.tensor(np.concatenate([v['cls'],v['patch_max']],1)) for s,v in data.items()}
 g=torch.tensor(data['source_out']['global_logits']);y=torch.tensor(data['source_out']['y']);dom=torch.tensor(data['source_out']['domains']);gc=g.argmax(1).eq(y)
 records={};cases=[]
 for fit in ['pooled','held0','held2','held3']:
  held=None if fit=='pooled' else int(fit[-1]);keep=np.ones(len(data['source_in']['y']),bool) if held is None else data['source_in']['domains']!=held;take=torch.ones(len(y),dtype=torch.bool) if held is None else dom.eq(held)
  deltas={};rms={};axes={}
  for rank in [1,9]:
   for policy,path in [('learned',DEST),('fixed',fixed)]:
    key=f'{policy}_rank{rank}';a=torch.load(path/f'{fit}_rank{rank}.pt',map_location='cpu',weights_only=False);assert a['steps']==400
    b=a['b']-a['b'].mean(0,keepdim=True);w=b@a['a'];z=(raw['source_in'][keep]-a['mean'])/a['std']/2**.5;v=(raw['source_out']-a['mean'])/a['std']/2**.5
    rms[key]=float(F.linear(z,w).square().mean().sqrt());deltas[key]=F.linear(v,w)
    if rank==1:axes[policy]=b[:,0]/b[:,0].norm()
  controls={}
  variants=list(deltas.items())
  for rank in [1,9]:
   for src,dst in [('learned','fixed'),('fixed','learned')]:
    key=f'{src}_rank{rank}';variants.append((f'{key}_to_{dst}_trainRMS',deltas[key]*rms[f'{dst}_rank{rank}']/rms[key]))
  for name,d in variants:
   f=g+d;fc=f.argmax(1).eq(y);ce=F.cross_entropy(f,y,reduction='none');controls[name]=dict(accuracy=float(torch.stack([fc[take&dom.eq(dd)].float().mean() for dd in dom[take].unique()]).mean()),CE=float(torch.stack([ce[take&dom.eq(dd)].mean() for dd in dom[take].unique()]).mean()),repairs=int((take&~gc&fc).sum()),harms=int((take&gc&~fc).sum()))
  records[fit]=dict(controls=controls,train_RMS=rms,rank1_axis_cosine_to_fixed=float(axes['learned']@axes['fixed']))
  if fit=='pooled':
   for i in range(len(y)):
    preds={k:int((g+d)[i].argmax()) for k,d in deltas.items()};gp=int(g[i].argmax());gt=int(y[i])
    if any(p!=gp for p in preds.values()):cases.append(dict(id=str(data['source_out']['ids'][i]),label=gt,global_prediction=gp,predictions=preds,learned_rank1_harm=gp==gt and preds['learned_rank1']!=gt,learned_rank1_repair=gp!=gt and preds['learned_rank1']==gt,learned_only_rank1_repair=gp!=gt and preds['learned_rank1']==gt and preds['fixed_rank1']!=gt))
 write(DEST/'axis_controls.json',dict(source_only=True,target_read=False,fitting=False,amplitudes_from_source_in_only=True,fits=records));write(DEST/'cases.json',cases)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image
 names=json.loads((DEST.parent/'experiment/subset_manifest.json').read_text())['class_names'];chosen=sorted(cases,key=lambda a:(not a['learned_rank1_harm'],not a['learned_only_rank1_repair'],not a['learned_rank1_repair'],a['id']))[:16]
 fig,axs=plt.subplots(4,4,figsize=(16,16))
 for ax in axs.flat:ax.axis('off')
 for ax,a in zip(axs.flat,chosen):
  with Image.open(ROOT/'domainbed/data/terra_incognita'/a['id']) as im:ax.imshow(im.convert('RGB'))
  p=a['predictions'];ax.set_title(f"GT {names[a['label']]} {Path(a['id']).stem[:8]}\nERM {names[a['global_prediction']]} | learned1 {names[p['learned_rank1']]}\nfixed1 {names[p['fixed_rank1']]} | learned9 {names[p['learned_rank9']]}",fontsize=9)
 fig.suptitle('Source-out low-rank correction: harms, learned-only repairs, other changes; no target');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(DEST/'cases_4x4.png',dpi=120);plt.close(fig);write(DEST/'grid_manifest.json',chosen)
 print(json.dumps(records,indent=2))
if __name__=='__main__':main()
