"""Normalized-domain source gradient axes: payoff and case EDA."""
import json
import numpy as np,torch
import torch.nn.functional as F
from pathlib import Path
from .source_low_rank_readout import CACHE
from .source_rank1_domain_agreement import DEST
from .shrinkage import ROOT,write,digest

def main():
 torch.set_num_threads(4);data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};raw={s:torch.tensor(np.concatenate([v['cls'],v['patch_max']],1)) for s,v in data.items()};g=torch.tensor(data['source_out']['global_logits']);y=torch.tensor(data['source_out']['y']);dom=torch.tensor(data['source_out']['domains']);gc=g.argmax(1).eq(y);records={};cases=[]
 for fit in ['pooled','held0','held2','held3']:
  held=None if fit=='pooled' else int(fit[-1]);keep=np.ones(len(data['source_in']['y']),bool) if held is None else data['source_in']['domains']!=held;take=torch.ones(len(y),dtype=torch.bool) if held is None else dom.eq(held);deltas={};norms={};moments={}
  for policy in ['learned_mean_normalized','fixed_mean_normalized','learned_minimum_norm_convex','fixed_minimum_norm_convex']:
   a=torch.load(DEST/f'{fit}_{policy}.pt',map_location='cpu',weights_only=False);w=(a['b']-a['b'].mean(0,keepdim=True))@a['a'];z=(raw['source_in'][keep]-a['mean'])/a['std']/2**.5;v=(raw['source_out']-a['mean'])/a['std']/2**.5;delta=F.linear(v,w);norms[policy]=float(F.linear(z,w).square().mean().sqrt());deltas[policy]=delta
   if policy=='learned_minimum_norm_convex':
    b=torch.tensor(a['axis_audit']['class_axis']);m=torch.tensor(a['axis_audit']['gradient_matrix']);t=m.T@b;direction=b[:,None]*t[None,:]
    for split in ['source_in','source_out']:
     mask=keep if split=='source_in' else np.ones(len(y),bool);xx=(raw[split][mask]-a['mean'])/a['std']/2**.5;gg=torch.tensor(data[split]['global_logits'][mask]);yy=torch.tensor(data[split]['y'][mask]);dd=torch.tensor(data[split]['domains'][mask]);desired=F.one_hot(yy,10).float()-F.softmax(gg,1)
     for d in dd.unique():
      q=dd.eq(d);md=desired[q].T@xx[q]/int(q.sum());moments[f'{split}_domain{d}']=dict(initial_descent_dot=float((md*direction).sum()),cosine_with_initial_direction=float(F.cosine_similarity(md.flatten(),direction.flatten(),dim=0)),n=int(q.sum()))
  controls={}
  for name,d in [('mean_learned',deltas['learned_mean_normalized']),('mean_fixed',deltas['fixed_mean_normalized']),('learned',deltas['learned_minimum_norm_convex']),('fixed',deltas['fixed_minimum_norm_convex']),('learned_to_fixed_trainRMS',deltas['learned_minimum_norm_convex']*norms['fixed_minimum_norm_convex']/norms['learned_minimum_norm_convex']),('fixed_to_learned_trainRMS',deltas['fixed_minimum_norm_convex']*norms['learned_minimum_norm_convex']/norms['fixed_minimum_norm_convex'])]:
   f=g+d;fc=f.argmax(1).eq(y);change=F.cross_entropy(f,y,reduction='none')-F.cross_entropy(g,y,reduction='none');controls[name]=dict(accuracy=float(torch.stack([fc[take&dom.eq(dd)].float().mean() for dd in dom[take].unique()]).mean()),repairs=int((take&~gc&fc).sum()),harms=int((take&gc&~fc).sum()),per_domain={str(int(dd)):dict(CE_change=float(change[take&dom.eq(dd)].mean()),accuracy=float(fc[take&dom.eq(dd)].float().mean())) for dd in dom[take].unique()})
  records[fit]=dict(controls=controls,train_RMS=norms,initial_direction_gradient_alignment=moments)
  if fit=='pooled':
   pooled_deltas=deltas.copy()
   preds={p:(g+d).argmax(1) for p,d in [('learned',deltas['learned_minimum_norm_convex']),('fixed',deltas['fixed_minimum_norm_convex'])]}
   for i in range(len(y)):
    gp=int(g[i].argmax());gt=int(y[i]);p={k:int(v[i]) for k,v in preds.items()}
    if any(v!=gp for v in p.values()):cases.append(dict(id=str(data['source_out']['ids'][i]),label=gt,global_prediction=gp,predictions=p,learned_harm=gp==gt and p['learned']!=gt,learned_repair=gp!=gt and p['learned']==gt))
 write(DEST/'payoff_and_gradient_alignment.json',dict(source_only=True,target_read=False,fits=records,gradient_sign='Positive initial_descent_dot predicts infinitesimal CE decrease along source-in leading rank1 direction, not final accuracy. Source-out gradients diagnostic only.'));write(DEST/'cases.json',cases)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image
 names=json.loads((DEST.parent/'experiment/subset_manifest.json').read_text())['class_names'];preds={p:(g+d).argmax(1) for p,d in [('learned',pooled_deltas['learned_minimum_norm_convex']),('fixed',pooled_deltas['fixed_minimum_norm_convex'])]};chosen=sorted(cases,key=lambda x:(not x['learned_harm'],not x['learned_repair'],x['id']))[:16]
 existing={x['id'] for x in chosen}
 for i in sorted(range(len(y)),key=lambda i:str(data['source_out']['ids'][i])):
  if len(chosen)>=16:break
  id=str(data['source_out']['ids'][i]);gt=int(y[i]);gp=int(g[i].argmax())
  if id not in existing and gp!=gt:
   chosen.append(dict(id=id,label=gt,global_prediction=gp,predictions={k:int(v[i]) for k,v in preds.items()},persistent_global_error=True))
 fig,axs=plt.subplots(4,4,figsize=(16,16))
 for ax in axs.flat:ax.axis('off')
 for ax,x in zip(axs.flat,chosen):
  with Image.open(ROOT/'domainbed/data/terra_incognita'/x['id']) as im:ax.imshow(im.convert('RGB'))
  ax.set_title(f"GT {names[x['label']]} {Path(x['id']).stem[:8]}\nERM {names[x['global_prediction']]}\nagreement learned {names[x['predictions']['learned']]} | fixed {names[x['predictions']['fixed']]}",fontsize=9)
 fig.suptitle('Source domain-agreement correction: changed cases then persistent errors; no target');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(DEST/'cases_4x4.png',dpi=120);plt.close(fig);write(DEST/'grid_manifest.json',chosen)
 print(json.dumps(records,indent=2))
if __name__=='__main__':main()
