"""Source-only axis-seed replication EDA; all seeds reported, none selected."""
import json
import numpy as np,torch
import torch.nn.functional as F
from .shrinkage import ROOT,write,digest
from .source_low_rank_readout import CACHE
from .source_rank1_initialization import DEST

def main():
 torch.set_num_threads(4);root=DEST.parent;data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};raw={s:torch.tensor(np.concatenate([v['cls'],v['patch_max']],1)) for s,v in data.items()};g=torch.tensor(data['source_out']['global_logits']);y=torch.tensor(data['source_out']['y']);dom=torch.tensor(data['source_out']['domains']);gc=g.argmax(1).eq(y);results={};cases=[]
 for fit in ['pooled','held0','held2','held3']:
  held=None if fit=='pooled' else int(fit[-1]);keep=np.ones(len(data['source_in']['y']),bool) if held is None else data['source_in']['domains']!=held;take=torch.ones(len(y),dtype=torch.bool) if held is None else dom.eq(held);deltas={};norms={}
  for seed in [0,1,2]:
   for policy in ['learned','fixed']:
    key=f'{policy}_seed{seed}'
    p=(root/('source_low_rank_readout' if policy=='learned' else 'source_low_rank_readout_fixed_axis')/f'{fit}_rank1.pt') if seed==0 else DEST/f'{fit}_{key}.pt'
    a=torch.load(p,map_location='cpu',weights_only=False);w=(a['b']-a['b'].mean(0,keepdim=True))@a['a'];z=(raw['source_in'][keep]-a['mean'])/a['std']/2**.5;v=(raw['source_out']-a['mean'])/a['std']/2**.5;deltas[key]=F.linear(v,w);norms[key]=float(F.linear(z,w).square().mean().sqrt())
  common=float(np.mean(list(norms.values())));controls={}
  for key,d in deltas.items():
   for suffix,delta in [('raw',d),('common_trainRMS',d*common/norms[key])]:
    f=g+delta;fc=f.argmax(1).eq(y);ce=F.cross_entropy(f,y,reduction='none');controls[key+'_'+suffix]=dict(accuracy=float(torch.stack([fc[take&dom.eq(dd)].float().mean() for dd in dom[take].unique()]).mean()),CE=float(torch.stack([ce[take&dom.eq(dd)].mean() for dd in dom[take].unique()]).mean()),repairs=int((take&~gc&fc).sum()),harms=int((take&gc&~fc).sum()))
  results[fit]=dict(common_trainRMS=common,original_trainRMS=norms,controls=controls)
  if fit=='pooled':
   ps={k:(g+d).argmax(1) for k,d in deltas.items()}
   for i in range(len(y)):
    preds={k:int(p[i]) for k,p in ps.items()};gt=int(y[i]);gp=int(g[i].argmax());good=sum(preds[f'learned_seed{s}']==gt for s in [0,1,2])
    if any(v!=gp for v in preds.values()):cases.append(dict(id=str(data['source_out']['ids'][i]),label=gt,global_prediction=gp,predictions=preds,learned_correct_seed_count=good,global_correct_new_harms=gp==gt and good<3,seed0_repair_not_replicated=gp!=gt and preds['learned_seed0']==gt and good<3))
 write(DEST/'amplitude_controls.json',dict(source_only=True,target_read=False,common_amplitude_is_mean_of_six_source_in_RMS=True,fits=results));write(DEST/'cases.json',cases)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image
 names=json.loads((root/'experiment/subset_manifest.json').read_text())['class_names'];chosen=sorted(cases,key=lambda x:(not x['global_correct_new_harms'],not x['seed0_repair_not_replicated'],x['id']))[:16];fig,axs=plt.subplots(4,4,figsize=(16,16))
 for ax in axs.flat:ax.axis('off')
 for ax,x in zip(axs.flat,chosen):
  with Image.open(ROOT/'domainbed/data/terra_incognita'/x['id']) as im:ax.imshow(im.convert('RGB'))
  p=x['predictions'];ax.set_title(f"GT {names[x['label']]} {__import__('pathlib').Path(x['id']).stem[:8]}\nERM {names[x['global_prediction']]}\nlearned seeds0/1/2: {' / '.join(names[p[f'learned_seed{s}']] for s in [0,1,2])}",fontsize=9)
 fig.suptitle('Rank1 initialization sensitivity on source-out: harms first; no target, no selected seed');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(DEST/'cases_4x4.png',dpi=120);plt.close(fig);write(DEST/'grid_manifest.json',chosen)
 print(json.dumps(results['pooled'],indent=2))
if __name__=='__main__':main()
