"""Immutable source-only diagnostic reader EDA and train-RMS controls."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from .source_context_reader import DEST
from .shrinkage import write,digest,ROOT

def main():
 torch.set_num_threads(4);cache=torch.load(DEST/'features.pt',weights_only=False,map_location='cpu');results=json.loads((DEST/'results.json').read_text());assert results['cache_sha256']==digest(DEST/'features.pt');allcontrols={};cases=[]
 for fit in ['pooled','held0','held2','held3']:
  held=None if fit=='pooled' else int(fit[-1]);keep=torch.ones(len(cache['y']),dtype=torch.bool) if held is None else cache['domains'].ne(held);take=torch.ones(len(cache['validation_y']),dtype=torch.bool) if held is None else cache['validation_domains'].eq(held);ds={};trainRMS={}
  for policy in ['original','paired_context']:
   a=torch.load(DEST/f'{fit}_{policy}_reader.pt',weights_only=False,map_location='cpu');head=torch.nn.Linear(768,10);head.load_state_dict(a['head']);z=(cache['raw']-a['mean'])/a['std']/2**.5;v=(cache['validation_raw']-a['mean'])/a['std']/2**.5
   with torch.no_grad():d=head(z[keep]);d-=d.mean(1,keepdim=True);trainRMS[policy]=float(d.square().mean().sqrt());ds[policy]=head(v);ds[policy]-=ds[policy].mean(1,keepdim=True)
  controls={};g=cache['validation_global'];y=cache['validation_y'];dom=cache['validation_domains'];globalcorrect=g.argmax(1).eq(y)
  for name,d in [('global',torch.zeros_like(g)),('original',ds['original']),('paired_context',ds['paired_context']),('original_to_paired_trainRMS',ds['original']*trainRMS['paired_context']/trainRMS['original']),('paired_to_original_trainRMS',ds['paired_context']*trainRMS['original']/trainRMS['paired_context'])]:
   f=g+d;fc=f.argmax(1).eq(y);ce=F.cross_entropy(f,y,reduction='none');controls[name]=dict(accuracy=float(torch.stack([fc[take&dom.eq(dd)].float().mean() for dd in dom[take].unique()]).mean()),CE=float(torch.stack([ce[take&dom.eq(dd)].mean() for dd in dom[take].unique()]).mean()),repairs=int((take&~globalcorrect&fc).sum()),harms=int((take&globalcorrect&~fc).sum()),centered_correction_RMS=float(d[take].square().mean().sqrt()))
  allcontrols[fit]=dict(train_centered_correction_RMS=trainRMS,controls=controls)
  if fit=='pooled':
   orig=(g+ds['original']).argmax(1);pair=(g+ds['paired_context']).argmax(1)
   for i in range(len(y)):
    if orig[i]!=pair[i] or pair[i]!=g[i].argmax():cases.append(dict(id=str(cache['validation_ids'][i]),label=int(y[i]),global_prediction=int(g[i].argmax()),original_reader_prediction=int(orig[i]),paired_reader_prediction=int(pair[i]),paired_new_harm_vs_original=bool(orig[i]==y[i] and pair[i]!=y[i]),paired_new_repair_vs_original=bool(orig[i]!=y[i] and pair[i]==y[i])))
 write(DEST/'norm_matched_controls.json',dict(source_only=True,target_read=False,fitting=False,scope='Frozen final heads; amplitudes computed from source-in original centered corrections with matching training exclusion, never source-out normalization',fits=allcontrols));write(DEST/'cases.json',cases)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image
 names=json.loads((DEST.parent/'experiment/subset_manifest.json').read_text())['class_names'];selected=sorted(cases,key=lambda x:(not x['paired_new_harm_vs_original'],not x['paired_new_repair_vs_original'],x['id']))[:16];fig,axes=plt.subplots(4,4,figsize=(16,16))
 for ax in axes.flat:ax.axis('off')
 for ax,x in zip(axes.flat,selected):
  with Image.open(ROOT/'domainbed/data/terra_incognita'/x['id']) as im:ax.imshow(im.convert('RGB'))
  ax.set_title(f"GT {names[x['label']]} {Path(x['id']).stem[:8]}\nERM {names[x['global_prediction']]} | orig {names[x['original_reader_prediction']]}\ncontext {names[x['paired_reader_prediction']]}",fontsize=9)
 fig.suptitle('Pooled source-out: context-reader new harms first, then repairs; selected cases, no target');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(DEST/'reader_cases_4x4.png',dpi=120);plt.close(fig);write(DEST/'reader_cases_manifest.json',selected)
 print(json.dumps(allcontrols,indent=2),flush=True)
if __name__=='__main__':main()
