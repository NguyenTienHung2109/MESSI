"""Source-only component-basis comparison visualization and prediction audit."""
import json
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .source_rank3_component_payoff import OUT,DEST,CACHE
from .shrinkage import ROOT,write

def main():
 torch.set_num_threads(4);result=json.loads((OUT/'results.json').read_text());names=json.loads((DEST.parent/'experiment/subset_manifest.json').read_text())['class_names'];bases=result['records']['pooled']['learned']['bases'];fig,axs=plt.subplots(1,5,figsize=(17,4))
 for ax,(name,x) in zip(axs,bases.items()):
  u=np.array(x['source_out']['U']);im=ax.imshow(u,cmap='RdBu',vmin=-.045,vmax=.045)
  for i in range(3):
   for j in range(3):ax.text(j,i,f'{u[i,j]:+.3f}',ha='center',va='center',fontsize=9)
  ax.set_xticks(range(3),['C0','C1','C2']);ax.set_yticks(range(3),['loc100','loc43','loc46']);ax.set_title(name.replace('SVD_rotation_seed','rotate ').replace('effective_W_SVD','SVD').replace('parameter_columns','parameter')+f"\nshared={x['source_out']['diagnostic_shared_components']}",fontsize=10)
 fig.suptitle('Same rank3 predictor, different component LOO utilities; diagnostic support only, no learned Q');fig.tight_layout(rect=[0,0,1,.87]);fig.savefig(OUT/'basis_utility.png',dpi=140);plt.close(fig)
 data=dict(np.load(CACHE/'source_out_tokens.npz'));raw=torch.tensor(np.concatenate([data['cls'],data['patch_max']],1));g=torch.tensor(data['global_logits']);yy=torch.tensor(data['y']);a=torch.load(DEST/'pooled_learned_risk_axis.pt',map_location='cpu',weights_only=False);b=a['b']-a['b'].mean(0,keepdim=True);w=b@a['a'];z=(raw-a['mean'])/a['std']/2**.5;u,s,vh=torch.linalg.svd(w,full_matrices=False);factors={'parameter':(z@a['a'].T)[:,:,None]*b.T[None,:,:],'SVD':(z@(s[:3,None]*vh[:3]).T)[:,:,None]*u[:,:3].T[None,:,:]};full=g+F.linear(z,w);gp=g.argmax(1);fp=full.argmax(1);audit={};selected=[]
 for name,comp in factors.items():
  replay=g+comp.sum(1);audit[name]=dict(max_logit_error=float((replay-full).abs().max()),prediction_disagreement=int((replay.argmax(1)!=fp).sum()));assert audit[name]['prediction_disagreement']==0
 for i in range(len(yy)):
  if fp[i]!=gp[i]:selected.append(i)
 selected=sorted(selected,key=lambda i:(not(gp[i]==yy[i] and fp[i]!=yy[i]),not(gp[i]!=yy[i] and fp[i]==yy[i]),str(data['ids'][i])))[:16]
 for i in sorted(range(len(yy)),key=lambda i:str(data['ids'][i])):
  if len(selected)>=16:break
  if i not in selected and gp[i]!=yy[i]:selected.append(i)
 fig,axs=plt.subplots(4,4,figsize=(16,16));manifest=[]
 for ax,i in zip(axs.flat,selected):
  ax.axis('off');id=str(data['ids'][i]);removed={k:[int((full[i]-comp[i,m]).argmax()) for m in range(3)] for k,comp in factors.items()};manifest.append(dict(id=id,label=int(yy[i]),global_prediction=int(gp[i]),full_prediction=int(fp[i]),removed_predictions=removed))
  with Image.open(ROOT/'domainbed/data/terra_incognita'/id) as image:ax.imshow(image.convert('RGB'))
  ax.set_title(f"GT {names[int(yy[i])]} {Path(id).stem[:8]} | full {names[int(fp[i])]}\nremove parameter C0/1/2: {'/'.join(names[j] for j in removed['parameter'])}\nremove SVD C0/1/2: {'/'.join(names[j] for j in removed['SVD'])}",fontsize=8)
 fig.suptitle('Rank3 source-out cases: full prediction fixed, component-removal responses depend on basis');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(OUT/'component_cases_4x4.png',dpi=120);plt.close(fig);write(OUT/'grid_manifest.json',manifest);write(OUT/'prediction_basis_audit.json',dict(source_only=True,target_read=False,pooled_learned=audit))
 print(audit)
if __name__=='__main__':main()
