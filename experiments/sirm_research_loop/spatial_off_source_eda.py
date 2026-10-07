"""Early source-only EDA of the completed spatial OFF bank."""
import json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from domainbed.deit_transformer import DeiTFeaturizer
from .spatial_bank_model import SpatialBankGateSIRM
from .spatial_bank_replay import spatial_head
from .spatial_bank_run import ROOT,DEST,PRIOR
from .shrinkage import write,digest
from .bias_control import scores

@torch.no_grad()
def run():
 torch.set_num_threads(4);bank=DEST/'experiment/runs/L38/OFF';assert (bank/'done').exists();out=DEST/'spatial_OFF_source_EDA';out.mkdir(exist_ok=True)
 path=bank/'source_best.pkl';sha=digest(path);p=torch.load(path,map_location='cpu',weights_only=False);c=p['config'];assert c['bank_input_mode']=='cls_patch_max'
 m=SpatialBankGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c);m.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()},strict=True);m.configure_stage();m.eval()
 cache=PRIOR/'frozen_token_source_diagnosis/source_out_tokens.npz';audit=json.loads((PRIOR/'frozen_token_source_diagnosis/extraction_audit.json').read_text());assert digest(cache)==next(r['cache_sha256'] for r in audit['rows'] if r['split']=='source_out')
 with np.load(cache) as a:raw=torch.tensor(np.concatenate([a['cls'],a['patch_max']],1));y=torch.tensor(a['y']);ids=a['ids'].copy();domains=a['domains'].copy();oldg=torch.tensor(a['global_logits'])
 g,delta,pi=spatial_head(m,raw);full=g+(delta*pi[:,:6,None]).sum(1);assert torch.equal(g.argmax(1),oldg.argmax(1));sc,pred=scores(full,y,domains);assert abs(sc['accuracy']-json.loads((bank/'selection.json').read_text())['source_val_mean'])<1e-12
 np.savez_compressed(out/'source_replay.npz',ids=ids,y=y.numpy(),domains=domains,g=g.numpy(),delta=delta.numpy(),pi=pi.numpy(),full=full.numpy())
 gp=g.argmax(1);fp=full.argmax(1);globalright=gp.eq(y);right=fp.eq(y);wrong=~globalright;capacity=(g[:,None,:]+delta).argmax(-1).eq(y[:,None]).any(1)
 ce=F.cross_entropy(full,y,reduction='none');terms=delta*pi[:,:6,None];u=torch.stack([F.cross_entropy(full-terms[:,e],y,reduction='none')-ce for e in range(6)],1)
 comp=g.masked_fill(F.one_hot(y,10).bool(),-torch.inf).argmax(1);idx=torch.arange(len(y));basegap=g[idx,y]-g[idx,comp];corr=full-g;added=corr[idx,y]-corr[idx,comp]
 manifest=json.loads((DEST/'experiment/subset_manifest.json').read_text());names=manifest['class_names'];lookup={r['id']:r for d in [0,2,3] for r in manifest['sources'][str(d)]['images']}
 rows=[]
 for i in torch.nonzero(globalright!=right).flatten().tolist():
  image=str(ids[i]);file=ROOT/'domainbed/data/terra_incognita'/image;assert digest(file)==lookup[image]['sha256']
  loo=(full[i][None,:]-terms[i]).argmax(1)
  rows.append(dict(image_id=image,original_sha256=digest(file),domain=int(domains[i]),label=names[int(y[i])],global_prediction=names[int(gp[i])],spatial_prediction=names[int(fp[i])],repair=bool(right[i]),global_margin=float(basegap[i]),margin_correction=float(added[i]),router_NULL=float(pi[i,-1]),marginal_U=u[i].tolist(),experts_whose_removal_reverts_to_global=torch.nonzero(loo.eq(gp[i])).flatten().tolist()))
 groups={}
 for d in [0,2,3]:
  for cls in y.unique().tolist():
   mask=torch.tensor(domains==d)&y.eq(cls)
   if int(mask.sum())<2:continue
   groups[f'{d}/{cls}']=dict(n=int(mask.sum()),global_wrong=int((wrong&mask).sum()),capacity=int((wrong&capacity&mask).sum()),repairs=int((wrong&right&mask).sum()),harms=int((globalright&~right&mask).sum()),mean_U=u[mask].mean(0).tolist())
 result=dict(source_only=True,target_read=False,checkpoint_sha256=sha,selected_step=p['selected_step'],accuracy=sc['accuracy'],global_correct=int(globalright.sum()),spatial_correct=int(right.sum()),repairs=sum(r['repair'] for r in rows),harms=sum(not r['repair'] for r in rows),single_unscaled_capacity_on_global_errors=int((wrong&capacity).sum()),wrong_global_margin_median=float(np.median(basegap[wrong].numpy())),wrong_correction_median=float(np.median(added[wrong].numpy())),groups_minimum2=groups,cases=rows,original_visual_review_complete=False,scope='Completed spatial OFF bank, fixed source-out original cached norm tokens; selected accuracy verified. Source-only exploratory case evidence, not candidate selection or spatial/SIRM result.')
 write(out/'results.json',result)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image
 if rows:
  fig,axs=plt.subplots(math.ceil(len(rows)/3),3,figsize=(15,4*math.ceil(len(rows)/3)),squeeze=False)
  for ax in axs.flat:ax.axis('off')
  for ax,r in zip(axs.flat,rows):
   ax.imshow(Image.open(ROOT/'domainbed/data/terra_incognita'/r['image_id']).convert('RGB'));ax.set_title(f"env{r['domain']} true={r['label']}\nERM={r['global_prediction']} → spatial={r['spatial_prediction']}\n{Path(r['image_id']).name[:12]}",fontsize=10)
  fig.tight_layout();fig.savefig(out/'all_source_correctness_changes.png',dpi=110);plt.close(fig)
 assert digest(path)==sha
 print(json.dumps({k:result[k] for k in ['selected_step','accuracy','global_correct','spatial_correct','repairs','harms','single_unscaled_capacity_on_global_errors','wrong_global_margin_median','wrong_correction_median']}))
 return result

if __name__=='__main__':run()
