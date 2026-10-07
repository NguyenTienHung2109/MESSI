"""Locked completed OFF source cases before any target or candidate promotion."""
import json
from pathlib import Path
import numpy as np
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover.math import residual_utilities
from .spatial_bank_model import SpatialBankGateSIRM
from .spatial_bank_replay import spatial_head
from .class_balanced_run import DEST,ROOT
from .bias_control import scores
from .shrinkage import digest,write
@torch.no_grad()
def run():
 torch.set_num_threads(4);bank=DEST/'experiment/runs/L38/OFF';assert (bank/'done').exists();selection=json.loads((bank/'selection.json').read_text());assert not selection['target_used'];path=bank/'source_best.pkl';payload=torch.load(path,map_location='cpu',weights_only=False);assert payload['selected_step']==selection['selected_step'];c=payload['config']
 cache=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/source_out_tokens.npz';audit=json.loads((cache.parent/'extraction_audit.json').read_text());assert digest(cache)==next(r['cache_sha256'] for r in audit['rows'] if r['split']=='source_out')
 with np.load(cache) as a:raw=torch.tensor(np.concatenate([a['cls'],a['patch_max']],1));y=torch.tensor(a['y']);domains=a['domains'];ids=a['ids'];cachedg=a['global_logits']
 m=SpatialBankGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c);m.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});m.configure_stage();m.eval();g,delta,pi=spatial_head(m,raw);full,_,u,_=residual_utilities(g,delta,pi,y);v,pred=scores(full,y,domains);assert abs(v['accuracy']-selection['source_val_mean'])<1e-12;assert np.array_equal(g.argmax(1).numpy(),cachedg.argmax(1));glob=g.argmax(1);right=full.argmax(1).eq(y);wrong=glob.ne(y);capacity=(g[:,None]+delta).argmax(2).eq(y[:,None]).any(1)
 ix=torch.arange(len(y));comp=g.clone();comp[ix,y]=-torch.inf;comp=comp.argmax(1);gap=g[ix,y]-g[ix,comp];correction=full-g;added=correction[ix,y]-correction[ix,comp]
 manifest=json.loads((DEST/'experiment/subset_manifest.json').read_text());names=manifest['class_names'];lookup={r['id']:r for env in manifest['sources'].values() for r in env['images']};cases=[]
 for i in torch.nonzero(wrong.eq(right)).flatten().tolist():
  image=str(ids[i]);sha=digest(ROOT/'domainbed/data/terra_incognita'/image);assert sha==lookup[image]['sha256'];loo=(full[i,None]-pi[i,:6,None]*delta[i]).argmax(1)
  cases.append({'image_id':image,'sha256':sha,'domain':int(domains[i]),'class':names[int(y[i])],'ERM_prediction':names[int(glob[i])],'equal_class_prediction':names[int(pred[i])],'repair':bool(right[i]),'global_margin_to_fixed_competitor':float(gap[i]),'routed_margin_correction':float(added[i]),'router_NULL':float(pi[i,-1]),'marginal_CE_utility':u[i].tolist(),'removal_predictions':[names[int(v)] for v in loo]})
 groups=[{'label':c,'class':name,'n':int(y.eq(c).sum()),'global_errors':int((y.eq(c)&wrong).sum()),'capacity':int((y.eq(c)&wrong&capacity).sum()),'repairs':int((y.eq(c)&wrong&right).sum()),'harms':int((y.eq(c)&~wrong&~right).sum())} for c,name in enumerate(names)]
 out=DEST/'early_OFF_source_EDA';out.mkdir(exist_ok=True);result={'source_only':True,'target_read':False,'training':False,'fitting':False,'checkpoint_sha256':digest(path),'source_token_sha256':digest(cache),'step':payload['selected_step'],'accuracy':v['accuracy'],'repairs':int((wrong&right).sum()),'harms':int((~wrong&~right).sum()),'global_errors':int(wrong.sum()),'unscaled_single_expert_capacity':int((wrong&capacity).sum()),'global_wrong_margin_median':float(np.median(gap[wrong].numpy())),'wrong_margin_correction_median':float(np.median(added[wrong].numpy())),'cases':cases,'classes':groups,'visual_review_complete':False,'scope':'Completed source-selected OFF bank, original cached tokens and all source correctness-changing originals hashed; full ON/control/target comparison remains pending. No candidate promotion.'};write(out/'results.json',result)
 np.savez_compressed(out/'source_replay.npz',ids=ids,y=y.numpy(),domains=domains,g=g.numpy(),delta=delta.numpy(),pi=pi.numpy(),full=full.numpy(),utility=u.numpy())
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image
 if cases:
  fig,axes=plt.subplots((len(cases)+3)//4,4,figsize=(16,4*((len(cases)+3)//4)),squeeze=False)
  for ax in axes.flat:ax.axis('off')
  for ax,row in zip(axes.flat,cases):ax.imshow(Image.open(ROOT/'domainbed/data/terra_incognita'/row['image_id']).convert('RGB'));ax.set_title(f"env{row['domain']} {row['class']}\n{row['ERM_prediction']} → {row['equal_class_prediction']}\n{Path(row['image_id']).stem[:8]}")
  fig.tight_layout();fig.savefig(out/'all_source_changes.png',dpi=120);plt.close(fig)
 print(json.dumps({k:v for k,v in result.items() if k not in ['cases']},indent=2));return result
if __name__=='__main__':run()
