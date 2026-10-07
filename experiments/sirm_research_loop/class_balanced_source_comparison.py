"""Source-only original/equal-class bank capacity at selected and common399 states."""
import json
from pathlib import Path
import numpy as np
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from .class_balanced_run import DEST,PRIOR,ROOT
from .spatial_bank_model import SpatialBankGateSIRM
from .spatial_bank_replay import spatial_head
from .bias_control import scores
from .shrinkage import write,digest
@torch.no_grad()
def run():
 torch.set_num_threads(4);cache=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/source_out_tokens.npz';audit=json.loads((cache.parent/'extraction_audit.json').read_text());assert digest(cache)==next(r['cache_sha256'] for r in audit['rows'] if r['split']=='source_out')
 with np.load(cache) as a:raw=torch.tensor(np.concatenate([a['cls'],a['patch_max']],1));y=torch.tensor(a['y']);dom=a['domains'].copy();ids=a['ids'].copy();cached_g=torch.tensor(a['global_logits'])
 out=DEST/'source_common_state';out.mkdir(exist_ok=True);results={};arrays={}
 for label,root,arm in [('original_OFF',PRIOR,'OFF'),('equal_OFF',DEST,'OFF'),('equal_ON',DEST,'ON')]:
  for state,file in [('selected','source_best.pkl'),('last','last.pkl')]:
   bank=root/'experiment/runs/L38'/arm;path=bank/file;payload=torch.load(path,map_location='cpu',weights_only=False);c=payload['config'];m=SpatialBankGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c);m.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});m.configure_stage();m.eval()
   g,d,pi=spatial_head(m,raw);full=g+(d*pi[:,:6,None]).sum(1);assert torch.equal(g.argmax(1),cached_g.argmax(1));v,pred=scores(full,y,dom);wrong=g.argmax(1).ne(y);right=full.argmax(1).eq(y);capacity=(g[:,None]+d).argmax(2).eq(y[:,None]).any(1);ix=torch.arange(len(y));competitor=g.clone();competitor[ix,y]=-torch.inf;competitor=competitor.argmax(1);gap=g[ix,y]-g[ix,competitor];correction=full-g;added=correction[ix,y]-correction[ix,competitor]
   v.update(step=payload.get('selected_step',payload.get('step')),repairs=int((wrong&right).sum()),harms=int((~wrong&~right).sum()),global_errors=int(wrong.sum()),unscaled_single_expert_capacity=int((wrong&capacity).sum()),wrong_global_margin_median=float(np.median(gap[wrong].numpy())),wrong_margin_correction_median=float(np.median(added[wrong].numpy())),checkpoint_sha256=digest(path),per_class=[{'label':label,'n':int(y.eq(label).sum()),'global_errors':int((y.eq(label)&wrong).sum()),'repairs':int((y.eq(label)&wrong&right).sum()),'harms':int((y.eq(label)&~wrong&~right).sum()),'capacity':int((y.eq(label)&wrong&capacity).sum())} for label in range(10)])
   if state=='selected':
    locked=json.loads((bank/'posthoc/complete.json').read_text());assert abs(v['accuracy']-locked['splits']['source_out']['source_mean'])<1e-12
   results[f'{label}/{state}']=v;arrays[f'{label}_{state}']=full.numpy()
   del m
 np.savez_compressed(out/'source_outputs.npz',ids=ids,y=y.numpy(),domains=dom,**arrays);write(out/'results.json',{'source_only':True,'target_read':False,'training':False,'fitting':False,'source_token_sha256':digest(cache),'states':results,'scope':'Frozen cached original tokens, all selected and common399 source states. Label-informed unscaled expert capacity is diagnostic only; actual repairs/harms and source selection remain primary.'});return results
if __name__=='__main__':print(json.dumps(run(),indent=2))
