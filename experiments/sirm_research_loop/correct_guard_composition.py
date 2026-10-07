"""Source-only expert/router and norm controls after OFF completion."""
import json,copy
from pathlib import Path
import torch
import numpy as np
import torch.nn.functional as F
from .routed_discovery_run import ROOT,DEST as PRIOR
from .routed_conditional_utility import load
from .source_probe_penalty_direction import outputs
from .shrinkage import digest,write
DEST=PRIOR.parent/'iteration_16';OUT=DEST/'source_analysis'
def run():
 torch.set_num_threads(4);assert (DEST/'experiment/runs/L38/OFF/done').exists();assert not (OUT/'composition_controls.json').exists();cache=PRIOR.parent/'iteration_12/frozen_token_source_diagnosis';data={};locks={}
 for split in ['source_in','source_out']:
  f=cache/(split+'_tokens.npz');locks[str(f)]=digest(f);a=dict(np.load(f));data[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']),a['ids'])
 results=[]
 for state in ['selected','last']:
  models={};before={};parts={};steps={}
  for name,root in [('prior',PRIOR),('guard',DEST)]:
   m,p,path=load(root,state);locks[str(path)]=digest(path);before[name]=copy.deepcopy(m.state_dict());raw,y,dom,_=data['source_in'];m.calibrate_features(raw,dom);steps[name]=p.get('selected_step',p.get('step'));models[name]=m;parts[name]={}
   for split,(raw,y,dom,ids) in data.items():
    with torch.no_grad():pi=m.router(m.bank_features(raw)).softmax(1);g,d,f=outputs(m,raw,pi)
    parts[name][split]=(g,d,pi)
  alpha={}
  # Lock scaling from source-in correction RMS, before source-out control replay.
  for name in ['prior','guard']:
   g,d,pi=parts[name]['source_in'];correction=(pi[:,:6,None]*d).sum(1);dom=data['source_in'][2];alpha[name]=torch.stack([correction[dom==v].square().mean() for v in dom.unique()]).mean().sqrt()
  ratio=float(alpha['prior']/alpha['guard']);assert np.isfinite(ratio) and ratio>0
  controls={}
  for experts,router,scale,label in [('prior','prior',1.,'prior_native'),('guard','guard',1.,'guard_native'),('prior','guard',1.,'prior_experts_guard_router'),('guard','prior',1.,'guard_experts_prior_router'),('guard','guard',ratio,'guard_source_in_RMS_matched')]:
   controls[label]={}
   for split,(raw,y,dom,ids) in data.items():
    g,d,_=parts[experts][split];pg,_,pi=parts[router][split];assert torch.equal(g,pg);f=g+scale*(pi[:,:6,None]*d).sum(1);gc=g.argmax(1)==y;fc=f.argmax(1)==y;ce=F.cross_entropy(f,y,reduction='none');controls[label][split]=dict(accuracy=float(torch.stack([fc[dom==v].float().mean() for v in dom.unique()]).mean()),CE=float(torch.stack([ce[dom==v].mean() for v in dom.unique()]).mean()),repairs=int((~gc&fc).sum()),harms=int((gc&~fc).sum()),cases={prefix:dict(prediction=int(f[i].argmax()),CE=float(ce[i])) for prefix in ['59f2c51e','598c5b04','59bfe004','5a096976'] for i in [next(i for i,id in enumerate(ids) if prefix in id)]} if split=='source_out' else {})
  results.append(dict(state=state,steps=steps,guard_to_prior_source_in_RMS_scale=ratio,scale_selection='equal-domain source-in correction RMS only; not fitted to accuracy, source-out or target',controls=controls))
  for name,m in models.items():m.load_state_dict(before[name]);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before[name].items());assert all(p.grad is None for p in m.parameters())
 for f,h in locks.items():assert digest(Path(f))==h
 write(OUT/'composition_controls.json',dict(source_only=True,target_read=False,parameter_updates=False,fitting=False,rows=results,input_sha256=locks,code_sha256=digest(Path(__file__)),limits='Expert/router recombination is diagnostic, not a trained treatment or deployable selection. Selected steps differ; common399 is the equal-time comparison. Uniform norm matching does not isolate all nonlinear dynamics.'))
 (OUT/'composition_code_snapshot.py').write_bytes(Path(__file__).read_bytes());print('Source composition controls complete')
if __name__=='__main__':run()
