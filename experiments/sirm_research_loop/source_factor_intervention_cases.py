"""Per-image margins for frozen source-only brightness/contrast interventions."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.sirm_discover.math import center_logits
from .routed_discovery_run import ROOT,DEST,PRIOR
from .routed_conditional_utility import load
from .label_expected_utility_helpers import predict_label,expected_utility
from .teacher_distribution_helpers import teacher
from .shrinkage import digest,write

def margin(logits,y):
 other=logits.copy();other[np.arange(len(y)),y]=-np.inf
 return logits[np.arange(len(y)),y]-other.max(1)

@torch.no_grad()
def run():
 torch.set_num_threads(4);out=DEST/'source_factor_intervention_cases';out.mkdir(exist_ok=True);assert not (out/'results.json').exists()
 source={s:dict(np.load(ROOT/f'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/{s}_tokens.npz')) for s in ['source_in','source_out']}
 posts=json.loads((DEST/'source_label_expected_utility/results.json').read_text());rank=json.loads((DEST/'source_residual_rank_diagnosis/results.json').read_text());rows=[];locks={};s=source['source_out'];y=s['y'];g0=s['global_logits'].astype(float);original=np.concatenate([s['cls'],s['patch_max']],1)
 for intervention in ['brightness','contrast']:
  folder=DEST/f'source_factor_{intervention}_intervention';proof=json.loads((folder/'results.json').read_text());cache=folder/'paired_source_out_raw.npz';assert digest(cache)==proof['paired_cache_sha256'];locks[str(cache)]=digest(cache);data=dict(np.load(cache));assert np.array_equal(data['ids'],s['ids'])
  for state in rank['states']:
   m,p,path=load(PRIOR if state.startswith('unscaled') else DEST,'selected');assert digest(path)==proof['checkpoint_sha256'][state];before={k:v.clone() for k,v in m.state_dict().items()}
   si=source['source_in'];m.calibrate_features(torch.tensor(np.concatenate([si['cls'],si['patch_max']],1)),torch.tensor(si['domains']))
   bankpath=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';assert digest(bankpath)==rank['input_sha256'][str(bankpath)];bank=dict(np.load(bankpath));d0=bank['source_out_bank_delta'].astype(float);native0=bank['source_out_bank_pi'].astype(float)
   rec=next(x for x in posts['states'][state] if x['input']=='CLS_plus_patch_max' and x['cohort']=='original');fit=DEST/'source_label_expected_utility'/rec['fit_file'];assert digest(fit)==rec['fit_sha256'];posterior=dict(np.load(fit));post0=predict_label(posterior,original.astype(float));pi0=teacher(expected_utility(g0,d0,post0))
   for factor in [1.,.7,1.3]:
    arr=data['raw_'+str(factor)];raw=torch.tensor(arr);z=m.bank_features(raw);g=m.global_head(raw[:,:384]).numpy().astype(float);delta=(center_logits(torch.stack([e(z) for e in m.residuals],1))-m.expert_source_mean[None]).numpy().astype(float);native=m.router(z).softmax(1).numpy().astype(float);pi=teacher(expected_utility(g,delta,predict_label(posterior,arr.astype(float))))
    views={'global':g,'native_full':g+np.einsum('nm,nmc->nc',native[:,:6],delta),'posterior_full':g+np.einsum('nm,nmc->nc',pi[:,:6],delta),'native_residual_held':g0+np.einsum('nm,nmc->nc',native0[:,:6],delta),'posterior_residual_held':g0+np.einsum('nm,nmc->nc',pi0[:,:6],delta),'posterior_routing_only':g0+np.einsum('nm,nmc->nc',pi[:,:6],d0)}
    margins={k:margin(v,y) for k,v in views.items()};rawmargin=margin((g0[:,None]+delta).reshape(-1,10),np.repeat(y,6)).reshape(-1,6)
    for i,id in enumerate(s['ids']):
     rows.append(dict(intervention=intervention,state=state,factor=factor,image_id=str(id),domain=int(s['domains'][i]),label=int(y[i]),global_original_margin=float(margin(g0,y)[i]),margins={k:float(v[i]) for k,v in margins.items()},predictions={k:int(v[i].argmax()) for k,v in views.items()},standalone_original_global_margin=rawmargin[i].tolist(),posterior_pi=pi[i].tolist(),original_posterior_pi=pi0[i].tolist()))
   m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and all(v.grad is None for v in m.parameters())
 for path,h in locks.items():assert digest(Path(path))==h
 write(out/'results.json',dict(source_only=True,target_read=False,bank_updates=False,posterior_refitting=False,minimum_group_samples=2,rows=rows,input_sha256=locks,code_sha256=digest(Path(__file__))))
 (out/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':run()
