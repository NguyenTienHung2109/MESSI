"""First-order native residual discovery guard; scalar-preserving gradient surrogate."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import center_logits
from .discovery_error_gradient_diagnosis import vector
from .native_guard_projection import project
from .shrinkage import digest
CACHE=Path('/home/ntduc/Desktop/Dat/MESSI/results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/source_in_tokens.npz')
SHA='84b68726b34446b35aa8764999be5bc2ae90dd2567d68f95b4fa0c2023de39a6'

def guard_discovery(model,chosen,params):
 if not hasattr(model,'_correct_guard_source'):
  assert digest(CACHE)==SHA;a=np.load(CACHE);raw=torch.tensor(np.concatenate([a['cls'],a['patch_max']],1),device=params[0].device);y=torch.tensor(a['y'],device=raw.device);dom=torch.tensor(a['domains'],device=raw.device);assert set(dom.tolist())=={0,2,3} and len(y)==1172
  with torch.no_grad():z=model.bank_features(raw);g=model.global_head(raw[:,:384]);correct=g.argmax(1)==y
  model._correct_guard_source=(z,g,y,dom,correct)
 z,g,y,dom,correct=model._correct_guard_source
 with torch.no_grad():pi=model.router(z).softmax(1)
 delta=center_logits(torch.stack([e(z) for e in model.residuals],1));mean=torch.stack([delta[dom==d].mean(0) for d in dom.unique()]).mean(0);logits=g+(pi[:,:6,None]*(delta-mean)).sum(1);ce=F.cross_entropy(logits,y,reduction='none');guards=[];cohorts=[]
 for d in dom.unique():
  for label in y.unique():
   mask=(dom==d)&(y==label)&correct;n=int(mask.sum())
   if n<2:continue
   grad=vector(ce[mask].mean(),params).cpu()
   if grad.norm()>0:guards.append(grad);cohorts.append(dict(domain=int(d),label=int(label),n=n))
 original=vector(chosen,params).cpu()
 if original.norm()==0:return chosen,dict(cohorts=cohorts,zero_discovery=True)
 protected,certificate=project(original,guards)
 # Compare direction only: restore the original discovery gradient norm.
 protected*=original.norm()/protected.norm();change=protected-original;offset=0;surrogate=chosen
 for parameter in params:
  n=parameter.numel();adjustment=change[offset:offset+n].reshape(parameter.shape).to(parameter);surrogate=surrogate+((parameter-parameter.detach())*adjustment).sum();offset+=n
 actual=vector(surrogate,params).cpu();assert torch.allclose(actual,protected,atol=1e-7,rtol=2e-4);assert torch.equal(surrogate.detach(),chosen.detach())
 if guards:
  dots=torch.stack([a/a.norm() for a in guards])@(actual/actual.norm());assert float(dots.min())>=-2e-6
 return surrogate,dict(cohorts=cohorts,projection=certificate,gradient_norm_before=float(original.norm()),gradient_norm_after=float(actual.norm()),gradient_max_replay_error=float((actual-protected).abs().max()),scalar_preserved=True,routing_detached=True,source_only=True,target_read=False,cache_sha256=SHA,scope='Discovery residual gradient only; classification/router/balance/SIRM unaffected. Equal-radius projection, group mean source-in CE first-order constraints, recalculated every update.')
