"""Shared confidence router; frozen source-only units, detached descriptors."""
import math
from pathlib import Path
import torch
from torch import nn
from experiments.sirm_discover.math import center_logits
from .direct_router_risk_model import DirectRouterRiskSpatialGateSIRM
from .source_direct_mixture_reliability_router import descriptor,FEATURES

class SharedConfidenceRouter(nn.Module):
 def __init__(self,owner,scale):
  super().__init__();object.__setattr__(self,'owner',owner)
  self.head=nn.Linear(11,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
  self.register_buffer('global_logit_unit',scale.detach().float().clamp_min(1e-8))
 def forward(self,z):
  m=self.owner
  with torch.no_grad():
   raw=z*m.bank_source_feature_std*math.sqrt(2)+m.bank_source_feature_mean
   g=m.global_head(raw[:,:m.original_feature_dim])
   d=m.center_delta(center_logits(torch.stack([e(z.detach()) for e in m.residuals],1)))
   x=descriptor(g,d,d)[:,:,:11].to(z.dtype)
   # Entropies in log(C) units; margin and residual units fixed by anchor.
   x[:,:,[1,4]]/=math.log(g.shape[1])
   x[:,:,[2,5,7,9,10]]/=self.global_logit_unit
   assert torch.isfinite(x).all()
  score=self.head(x).squeeze(-1)
  return torch.cat([score,score.new_zeros(len(score),1)],1)

class ConfidenceRouterSpatialGateSIRM(DirectRouterRiskSpatialGateSIRM):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  cache=torch.load(Path(self.config['center_feature_cache']),map_location='cpu',weights_only=False)
  anchor=torch.load(Path(self.config['strong_anchor_checkpoint']),map_location='cpu',weights_only=False)
  state=anchor['model_dict'];g=torch.nn.functional.linear(cache['features'][:,:384],state['global_head.weight'],state['global_head.bias'])
  scale=center_logits(g).square().mean().sqrt()
  self.router=SharedConfidenceRouter(self,scale)
  self.configure_stage();self.optimizer=torch.optim.Adam(self.parameters(),lr=self.config['lr'],weight_decay=self.config['weight_decay'])
 def update(self,minibatches):
  row=super().update(minibatches)
  self.last_diagnostics.update(router_architecture='shared11_confidence_linear_null0',router_features=FEATURES[:11],router_descriptors_detached=True,router_normalization='fixed frozen-anchor source-in logitRMS; entropy/log10; no adaptive statistics',router_global_logit_unit=float(self.router.global_logit_unit))
  return row
