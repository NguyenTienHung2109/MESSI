"""Source-calibrated confidence router and explicit router optimizer scale."""
import torch
from .confidence_router_model import ConfidenceRouterSpatialGateSIRM,SharedConfidenceRouter

class ConditionedConfidenceRouter(SharedConfidenceRouter):
 def __init__(self,owner,scale):
  super().__init__(owner,scale)
  self.register_buffer('descriptor_mean',torch.zeros(11))
  self.register_buffer('descriptor_std',torch.ones(11))
  self.register_buffer('descriptor_calibration_step',torch.tensor(-1,dtype=torch.long))
 def features(self,z):
  import math
  from experiments.sirm_discover.math import center_logits
  from .source_direct_mixture_reliability_router import descriptor
  m=self.owner
  with torch.no_grad():
   raw=z*m.bank_source_feature_std*math.sqrt(2)+m.bank_source_feature_mean
   g=m.global_head(raw[:,:m.original_feature_dim]);d=m.center_delta(center_logits(torch.stack([e(z.detach()) for e in m.residuals],1)))
   x=descriptor(g,d,d)[:,:,:11].to(z.dtype);x[:,:,[1,4]]/=math.log(g.shape[1]);x[:,:,[2,5,7,9,10]]/=self.global_logit_unit
  return x
 def forward(self,z):
  x=self.features(z);x=(x-self.descriptor_mean)/self.descriptor_std
  score=self.head(x).squeeze(-1)
  return torch.cat([score,score.new_zeros(len(score),1)],1)

class ConditionedRouterSpatialGateSIRM(ConfidenceRouterSpatialGateSIRM):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.router=ConditionedConfidenceRouter(self,self.router.global_logit_unit)
  import pathlib
  cache=torch.load(pathlib.Path(self.config['center_feature_cache']),map_location='cpu',weights_only=False)
  self._descriptor_source_cache=(cache['features'],cache['domains'])
  router=list(self.router.parameters());ids={id(p) for p in router};other=[p for p in self.parameters() if id(p) not in ids]
  self.optimizer=torch.optim.Adam([{'params':other,'lr':self.config['lr']},{'params':router,'lr':self.config['router_optimizer_lr']}],weight_decay=self.config['weight_decay'])
  self.calibrate_features(*self._descriptor_source_cache)
 @torch.no_grad()
 def calibrate_features(self,features,domains):
  super().calibrate_features(features,domains)
  if not isinstance(self.router,ConditionedConfidenceRouter):return
  # Use source-calibrated centering, not batch-dependent centering for statistics.
  old_mode=self.training;self.training=False
  try:
   xs=[self.router.features(self.bank_features(batch.to(self.expert_source_mean.device))).double() for batch in features.split(128)]
   x=torch.cat(xs);mean=x.mean((0,1));std=x.std((0,1),unbiased=False).clamp_min(self.config['router_descriptor_std_floor'])
   self.router.descriptor_mean.copy_(mean);self.router.descriptor_std.copy_(std);self.router.descriptor_calibration_step.copy_(self.step)
  finally:self.training=old_mode
 def update(self,minibatches):
  if int(self.step)%self.config['utility_interval']==0:self.calibrate_features(*self._descriptor_source_cache)
  row=super().update(minibatches)
  self.last_diagnostics.update(router_normalization='source-in current-expert pooled mean/std of anchor-unit descriptors; detached refresh5 and evaluation calibration; std floor0.01',router_optimizer_lr=self.config['router_optimizer_lr'],router_descriptor_calibration_step=int(self.router.descriptor_calibration_step),router_descriptor_std=self.router.descriptor_std.tolist(),router_head_weight_norm=float(self.router.head.weight.detach().norm()))
  return row
