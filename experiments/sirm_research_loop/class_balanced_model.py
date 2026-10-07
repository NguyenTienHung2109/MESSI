"""Equal-present-class discovery with original per-batch residual gradient norm."""
from unittest.mock import patch
import torch
from experiments.sirm_discover.math import expert_ce,multiple_choice_loss
from .class_coherent_discovery import class_coherent_cost
from .class_coherent_model import ClassCoherentGateSIRM
from .spatial_bank_model import SpatialBankGateSIRM

def equal_class_cost(losses,y):
 if losses.ndim!=2 or len(losses)!=len(y) or not len(y):raise ValueError('nonempty paired image/expert losses and labels required')
 return torch.stack([losses[y.eq(c)].mean(0).amin() for c in y.unique(sorted=True)]).mean()

def gradient_norm(loss,params):
 gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
 return torch.sqrt(sum(g.detach().double().square().sum() for g in gs if g is not None))

class ClassBalancedDiscovery:
 def update(self,minibatches):
  assert self.config.get('discovery_class_weighting')=='equal_class_original_gradient_norm'
  observed={};params=list(self.residuals.parameters())
  def loss(g,delta,y,temperature):
   per=expert_ce(g[:,None,:]+delta,y);original=class_coherent_cost(per,y);balanced=equal_class_cost(per,y)
   oldnorm=gradient_norm(original,params);newnorm=gradient_norm(balanced,params)
   fallback=bool(newnorm==0 and oldnorm>0)
   scale=(oldnorm/newnorm).detach().to(balanced.dtype) if newnorm>0 else balanced.new_tensor(0.)
   chosen=original if fallback else balanced*scale
   matched=gradient_norm(chosen,params)
   assert torch.allclose(matched,oldnorm,rtol=2e-5,atol=1e-9)
   _,balance=multiple_choice_loss(g,delta,y,temperature)
   observed.update(scale=float(scale),fallback=fallback,original_loss=float(original.detach()),equal_class_loss=float(balanced.detach()),original_gradient_norm=float(oldnorm),equal_class_gradient_norm=float(newnorm),matched_gradient_norm=float(matched))
   return chosen,balance
  def replay(per,y):
   assert observed
   return class_coherent_cost(per,y) if observed['fallback'] else equal_class_cost(per,y)*observed['scale']
  with patch('experiments.sirm_research_loop.class_coherent_model.class_coherent_loss',loss),patch('experiments.sirm_research_loop.class_coherent_model.class_coherent_cost',replay):
   row=super().update(minibatches)
  self.last_diagnostics.update(discovery_class_weighting='equal_class_original_gradient_norm',discovery_outer_weights='equal over labels present; within-class expert winner unchanged',discovery_original_class_loss=observed['original_loss'],discovery_equal_class_unscaled_loss=observed['equal_class_loss'],discovery_gradient_norm_match=observed)
  return row

class ClassBalancedGateSIRM(ClassBalancedDiscovery,ClassCoherentGateSIRM):pass
class ClassBalancedSpatialGateSIRM(ClassBalancedDiscovery,SpatialBankGateSIRM):pass
