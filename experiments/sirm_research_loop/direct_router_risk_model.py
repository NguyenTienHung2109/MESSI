"""Actual-mixture CE auxiliary routing; existing bank and guard unchanged."""
from unittest.mock import patch
import torch
import torch.nn.functional as F
from .correct_guard_model import CorrectGuardSpatialGateSIRM

class DirectRouterRiskSpatialGateSIRM(CorrectGuardSpatialGateSIRM):
 def update(self,minibatches):
  y=torch.cat([b for _,b in minibatches]);observed={};calls=0
  def actual_risk(log_pi,teacher,**kwargs):
   nonlocal calls
   calls+=1;assert calls==1 and kwargs=={'reduction':'batchmean'}
   f=self._cap_forward;g=f['global_logits'].detach();delta=f['delta'].detach();pi=log_pi.exp()
   mixture=g+(pi[:,:self.num_experts,None]*delta).sum(1)
   loss=F.cross_entropy(mixture,y)
   residual_grads=torch.autograd.grad(loss,list(self.residuals.parameters()),retain_graph=True,allow_unused=True)
   assert all(v is None for v in residual_grads)
   assert torch.allclose(pi,f['pi'],atol=1e-6,rtol=1e-5)
   observed.update(router_objective='actual_joint_mixture_CE_detached_experts',router_objective_raw_CE=float(loss.detach()),router_objective_teacher_ignored=True,router_objective_residual_gradient_absent=True,router_probability_replay_max_error=float((pi-f['pi']).abs().max()))
   return loss
  # The inherited cap captures this function and still caps only auxiliary
  # router gradients against classification; all other losses stay unchanged.
  with patch('experiments.sirm_discover.model.F.kl_div',actual_risk):row=super().update(minibatches)
  assert calls==1;self.last_diagnostics.update(observed);return row
