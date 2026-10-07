"""Exact teacher values with analytical entmax Jacobian for diagnostic guards."""
import torch
from experiments.sirm_discover.math import entmax15
class Entmax15Jacobian(torch.autograd.Function):
 @staticmethod
 def forward(ctx,scores):
  probabilities=entmax15(scores);ctx.save_for_backward(probabilities);return probabilities
 @staticmethod
 def backward(ctx,grad):
  p,=ctx.saved_tensors;root=p.sqrt();weighted=(grad*root).sum(-1,keepdim=True)/root.sum(-1,keepdim=True)
  return root*(grad-weighted)

def differentiable_route(g,delta,posterior):
 u=((g[:,None,:]+delta).log_softmax(-1)-g.log_softmax(-1)[:,None,:]).mul(posterior[:,None,:]).sum(-1)
 probabilities=Entmax15Jacobian.apply(torch.cat([u/.1,torch.zeros_like(u[:,:1])],1));evidence=((u-.002)/.1).clamp(0,1);residual=probabilities[:,:6]*evidence
 return torch.cat([residual,(1-residual.sum(1,keepdim=True)).clamp(0,1)],1)
