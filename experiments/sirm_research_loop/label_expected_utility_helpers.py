"""Source-only label posterior and conditional expected standalone utility."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax,log_softmax
L2=.01
def fit_label(x,target,groups):
 weight=np.array([1/(len(set(groups))*sum(groups==d)) for d in groups]);mean=x.mean(0);std=x.std(0).clip(1e-8)
 z=torch.tensor((x-mean)/std,dtype=torch.float64);t=torch.tensor(target,dtype=torch.float64);w=torch.tensor(weight,dtype=torch.float64)
 coef=torch.zeros((x.shape[1],target.shape[1]),dtype=torch.float64,requires_grad=True);intercept=torch.zeros(target.shape[1],dtype=torch.float64,requires_grad=True)
 opt=torch.optim.LBFGS([coef,intercept],max_iter=500,tolerance_grad=1e-8,tolerance_change=1e-12,line_search_fn='strong_wolfe');calls=0
 def closure():
  nonlocal calls
  opt.zero_grad();loss=(-(t*F.log_softmax(z@coef+intercept,1)).sum(1)*w).sum()+L2/2*coef.square().sum();loss.backward();calls+=1;return loss
 opt.step(closure);loss=float(closure().detach());grad=max(float(coef.grad.abs().max()),float(intercept.grad.abs().max()));assert grad<1e-5,(grad,calls)
 return dict(mean=mean,std=std,coef=coef.detach().numpy(),intercept=intercept.detach().numpy(),objective=loss,gradient_max=grad,closure_calls=calls)
def predict_label(model,x):return softmax((x-model['mean'])/model['std']@model['coef']+model['intercept'],axis=1)

def expected_utility(g,delta,posterior):
    assert posterior.shape==g.shape and np.allclose(posterior.sum(1),1)
    gain=log_softmax(g[:,None]+delta,axis=2)-log_softmax(g,axis=1)[:,None]
    return np.einsum('nmy,ny->nm',gain,posterior)
def label_metrics(p,y,dom,ids,g):
    correct=p.argmax(1)==y;gw=g.argmax(1)!=y;ce=-np.log(p[np.arange(len(y)),y].clip(1e-300))
    return dict(n=len(y),equal_domain_accuracy=float(np.mean([correct[dom==d].mean() for d in sorted(set(dom))])),equal_domain_CE=float(np.mean([ce[dom==d].mean() for d in sorted(set(dom))])),repairs=int((gw&correct).sum()),harms=int((~gw&~correct).sum()),repair_ids=ids[gw&correct].tolist(),harm_ids=ids[~gw&~correct].tolist())
