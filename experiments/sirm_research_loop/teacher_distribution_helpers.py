"""Fixed-regularization linear soft teacher fitting and source-only evaluation."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp,softmax
from experiments.sirm_discover.math import router_teacher,residual_utilities,soft_null_q
L2=.01

def teacher(u):return router_teacher(torch.tensor(u),.1,.002).numpy()
def fit_student(x,target,groups):
 weight=np.array([1/(len(set(groups))*sum(groups==d)) for d in groups]);mean=x.mean(0);std=x.std(0).clip(1e-8)
 z=torch.tensor((x-mean)/std,dtype=torch.float64);t=torch.tensor(target,dtype=torch.float64);w=torch.tensor(weight,dtype=torch.float64)
 coef=torch.zeros((x.shape[1],7),dtype=torch.float64,requires_grad=True);intercept=torch.zeros(7,dtype=torch.float64,requires_grad=True)
 opt=torch.optim.LBFGS([coef,intercept],max_iter=500,tolerance_grad=1e-8,tolerance_change=1e-12,line_search_fn='strong_wolfe');calls=0
 def closure():
  nonlocal calls
  opt.zero_grad();loss=(-(t*F.log_softmax(z@coef+intercept,1)).sum(1)*w).sum()+L2/2*coef.square().sum();loss.backward();calls+=1;return loss
 opt.step(closure);loss=float(closure().detach());grad=max(float(coef.grad.abs().max()),float(intercept.grad.abs().max()));assert grad<1e-5,(grad,calls)
 return dict(mean=mean,std=std,coef=coef.detach().numpy(),intercept=intercept.detach().numpy(),objective=loss,gradient_max=grad,closure_calls=calls)
def predict_pi(model,x):return softmax((x-model['mean'])/model['std']@model['coef']+model['intercept'],axis=1)
def evaluate_pi(g,delta,y,dom,ids,pi,true_u):
 assert np.allclose(pi.sum(1),1) and (pi>=0).all()
 full=g+np.einsum('nm,nmj->nj',pi[:,:6],delta);correct=full.argmax(1)==y;gw=g.argmax(1)!=y;ce=logsumexp(full,1)-full[np.arange(len(y)),y];t=teacher(true_u)
 kl=np.sum(np.where(t>0,t*(np.log(t.clip(1e-300))-np.log(pi.clip(1e-300))),0),1)
 return dict(n=len(y),equal_domain_accuracy=float(np.mean([correct[dom==d].mean() for d in sorted(set(dom))])),equal_domain_CE=float(np.mean([ce[dom==d].mean() for d in sorted(set(dom))])),repairs=int((gw&correct).sum()),harms=int((~gw&~correct).sum()),bird_correct=int((correct&(y==0)).sum()),NULL=float(pi[:,-1].mean()),teacher_KL=float(np.mean([kl[dom==d].mean() for d in sorted(set(dom))])),teacher_pi_L1=float(np.abs(pi-t).sum(1).mean()),harmful_standalone_mass=float((pi[:,:6]*(true_u<0)).sum(1).mean()),repair_ids=ids[gw&correct].tolist(),harm_ids=ids[~gw&~correct].tolist())
def gates_pi(g,delta,y,dom,pi):
 _,_,u,_=residual_utilities(torch.tensor(g),torch.tensor(delta),torch.tensor(pi),torch.tensor(y));U=torch.stack([u[dom==d].mean(0) for d in [0,2,3]]);q,_,_,v=soft_null_q(U,torch.ones_like(U,dtype=torch.bool),.05,.002,.002)
 return dict(U=U.tolist(),V=v.tolist(),Q=q.tolist(),shared_experts=torch.nonzero((q>0).sum(0)>=2).flatten().tolist())
