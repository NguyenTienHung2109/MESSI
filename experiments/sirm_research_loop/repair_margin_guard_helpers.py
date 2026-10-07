"""Project a residual gradient onto source-only repair-margin halfspaces."""
import numpy as np
import torch
from scipy.optimize import minimize

def project(gradient,guards):
 if not guards:
  return gradient.clone(),dict(cohorts=0,changed=False,retained_norm_fraction=1.)
 A=torch.stack([g/g.norm() for g in guards]);norm=gradient.norm();v=gradient/norm
 gram=(A@A.T).numpy();linear=(A@v).numpy()
 def objective(lam):return float(.5*lam@gram@lam+linear@lam),gram@lam+linear
 result=minimize(objective,np.zeros(len(guards)),jac=True,method='L-BFGS-B',bounds=[(0,None)]*len(guards),options=dict(maxiter=1000,ftol=1e-14,gtol=1e-10))
 _,grad=objective(result.x);pg=result.x-np.maximum(result.x-grad,0)
 # Convex dual is certified by KKT residuals even if line search stops at machine precision.
 assert np.abs(pg).max()<1e-7,(result.message,pg)
 projected=v+A.T@torch.tensor(result.x)
 dots=A@projected;assert float(dots.min())>=-1e-7
 assert projected.norm()>1e-10
 return projected*norm,dict(cohorts=len(guards),changed=bool((linear<0).any()),retained_norm_fraction=float(projected.norm()),pre_constraint_dots=linear.tolist(),post_constraint_dots=dots.tolist(),dual_weights=result.x.tolist(),projected_gradient_max=float(np.abs(pg).max()),solver_success=bool(result.success),solver_message=str(result.message))
