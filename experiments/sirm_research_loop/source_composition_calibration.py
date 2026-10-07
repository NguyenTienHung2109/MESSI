"""Source-only convex bounded composition controls; no bank or target update."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .routed_discovery_run import DEST,ROOT
from .shrinkage import digest,write
OUT=DEST/'source_composition_calibration'
STATES=['unscaled_OFF_selected','unscaled_OFF_last','routed_OFF_selected','routed_OFF_last']

def fit(g,t,y,d,mode):
 weight=np.array([1/(len(set(d))*sum(d==x)) for x in d]);n=1 if mode=='uniform' else 6
 def calc(v):
  w=np.repeat(v,6) if n==1 else v;l=g+np.einsum('nmj,m->nj',t,w);loss=logsumexp(l,axis=1)-l[np.arange(len(y)),y];prob=np.exp(l-logsumexp(l,axis=1)[:,None]);prob[np.arange(len(y)),y]-=1;grad=np.einsum('n,nj,nmj->m',weight,prob,t)
  return float(weight@loss),np.array([grad.sum()]) if n==1 else grad
 r=minimize(calc,np.ones(n),jac=True,method='L-BFGS-B',bounds=[(0,1)]*n,options=dict(maxiter=1000,ftol=1e-12,gtol=1e-8));value,grad=calc(r.x);projected=r.x-np.clip(r.x-grad,0,1);assert r.success and np.max(np.abs(projected))<1e-6,(r.message,projected)
 return dict(weights=(np.repeat(r.x,6) if n==1 else r.x).tolist(),training_CE=value,converged=bool(r.success),iterations=int(r.nit),projected_gradient_max=float(np.max(np.abs(projected))))

def scores(g,t,y,d,w,ids):
 l=g+np.einsum('nmj,m->nj',t,w);pred=l.argmax(1);gp=g.argmax(1);ce=logsumexp(l,axis=1)-l[np.arange(len(y)),y];wrong=gp!=y;right=pred==y
 return dict(n=len(y),equal_domain_accuracy=float(np.mean([right[d==x].mean() for x in sorted(set(d))])),equal_domain_CE=float(np.mean([ce[d==x].mean() for x in sorted(set(d))])),repairs=int((wrong&right).sum()),harms=int((~wrong&~right).sum()),repair_ids=[str(x) for x in ids[wrong&right]],harm_ids=[str(x) for x in ids[~wrong&~right]],bird_correct=int((right&(y==0)).sum()))

def run():
 OUT.mkdir(exist_ok=True);plan=dict(source_only=True,target_read=False,primary_training=False,checkpoint_selection_changed=False,states=STATES,controls=['uniform_scale','six_expert_scales'],bounds=[0,1],objective='Equal-domain source-in original-view CE on frozen logits and routed components, no regularization or hyperparameter grid.',solver='L-BFGS-B, maxiter1000, ftol1e-12, gtol1e-8; projected gradient must be <1e-6.',fitting_scopes=['all_source_in','other_two_source_in_for_each_held_domain'],evaluation='source-out only; all/fixed validation and calibration-domain holds. Bank/backbone previously saw all training domains; no claim of independent domain generalization. Source-out previously selected/inspected.',interpretation='Compare selective component suppression with uniform shrinkage; this is a fitted static composition diagnostic, not SIRM or deployable label-conditioned gating. Bounds prohibit amplifying components. No variant selected by target.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());results={};locks={};token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';audit=json.loads((token/'extraction_audit.json').read_text())
 for state in STATES:
  path=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';a=np.load(path);locks[state]=digest(path);data={}
  for split in ['source_in','source_out']:
   cpath=token/f'{split}_tokens.npz';assert digest(cpath)==next(x['cache_sha256'] for x in audit['rows'] if x['split']==split);b=np.load(cpath);assert np.array_equal(a[split+'_ids'],b['ids']) and np.array_equal(a[split+'_y'],b['y']);ids=a[split+'_ids'];y=a[split+'_y'];dom=a[split+'_domains'];g=b['global_logits'].astype(float);t=a[split+'_bank_delta'].astype(float)*a[split+'_bank_pi'][:,:6,None].astype(float);assert np.array_equal((g+t.sum(1)).argmax(1),a[split+'_bank_full'].argmax(1));data[split]=(g,t,y,dom,ids)
  gi,ti,yi,di,ii=data['source_in'];g,t,y,d,ids=data['source_out'];assert not set(ii)&set(ids);records=[]
  for held in [None,0,2,3]:
   tr=np.ones(len(yi),bool) if held is None else di!=held;va=np.ones(len(y),bool) if held is None else d==held
   for mode in ['uniform','selective']:
    f=fit(gi[tr],ti[tr],yi[tr],di[tr],mode);records.append(dict(held_calibration_domain=held,mode=mode,fit=f,source_out=scores(g[va],t[va],y[va],d[va],f['weights'],ids[va]),native=scores(g[va],t[va],y[va],d[va],np.ones(6),ids[va]),global_only=scores(g[va],t[va],y[va],d[va],np.zeros(6),ids[va])))
  results[state]=records;assert digest(path)==locks[state]
 write(OUT/'results.json',dict(plan=plan,cache_sha256=locks,states=results,fitting=True,primary_training=False,target_read=False,bank_parameters_changed=False,scope='Source-in fitted bounded scalars only; no optimizer step on any bank. True cached frozen-global logits, original source-in/out and exact identity pairing.'));write(OUT/'completion_audit.json',dict(completed=True,target_read=False,converged_fits=32,bank_updates=False,code_sha256=digest(Path(__file__))))
if __name__=='__main__':run()
