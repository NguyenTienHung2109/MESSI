"""Source-in learned interpolation of locked global/auxiliary posterior utility."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax,logsumexp
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
from .label_expected_utility_helpers import predict_label,expected_utility
from .teacher_distribution_helpers import teacher,gates_pi
from .source_residual_rank_diagnosis import summarize
OUT=DEST/'source_global_anchored_utility'

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();source={};locks={};rank=json.loads((DEST/'source_residual_rank_diagnosis/results.json').read_text())
 for split in ['source_in','source_out']:
  f=ROOT/f'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/{split}_tokens.npz';assert digest(f)==rank['input_sha256'][str(f)];source[split]=dict(np.load(f));locks[str(f)]=digest(f)
 plan=dict(source_only=True,target_read=False,primary_training=False,bank_updates=False,posterior_refitting=False,fit='One alpha in[0,1] per state/input/calibration hold minimizes equal-domain original source-in full composed CE. Bounded scalar optimization xatol1e-6/max100 plus both endpoints; no validation or target selection. Nonconvex objective; no guaranteed global optimality claimed.',utility='U_alpha=(1-alpha)*E_global_posterior[CE gain]+alpha*E_aux_posterior[CE gain]. Linear inposterior; existing .1/.002 teacher, evidence and Q thresholds unchanged. alpha0 is algebraic NULL control; alpha1 original teacher.',cohort='Original posterior only; inputs CLS+patchmax and global/expert logits/pi. Full fitting and excluded calibration-domain controls. Source-selected banks fixed.',limits='Readout calibration is not Stage2 training or SIRM success. Holds exclude posterior/scalar fitting, not bank learning/centering. Source-out previously inspected.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[];cases=[]
 for state in rank['states']:
  bankpath=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';assert digest(bankpath)==rank['input_sha256'][str(bankpath)];bank=dict(np.load(bankpath));locks[str(bankpath)]=digest(bankpath)
  for held in [None,0,2,3]:
   parent=DEST/('source_label_expected_utility' if held is None else 'source_label_expected_utility_holds');fitresult=json.loads((parent/'results.json').read_text())
   for kind in ['CLS_plus_patch_max','global_and_expert_logits_and_pi']:
    rec=next(r for r in fitresult['states'][state] if r['input']==kind and r['cohort']=='original' and r.get('held_calibration_domain')==held);fitfile=parent/rec['fit_file'];assert digest(fitfile)==rec['fit_sha256'];model=dict(np.load(fitfile));locks[str(fitfile)]=digest(fitfile);parts={}
    for split,b in source.items():
     assert np.array_equal(b['ids'],bank[split+'_ids']);keep=np.ones(len(b['y']),bool) if held is None else (b['domains']!=held if split=='source_in' else b['domains']==held);g=b['global_logits'][keep].astype(float);d=bank[split+'_bank_delta'][keep].astype(float);pi=bank[split+'_bank_pi'][keep].astype(float);x=np.concatenate([b['cls'][keep],b['patch_max'][keep]],1).astype(float) if kind=='CLS_plus_patch_max' else np.concatenate([g-g.mean(1,keepdims=True),d.reshape(len(g),-1),pi],1);post=predict_label(model,x);gu=expected_utility(g,d,softmax(g,1));pu=expected_utility(g,d,post);assert gu.max()<1e-9 and np.allclose(teacher(gu)[:,-1],1)
     parts[split]=dict(g=g,d=d,y=b['y'][keep],dom=b['domains'][keep],ids=b['ids'][keep],global_utility=gu,posterior_utility=pu)
    t=parts['source_in'];weights=np.array([1/(len(set(t['dom']))*sum(t['dom']==domain)) for domain in t['dom']]);calls=[]
    def objective(alpha):
     pi=teacher((1-alpha)*t['global_utility']+alpha*t['posterior_utility']);f=t['g']+np.einsum('nm,nmc->nc',pi[:,:6],t['d']);loss=float(weights@(logsumexp(f,axis=1)-f[np.arange(len(f)),t['y']]));calls.append(dict(alpha=float(alpha),CE=loss));return loss
    result=minimize_scalar(objective,bounds=(0.,1.),method='bounded',options={'xatol':1e-6,'maxiter':100});assert result.success
    candidates=[(objective(0.),0.),(objective(1.),1.),(float(result.fun),float(result.x))];loss,alpha=min(candidates,key=lambda z:(z[0],z[1]));record=dict(state=state,input=kind,held_calibration_domain=held,alpha=alpha,source_in_objective=loss,optimization_success=bool(result.success),optimization_calls=calls,posterior_sha256=digest(fitfile))
    for split,b in parts.items():
     metrics={}
     for policy,aa in [('global_control',0.),('original_teacher',1.),('source_in_fitted',alpha)]:
      pi=teacher((1-aa)*b['global_utility']+aa*b['posterior_utility']);metrics[policy]=summarize(b['g'],b['d'],pi,b['y'],b['dom'],b['ids']);f=b['g']+np.einsum('nm,nmc->nc',pi[:,:6],b['d']);pred=f.argmax(1)
      for i,id in enumerate(b['ids']):cases.append(dict(state=state,input=kind,held_calibration_domain=held,split=split,policy=policy,alpha=aa,image_id=str(id),label=int(b['y'][i]),prediction=int(pred[i]),NULL=float(pi[i,-1])))
      if split=='source_out' and policy=='source_in_fitted' and held is None:record['Q']=gates_pi(b['g'],b['d'],b['y'],b['dom'],pi)
     record[split]=metrics
    rows.append(record);print(state,held,kind,'alpha',alpha,flush=True)
 for path,h in locks.items():assert digest(Path(path))==h
 assert len(rows)==16
 write(OUT/'results.json',dict(plan=plan,rows=rows,source_only=True,target_read=False,bank_updates=False,input_sha256=locks));write(OUT/'cases.json',cases)
if __name__=='__main__':run()
