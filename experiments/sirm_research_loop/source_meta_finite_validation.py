"""Independent source-out replay of source-in-normalized finite domain updates."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
from .source_structured_contrast_readout import score
OUT=DEST/'source_meta_finite_validation'

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();parent=DEST/'source_meta_domain_readout';ref=json.loads((parent/'results.json').read_text());sensitivity=DEST/'source_meta_update_sensitivity/results.json';prior=json.loads(sensitivity.read_text());locks={str(parent/'results.json'):digest(parent/'results.json'),str(sensitivity):digest(sensitivity)};source={}
 for split in ['source_in','source_out']:
  f=ROOT/f'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/{split}_tokens.npz';assert digest(f)==ref['input_sha256'][str(f)];locks[str(f)]=digest(f);source[split]=dict(np.load(f))
 plan=dict(source_only=True,target_read=False,fitting=False,bank_updates=False,source_out_read=True,normalization='Replay the fixed0.1 and source-in-only RMS0.1/backtracked steps already recorded in sensitivity analysis. Source-out never sets direction or step. All-source update added as a control; its direction uses equal-domain source-in risk and normalization/backtracking uses those fitting images only.',states=prior['plan']['states'],held_domains=[None,0,2,3],limits='Disposable readout perturbations, not400-step SIRM. Source-out is already inspected. Domain-held fitting excludes readout only, not backbone. Repeated cases are not independent replications. Step acceptance establishes adaptation-risk decrease, not global numerical stability.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[];cases=[]
 for hold in plan['held_domains']:
  rr=next(r for r in ref['rows'] if r['held_domain']==hold and r['family']=='matched_double_risk');file=parent/rr['fit_file'];assert digest(file)==rr['fit_sha256'];locks[str(file)]=digest(file);m=dict(np.load(file));si=source['source_in'];mask=np.ones(len(si['y']),bool) if hold is None else si['domains']!=hold;dom=si['domains'][mask];ds=sorted(set(dom));xx=np.concatenate([si['cls'],si['patch_max']],1).astype(float);z=torch.tensor((xx[mask]-m['mean'])/m['std'],dtype=torch.float64);g=torch.tensor(si['global_logits'][mask],dtype=torch.float64);y=torch.tensor(si['y'][mask]);b=torch.tensor(m['basis'],dtype=torch.float64);pg=g.softmax(1);ix={d:torch.tensor(np.flatnonzero(dom==d)) for d in ds}
  def risk(c,used):
   logits=g+z@c@b.T;loss=F.cross_entropy(logits,y,reduction='none')+(pg*(pg.log()-logits.log_softmax(1))).sum(1);return torch.stack([loss[ix[d]].mean() for d in used]).mean()+.01/2*c.square().sum()
  for state in plan['states']:
   original=np.zeros_like(m['coef']) if state=='zero_correction' else m['coef'];c=torch.tensor(original,dtype=torch.float64,requires_grad=True)
   for episode in [None]+[int(d) for d in ds]:
    used=ds if episode is None else [d for d in ds if d!=episode];base=risk(c,used);grad=torch.autograd.grad(base,c)[0];direction=z@grad@b.T;rms=float(torch.stack([direction[ix[d]].square().mean() for d in used]).mean().sqrt().detach());alpha=.1/max(rms,1e-12);halves=0
    with torch.no_grad():
     while float(risk(c-alpha*grad,used))>=float(base.detach()) and halves<20:alpha/=2;halves+=1
     assert float(risk(c-alpha*grad,used))<float(base.detach())
    for kind,step in [('fixed0.1',.1),('stable_output_normalized',alpha)]:
     if episode is not None:
      check=next(v for v in prior['rows'] if v['excluded_fitting_domain']==hold and v['state']==state and v['episode_held_domain']==episode and v['step_kind']==kind);assert abs(check['alpha']-step)<1e-10
     cc=(c-step*grad).detach().numpy();so=source['source_out'];ev=np.ones(len(so['y']),bool) if hold is None else so['domains']==hold
     if hold is None and episode is not None:ev=so['domains']==episode
     vx=np.concatenate([so['cls'],so['patch_max']],1).astype(float);vz=(vx[ev]-m['mean'])/m['std'];vg=so['global_logits'][ev].astype(float);corr=vz@cc@m['basis'].T;before=vz@original@m['basis'].T;metric,pred=score(vg,corr,so['y'][ev],so['domains'][ev],so['ids'][ev]);control,p0=score(vg,before,so['y'][ev],so['domains'][ev],so['ids'][ev]);valid_y=so['y'][ev];row=dict(excluded_fitting_domain=hold,state=state,episode_held_domain=episode,step_kind=kind,alpha=step,backtracks=halves if kind!='fixed0.1' else None,source_in_step_logit_RMS=step*rms,source_out_step_logit_RMS=float(np.sqrt(((corr-before)**2).mean())),source_out=metric,before_update=control,accuracy_change_pp=100*(metric['equal_domain_accuracy']-control['equal_domain_accuracy']),CE_change=metric['equal_domain_CE']-control['equal_domain_CE'],repairs_vs_before=int(((p0!=valid_y)&(pred==valid_y)).sum()),harms_vs_before=int(((p0==valid_y)&(pred!=valid_y)).sum()));rows.append(row)
     for i,id in enumerate(so['ids'][ev]):
      if pred[i]!=p0[i]:cases.append(dict(excluded_fitting_domain=hold,state=state,episode_held_domain=episode,step_kind=kind,image_id=str(id),label=int(valid_y[i]),before_prediction=int(p0[i]),after_prediction=int(pred[i])))
   print(hold,state,'done',flush=True)
 for f,h in locks.items():assert digest(Path(f))==h
 assert len(rows)==52
 write(OUT/'results.json',dict(plan=plan,rows=rows,input_sha256=locks,source_only=True,target_read=False,bank_updates=False));write(OUT/'cases.json',cases)
if __name__=='__main__':run()
