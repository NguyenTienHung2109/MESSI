"""Source-in-only curvature and virtual-update diagnostic; no fitting or target reads."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
OUT=DEST/'source_meta_update_sensitivity'

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();parent=DEST/'source_meta_domain_readout';r=json.loads((parent/'results.json').read_text());f=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/source_in_tokens.npz';assert digest(f)==r['input_sha256'][str(f)];data=dict(np.load(f));x=np.concatenate([data['cls'],data['patch_max']],1).astype(float);locks={str(f):digest(f),str(parent/'results.json'):digest(parent/'results.json')}
 plan=dict(source_only=True,target_read=False,source_out_read=False,fitting=False,bank_updates=False,states=['zero_correction','fitted_matched_control'],held_domains=[None,0,2,3],virtual_steps=['fixed0.1','stable_output_normalized'],normalization='Start at correction-logit RMS change0.1 on fitting images; deterministically halve step until adaptation CE+KL+L2 decreases. At most20 halvings. No held-domain labels enter normalization/acceptance; no source-out/target selection. Normalize RMS on adaptation domains with equal-domain averaging.',curvature='32 seeded power iterations on exact adaptation Hessian-vector products, plus final Rayleigh residual. Local curvature estimate only; backtracking supplies observed adaptation-risk decrease, not a global bound.',limits='Finite readout updates, not trained SIRM/factors. Held-domain episodes come from source-in only; backbone saw all source domains. Only alpha normalization changes; fixed baseline output, model map and L2.01 unchanged.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[]
 for hold in plan['held_domains']:
  rr=next(v for v in r['rows'] if v['held_domain']==hold and v['family']=='matched_double_risk');file=parent/rr['fit_file'];assert digest(file)==rr['fit_sha256'];locks[str(file)]=digest(file);m=dict(np.load(file));mask=np.ones(len(x),bool) if hold is None else data['domains']!=hold;domains=data['domains'][mask];ds=sorted(set(domains));z=torch.tensor((x[mask]-m['mean'])/m['std'],dtype=torch.float64);g=torch.tensor(data['global_logits'][mask],dtype=torch.float64);y=torch.tensor(data['y'][mask]);b=torch.tensor(m['basis'],dtype=torch.float64);pg=g.softmax(1);idx={d:torch.tensor(np.flatnonzero(domains==d)) for d in ds}
  def mean(v,used):return torch.stack([v[idx[d]].mean() for d in used]).mean()
  def risk(c,used,reg=False):
   logits=g+z@c@b.T;loss=F.cross_entropy(logits,y,reduction='none')+(pg*(pg.log()-logits.log_softmax(1))).sum(1);return mean(loss,used)+(.01/2*c.square().sum() if reg else 0)
  for state in plan['states']:
   original=np.zeros_like(m['coef']) if state=='zero_correction' else m['coef'];c=torch.tensor(original,dtype=torch.float64,requires_grad=True)
   for held in ds:
    used=[d for d in ds if d!=held];base=risk(c,used,True);grad=torch.autograd.grad(base,c,create_graph=True)[0];gh=torch.autograd.grad(risk(c,[held]),c,retain_graph=True)[0];cos=float((grad*gh).sum().detach()/(grad.norm()*gh.norm()).detach());norm=float(grad.norm().detach());direction=z@grad@b.T;rms=float(mean(direction.square().mean(1),used).sqrt().detach());gen=torch.Generator().manual_seed(0);v=torch.randn(c.shape,generator=gen,dtype=torch.float64);v=v/v.norm()
    for _ in range(32):
     hv=torch.autograd.grad((grad*v).sum(),c,retain_graph=True)[0];v=(hv/hv.norm()).detach()
    hv=torch.autograd.grad((grad*v).sum(),c,retain_graph=True)[0];eig=float((v*hv).sum().detach());resid=float((hv-eig*v).norm().detach());adaptive=.1/max(rms,1e-12);backtracks=0
    with torch.no_grad():
     while float(risk(c-adaptive*grad,used,True))>=float(base.detach()) and backtracks<20:adaptive/=2;backtracks+=1
     assert float(risk(c-adaptive*grad,used,True))<float(base.detach())
     for name,alpha in [('fixed0.1',.1),('stable_output_normalized',adaptive)]:
      cc=c-alpha*grad;full=g+z@cc@b.T;prior=g+z@c@b.T;changed=(full.argmax(1)!=prior.argmax(1));corr0=prior.argmax(1)==y;corr1=full.argmax(1)==y;heldk=idx[held]
      row=dict(excluded_fitting_domain=hold,state=state,episode_held_domain=int(held),step_kind=name,alpha=alpha,backtracks=backtracks if name!='fixed0.1' else None,adapt_gradient_norm=norm,adapt_held_gradient_cosine=cos,adapt_hessian_top_eigen_estimate=eig,hessian_eigen_residual=resid,alpha_times_local_curvature=alpha*eig,adapt_update_logit_RMS=alpha*rms,held_update_logit_RMS=float(direction[heldk].square().mean().sqrt())*alpha,adapt_regularized_risk_change=float(risk(cc,used,True)-base.detach()),held_risk_change=float(risk(cc,[held])-risk(c,[held])),held_changed_predictions=int(changed[heldk].sum()),held_repairs=int((~corr0[heldk]&corr1[heldk]).sum()),held_harms=int((corr0[heldk]&~corr1[heldk]).sum()));rows.append(row)
   print(hold,state,'done',flush=True)
 for f,h in locks.items():assert digest(Path(f))==h
 assert len(rows)==36
 write(OUT/'results.json',dict(plan=plan,rows=rows,input_sha256=locks,source_only=True,target_read=False,source_out_read=False,bank_updates=False))
if __name__=='__main__':run()
