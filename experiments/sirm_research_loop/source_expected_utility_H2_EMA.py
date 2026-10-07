"""Fixed selected-bank H2 replay for expected-utility routing and Q/ESS."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.sirm_discover.math import center_logits,residual_utilities,soft_null_q,component_eligibility
from .routed_discovery_run import ROOT,DEST,PRIOR
from .routed_conditional_utility import load
from .source_accessibility_H2 import inputs
from .label_expected_utility_helpers import predict_label,expected_utility
from .teacher_distribution_helpers import teacher
from .shrinkage import digest,write
OUT=DEST/'source_expected_utility_controls'

def run():
 torch.set_num_threads(4);parent=DEST/'source_label_expected_utility';pr=json.loads((parent/'results.json').read_text());proof=json.loads((DEST/'source_conditional_H2/extraction_audit.json').read_text());si=dict(np.load(ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/source_in_tokens.npz'))
 rows=[];locks={}
 for state,fits in pr['states'].items():
  if not state.endswith('selected'):continue
  m,p,path=load(PRIOR if state.startswith('unscaled') else DEST,'selected');assert digest(path)==pr['checkpoint_sha256'][state];locks[state]=digest(path)
  m.requires_grad_(False);before={k:v.clone() for k,v in m.state_dict().items()};raw=torch.tensor(np.concatenate([si['cls'],si['patch_max']],1));m.calibrate_features(raw,torch.tensor(si['domains']))
  models=[]
  for rec in fits:
   file=parent/rec['fit_file'];assert digest(file)==rec['fit_sha256'];models.append((rec,dict(np.load(file)),None))
  with torch.no_grad():
   for item in proof['rows']:
    f=DEST/'source_conditional_H2'/f"H2_step{item['step']}.pt";assert digest(f)==item['sha256'];b=torch.load(f,map_location='cpu',weights_only=False);raw=b['raw'];z=m.bank_features(raw);g=m.global_head(raw[:,:384]).double();oldpi=m.router(z).softmax(1).double();delta=center_logits(torch.stack([e(z) for e in m.residuals],1));delta=(delta-delta.mean(0,keepdim=True)).double();y=b['y'];dom=torch.tensor([0 if int(d)==0 else 1 if int(d)==2 else 2 for d in b['domains']]);assert all(int((dom==d).sum())==32 for d in range(3))
    nextmodels=[]
    for rec,model,ema in models:
     posterior=predict_label(model,inputs(rec['input'],raw,g,delta,oldpi));pi=torch.tensor(teacher(expected_utility(g.numpy(),delta.numpy(),posterior)))
     _,_,u,_=residual_utilities(g,delta,pi,y);U=torch.stack([u[dom==d].mean(0) for d in range(3)]);ema=U.clone() if ema is None else .9*ema+.1*U
     q,_,_,v=soft_null_q(ema,torch.ones_like(ema,dtype=torch.bool),.05,.002,.002);valid,ess,pairs=component_eligibility(pi,dom,q,2.)
     rows.append(dict(state=state,input=rec['input'],cohort=rec['cohort'],step=item['step'],posterior_sha256=rec['fit_sha256'],U=U.tolist(),EMA=ema.tolist(),V=v.tolist(),Q=q.tolist(),shared_experts=torch.nonzero((q>0).sum(0)>=2).flatten().tolist(),eligible_pairs=pairs,ESS=ess.tolist(),NULL=float(pi[:,-1].mean()),posterior_label_accuracy=float((posterior.argmax(1)==y.numpy()).mean())))
     nextmodels.append((rec,model,ema))
    models=nextmodels
  m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and digest(path)==locks[state] and all(p.grad is None for p in m.parameters())
 assert len(rows)==480
 write(OUT/'H2_EMA_replay.json',dict(source_only=True,target_read=False,bank_tensors_restored=True,grad_fields_unassigned=True,rows=rows,checkpoint_sha256=locks,limits='Fixed selected-bank replay, EMA initialized at first saved step200, not historical online Stage2. Source H2 occurrences were already used for H2/joint fits. Each domain32 images; same U/V thresholds, beta.9, cadence5 and ESS2. No objective/gradient changes or target access.',code_sha256=digest(Path(__file__))))
if __name__=='__main__':run()
