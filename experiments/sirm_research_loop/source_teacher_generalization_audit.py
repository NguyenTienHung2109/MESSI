"""Fixed teacher-student versus fitting-cohort constant-prior control."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.sirm_discover.math import center_logits
from .routed_discovery_run import ROOT,DEST,PRIOR
from .routed_conditional_utility import load
from .source_accessibility_H2 import inputs
from .source_conditional_accessibility import utility
from .teacher_distribution_helpers import teacher,predict_pi,evaluate_pi
from .shrinkage import digest,write
OUT=DEST/'source_teacher_distribution_probe'
@torch.no_grad()
def run():
 torch.set_num_threads(4);fits=json.loads((OUT/'results.json').read_text());original=json.loads((DEST/'source_conditional_accessibility/results.json').read_text());proof=json.loads((DEST/'source_conditional_H2/extraction_audit.json').read_text());token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';rows=[]
 for state,records in fits['states'].items():
  m,p,path=load(PRIOR if state.startswith('unscaled') else DEST,'selected' if state.endswith('selected') else 'last');assert digest(path)==fits['checkpoint_sha256'][state];before={k:v.clone() for k,v in m.state_dict().items()}
  si=np.load(token/'source_in_tokens.npz');train=torch.tensor(np.concatenate([si['cls'],si['patch_max']],1));m.calibrate_features(train,torch.tensor(si['domains']));bankpath=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';assert digest(bankpath)==original['cache_sha256'][state];bank=np.load(bankpath);data={}
  for split in ['source_in','source_out']:
   b=np.load(token/f'{split}_tokens.npz');assert np.array_equal(bank[split+'_ids'],b['ids']);g=b['global_logits'].astype(float);d=bank[split+'_bank_delta'].astype(float);pi=bank[split+'_bank_pi'].astype(float)
   data[split]=dict(g=g,d=d,y=b['y'],dom=b['domains'],ids=b['ids'],u=utility(g,d,b['y']),X={'CLS_plus_patch_max':np.concatenate([b['cls'],b['patch_max']],1).astype(float),'global_and_expert_logits_and_pi':np.concatenate([g-g.mean(1,keepdims=True),d.reshape(len(g),-1),pi],1)})
  hu=[];hd=[]
  for rec in proof['rows']:
   f=DEST/'source_conditional_H2'/f"H2_step{rec['step']}.pt";assert digest(f)==rec['sha256'];b=torch.load(f,map_location='cpu',weights_only=False);z=m.bank_features(b['raw']);g=m.global_head(b['raw'][:,:384]);d=center_logits(torch.stack([e(z) for e in m.residuals],1));d=d-d.mean(0,keepdim=True);hu.append(utility(g.double().numpy(),d.double().numpy(),b['y'].numpy()));hd.append(b['domains'].numpy())
  hteacher=teacher(np.concatenate(hu));hdom=np.concatenate(hd);ot=teacher(data['source_in']['u']);odom=data['source_in']['dom']
  for record in records:
   file=OUT/f"{state}_{record['input']}_{record['cohort']}.npz";assert digest(file)==record['fit_sha256'];model=dict(np.load(file));cohort=record['cohort'];o_prior=np.mean([ot[odom==d].mean(0) for d in [0,2,3]],0);h_prior=np.mean([hteacher[hdom==d].mean(0) for d in [0,2,3]],0);prior=o_prior if cohort=='original' else h_prior if cohort=='H2' else (o_prior+h_prior)/2;result={}
   for split,b in data.items():
    pi=predict_pi(model,b['X'][record['input']]);cp=np.broadcast_to(prior,pi.shape);wrong=b['g'].argmax(1)!=b['y'];out={}
    for name,mask in [('all',np.ones(len(wrong),bool)),('global_correct',~wrong),('global_errors',wrong)]:
     out[name]={policy:evaluate_pi(b['g'][mask],b['d'][mask],b['y'][mask],b['dom'][mask],b['ids'][mask],prob[mask],b['u'][mask]) for policy,prob in [('student',pi),('constant_prior',cp)]}
    result[split]=out
   rows.append(dict(state=state,input=record['input'],cohort=cohort,fitting_prior=prior.tolist(),metrics=result))
  m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and all(p.grad is None for p in m.parameters());print(state,'audit complete',flush=True)
 write(OUT/'generalization_audit.json',dict(source_only=True,target_read=False,refitting=False,constant_control='Equal-domain source-in teacher prior for the same fit cohort; joint half per view. No validation labels in prior.',rows=rows,code_sha256=digest(Path(__file__))))
if __name__=='__main__':run()
