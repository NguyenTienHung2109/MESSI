"""Equal-norm source-only finite perturbations, recalibrating the actual source mean."""
import json,gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import center_logits,expert_ce
from .spatial_class_allocation import OUT,load,STEPS,ROOT,DEST,PRIOR,vector
from .spatial_bank_replay import spatial_head
from .shrinkage import write,digest
RADIUS=.01

def directions(m,raw,y,mode):
 z=m.bank_features(raw)
 with torch.no_grad():g=m.global_head(raw[:,:384])
 delta=center_logits(torch.stack([e(z) for e in m.residuals],1));delta=delta-(delta.mean(0,keepdim=True) if mode=='batch' else m.expert_source_mean[None]);losses=expert_ce(g[:,None]+delta,y)
 cost={k:losses.sum()*0 for k in ['original','equal_class']};labels=y.unique()
 for c in labels:
  mask=y.eq(c);term=losses[mask].mean(0).amin();cost['original']+=term*mask.sum()/len(y);cost['equal_class']+=term/len(labels)
 return {k:vector(m.config['lambda_discover']*v,list(m.residuals.parameters())) for k,v in cost.items()}
@torch.no_grad()
def score(m,raw,y,dom):
 g,d,pi=spatial_head(m,raw);full=g+(d*pi[:,:6,None]).sum(1)
 return {'CE':float(torch.stack([F.cross_entropy(full[dom.eq(d)].double(),y[dom.eq(d)]) for d in dom.unique()]).mean()),'accuracy':float(torch.stack([full[dom.eq(d)].argmax(1).eq(y[dom.eq(d)]).double().mean() for d in dom.unique()]).mean())}

def main():
 torch.set_num_threads(4);assert json.loads((OUT/'progress.json').read_text())['status']=='complete'
 plan={'source_only':True,'target_read':False,'training':False,'fitting':False,'optimizer_updates':False,'code_sha256':digest(Path(__file__)),'perturbation_L2_norm':RADIUS,'radius_chosen_before_outcomes':True,'scope':'Two disposable equal-L2 residual-parameter descent perturbations per saved state/source batch. Recalibrate equal-domain source-in means after each perturbation, matching inference procedure; frozen router/global/backbone unchanged. No selection, checkpoint save or deployment. Finite source validation effect, not historical gradients or OOD evidence.'};write(OUT/'finite_control_plan.json',plan)
 cache=torch.load(DEST/'source_feature_cache.pt',weights_only=False);rawtrain=cache['features'];domtrain=cache['domains']
 with np.load(PRIOR/'frozen_token_source_diagnosis/source_in_tokens.npz') as a:
  assert a['ids'].tolist()==cache['ids'];trainy=torch.tensor(a['y']);assert np.array_equal(a['domains'],domtrain.numpy())
 with np.load(PRIOR/'frozen_token_source_diagnosis/source_out_tokens.npz') as a:rawval=torch.tensor(np.concatenate([a['cls'],a['patch_max']],1));valy=torch.tensor(a['y']);valdom=torch.tensor(a['domains'])
 records=[]
 for arm in ['CLS_CONTROL','OFF']:
  for state in ['selected','last']:
   m,p,path=load(arm,state);original={k:v.clone() for k,v in m.state_dict().items()};m.calibrate_features(rawtrain,domtrain);mean=m.expert_source_mean.clone();baseline=score(m,rawval,valy,valdom);params=list(m.residuals.parameters());paramcopies=[q.detach().clone() for q in params]
   batches=[('original_source_in',None,rawtrain,trainy,'saved')]
   for step in STEPS:
    b=torch.load(OUT/f'H2_step{step}.pt',weights_only=False);batches.append(('H2',step,b['raw'],b['y'],'batch'))
   for view,step,raw,y,mode in batches:
    vectors=directions(m,raw,y,mode);results={}
    for policy,v in vectors.items():
     assert v.norm()>0;move=(-v/v.norm()*RADIUS).float();pos=0
     with torch.no_grad():
      for q,copy in zip(params,paramcopies):n=q.numel();q.copy_(copy+move[pos:pos+n].reshape_as(q));pos+=n
      observed=torch.cat([(q-copy).flatten().double() for q,copy in zip(params,paramcopies)]).norm();assert abs(float(observed)-RADIUS)<2e-6
     m.calibrate_features(rawtrain,domtrain);s=score(m,rawval,valy,valdom);results[policy]={'actual_parameter_delta_norm':float(observed),'CE':s['CE'],'delta_CE':s['CE']-baseline['CE'],'accuracy':s['accuracy'],'delta_accuracy':s['accuracy']-baseline['accuracy'],'raw_gradient_norm':float(v.norm())}
     with torch.no_grad():
      for q,copy in zip(params,paramcopies):q.copy_(copy)
      m.expert_source_mean.copy_(mean)
    records.append({'arm':arm,'state':state,'view':view,'batch_step':step,'centering':mode,'baseline':baseline,'policies':results,'equal_class_better_CE':results['equal_class']['delta_CE']<results['original']['delta_CE']})
   m.load_state_dict(original);assert all(torch.equal(v,m.state_dict()[k]) for k,v in original.items());assert all(q.grad is None for q in m.parameters());del m,original;gc.collect()
 summary=[]
 for arm in ['CLS_CONTROL','OFF']:
  for state in ['selected','last']:
   for view in ['original_source_in','H2']:
    rr=[r for r in records if r['arm']==arm and r['state']==state and r['view']==view];summary.append({'arm':arm,'state':state,'view':view,'n':len(rr),'equal_class_better':sum(r['equal_class_better_CE'] for r in rr),'policies':{k:{'mean_delta_CE':float(np.mean([r['policies'][k]['delta_CE'] for r in rr])),'decreases_CE':sum(r['policies'][k]['delta_CE']<0 for r in rr)} for k in ['original','equal_class']}})
 write(OUT/'finite_control_results.json',{'plan':plan,'all_original_state_tensors_restored':True,'no_grad_fields_assigned':True,'records':records,'summary':summary});print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
