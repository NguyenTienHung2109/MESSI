"""Frozen-bank source-only utility-sign accessibility with router-domain holdouts."""
import copy,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from .routed_conditional_utility import load
from .source_probe_penalty_direction import outputs
from .source_direct_mixture_reliability_router import descriptor,FEATURES
from .shrinkage import digest,write
ROOT=Path('results/sirm_research_loop/iteration_17')
OUT=ROOT/'source_reliability_probe'
DATA=Path('results/terra_attention_diagnosis/phase1/source_object_probe/blur_residual_control')
def assess(score,u,dom):
 rows=[]
 for d in sorted(dom.unique().tolist()):
  m=dom.eq(d);ss=score[m].numpy().ravel();uu=u[m].numpy().ravel();nz=uu!=0;yy=uu[nz]>0;ss=ss[nz]
  rows.append(dict(domain=d,n_images=int(m.sum()),n_expert_instances=len(yy),positive_fraction=float(yy.mean()),AUROC=float(roc_auc_score(yy,ss)) if len(set(yy))==2 else None,balanced_accuracy=float(balanced_accuracy_score(yy,ss>=0)) if len(set(yy))==2 else None,mean_utility_selected=float(uu[(score[m].numpy().ravel()>=0)].mean()) if (score[m]>=0).any() else None,mean_utility_rejected=float(uu[(score[m].numpy().ravel()<0)].mean()) if (score[m]<0).any() else None))
 return rows

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists(),'Do not overwrite a diagnostic'
 audit=json.loads((ROOT/'final_analysis_audit.json').read_text());assert audit['analysis_complete']
 for rel,h in audit['sha256'].items():assert digest(ROOT/rel)==h,rel
 paths={s:DATA/f'{s}_features.pt' for s in ['source_in','source_out']};locks={str(p):digest(p) for p in paths.values()}
 write(OUT/'plan.json',dict(hypothesis='Shared confidence descriptors predict sign of routed marginal utility across source cameras, not merely in fitting data.',source_only=True,target_read=False,diagnostic_fitting=True,stage2_training=False,features=FEATURES[:11],architecture='Shared11->1 linear utility-sign probe; no label/domain/expert identity inputs at prediction',fits='selected350 and final399 frozen OFF banks; all-source and3 router-domain-excluded fits,400 full-batch Adam .01 updates,L2 .01; train-only feature standardization. Equal-domain and equal-sign BCE weighting. Sign0 excluded from fit and sign metrics.',selection='Fixed final400 diagnostic; no checkpoint/L2/threshold tuning',evaluation='Source-out equal-domain summaries; held domains excluded only from probe, not frozen backbone/bank. Expert observations and frames correlated. Not a deployed routing/accuracy test; marginal utility measured under original full mixture.',input_sha256=locks,code_sha256=digest(Path(__file__))))
 (OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());data={s:torch.load(p,map_location='cpu',weights_only=False) for s,p in paths.items()};assert not set(data['source_in']['ids'])&set(data['source_out']['ids']);results={}
 for state in ['selected','last']:
  m,p,path=load(ROOT,state);m.requires_grad_(False);before=copy.deepcopy(m.state_dict());locks[str(path)]=digest(path);m.calibrate_features(data['source_in']['original'],data['source_in']['domains']);fixed={}
  with torch.no_grad():
   for split,a in data.items():
    raw=a['original'];pi=m.router(m.bank_features(raw)).softmax(1);g,d,f=outputs(m,raw,pi);ce=F.cross_entropy(f,a['labels'],reduction='none');u=torch.stack([F.cross_entropy(f-pi[:,e,None]*d[:,e],a['labels'],reduction='none')-ce for e in range(6)],1);xx=descriptor(g,d,d)[:,:,:11];fixed[split]=dict(x=xx,u=u,dom=a['domains'])
  fits={}
  for held in [None,0,2,3]:
   a=fixed['source_in'];keep=torch.ones(len(a['dom']),dtype=torch.bool) if held is None else a['dom'].ne(held);xx=a['x'][keep];u=a['u'][keep];dom=a['dom'][keep];mean=xx.mean((0,1));std=xx.std((0,1),unbiased=False).clamp_min(1e-8);xx=(xx-mean)/std;weight=torch.zeros_like(u,dtype=torch.double)
   for dd in dom.unique():
    for flag in [False,True]:
     mask=dom.eq(dd)[:,None]&(u>0 if flag else u<0);assert mask.sum()>=2;weight[mask]=1./mask.sum()
   weight/=weight.sum();head=torch.nn.Linear(11,1).double();head.weight.data.zero_();head.bias.data.zero_();opt=torch.optim.Adam(head.parameters(),lr=.01)
   for _ in range(400):
    score=head(xx).squeeze(-1);loss=(F.binary_cross_entropy_with_logits(score,(u>0).double(),reduction='none')*weight).sum()+.01*head.weight.square().sum();opt.zero_grad();loss.backward();opt.step()
   with torch.no_grad():
    record={}
    for split,b in fixed.items():
     take=keep if split=='source_in' else (torch.ones(len(b['dom']),dtype=torch.bool) if held is None else b['dom'].eq(held));scores=head((b['x'][take]-mean)/std).squeeze(-1);record[split]=assess(scores,b['u'][take],b['dom'][take])
    torch.save(dict(weight=head.weight.detach(),bias=head.bias.detach(),mean=mean,std=std),OUT/f'{state}_held{held}.pt')
   fits[str(held)]=record;print(state,held,record['source_out'],flush=True)
  results[state]=fits;m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert all(q.grad is None for q in m.parameters())
 for p,h in locks.items():assert digest(Path(p))==h
 write(OUT/'results.json',dict(source_only=True,target_read=False,diagnostic_fitting=True,bank_unchanged=True,goal_achieved=False,states=results,input_sha256=locks))
 print('complete',flush=True)
if __name__=='__main__':main()
