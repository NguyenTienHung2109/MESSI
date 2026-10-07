"""Source-only native discovery projection; disposable finite moves, no fitting."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import expert_ce,center_logits
from .routed_discovery_run import ROOT,DEST
from .routed_conditional_utility import load
from .source_probe_penalty_direction import outputs
from .class_balanced_model import equal_class_cost
from .class_coherent_discovery import class_coherent_cost
from .discovery_error_gradient_diagnosis import vector
from .repair_margin_guard_helpers import project
from .shrinkage import digest,write
OUT=DEST/'native_discovery_correct_guard'
def cosine(a,b):return float(a.dot(b)/(a.norm()*b.norm())) if a.norm()>0 and b.norm()>0 else None

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists()
 audit=DEST/'final_analysis_audit.json';oldaudit=json.loads(audit.read_text())
 for f,h in oldaudit['sha256'].items():assert digest(DEST/f)==h,f
 m,p,path=load(DEST,'selected');before={k:v.clone() for k,v in m.state_dict().items()};locks={str(path):digest(path)}
 cache=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';ca=json.loads((cache/'extraction_audit.json').read_text());data={}
 for split in ['source_in','source_out']:
  f=cache/(split+'_tokens.npz');assert digest(f)==next(r['cache_sha256'] for r in ca['rows'] if r['split']==split);locks[str(f)]=digest(f);a=dict(np.load(f));data[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']),a['ids'])
 raw,y,dom,_=data['source_in'];val,vy,vd,ids=data['source_out'];m.eval();m.calibrate_features(raw,dom);params=list(m.residuals.parameters());saved=[q.detach().clone() for q in params]
 with torch.no_grad():z=m.bank_features(raw);g=m.global_head(raw[:,:384]);pi=m.router(z).softmax(1);vpi=m.router(m.bank_features(val)).softmax(1)
 # Differentiate full-source equal-domain centering, matching finite recalibration.
 d=center_logits(torch.stack([e(z) for e in m.residuals],1));mean=torch.stack([d[dom==v].mean(0) for v in dom.unique()]).mean(0);full=g+(pi[:,:6,None]*(d-mean)).sum(1);per=F.cross_entropy(full,y,reduction='none');correct=g.argmax(1)==y
 guards=[];cohorts=[]
 for domain in dom.unique():
  for label in y.unique():
   mask=(dom==domain)&(y==label)&correct;n=int(mask.sum())
   if n<2:continue
   gv=vector(per[mask].mean(),params)
   if gv.norm()>0:guards.append(gv);cohorts.append(dict(domain=int(domain),label=int(label),n=n,CE=float(per[mask].mean().detach()),gradient_norm=float(gv.norm())))
 vg,_,vf=outputs(m,val,vpi);vl=F.cross_entropy(vf,vy,reduction='none');valgrad=vector(torch.stack([vl[vd==v].mean() for v in vd.unique()]).mean(),params)
 prefixes=['59f2c51e','598c5b04','59bfe004','5a096976'];ix={s:next(i for i,id in enumerate(ids) if s in id) for s in prefixes};casegrad={s:vector(vl[i],params) for s,i in ix.items()}
 def stats():
  with torch.no_grad():
   tg,_,tf=outputs(m,raw,pi);_,_,out=outputs(m,val,vpi);tc=F.cross_entropy(tf,y,reduction='none');oc=F.cross_entropy(out,vy,reduction='none');pred=out.argmax(1);gc=vg.argmax(1)==vy;pc=pred==vy
   return dict(source_in_correct_cohort_CE=[float(tc[(dom==c['domain'])&(y==c['label'])&correct].mean()) for c in cohorts],source_out_CE=float(torch.stack([oc[vd==v].mean() for v in vd.unique()]).mean()),source_out_accuracy=float(torch.stack([pc[vd==v].float().mean() for v in vd.unique()]).mean()),repairs=int((~gc&pc).sum()),harms=int((gc&~pc).sum()),cases={s:dict(CE=float(oc[i]),prediction=int(pred[i])) for s,i in ix.items()})
 baseline=stats();ha=json.loads((DEST/'source_conditional_H2/extraction_audit.json').read_text());rows=[]
 plan=dict(source_only=True,target_read=False,training=False,fitting=False,selected_step=p['selected_step'],checkpoint_sha256=digest(path),guard='Source-in global-correct domain x class mean native CE, min2 samples, differentiable equal-domain expert centering. Frozen bank/router/global; residual gradient projection onto non-increasing first-order CE halfspaces.',probe='All40 scheduled H2 discovery gradients at ONE saved selected state, not historical gradients. Norm-matched actual routed discovery. Raw/projected directions compared at equal residual parameter L2 radius0.01. Finite moves recalibrate source-in centering. No candidate or checkpoint selection.',limits='Mean group protection is neither per-image nor source-out protection; finite moves are not Adam updates. Already inspected source-out is exploratory. No target access.',previous_controls='Unlike earlier one pooled correct-margin guard with probe utility routing and error-cohort margin guards, this guards domain/class correct CE against actual native routed discovery.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 for br in ha['rows']:
  f=DEST/f"source_conditional_H2/H2_step{br['step']}.pt";assert digest(f)==br['sha256'];locks[str(f)]=digest(f);b=torch.load(f,map_location='cpu',weights_only=False)
  with torch.no_grad():bpi=m.router(m.bank_features(b['raw'])).softmax(1)
  bg,bd,_=outputs(m,b['raw'],bpi,True);reference=class_coherent_cost(expert_ce(bg[:,None]+bd,b['y']),b['y']);loss=equal_class_cost(expert_ce(bg[:,None]+bpi[:,:6,None]*bd,b['y']),b['y']);rv=vector(reference,params);v=vector(loss,params);v=v*rv.norm()/v.norm()*m.config['lambda_discover'];protected,cert=project(v,guards)
  row=dict(step=int(br['step']),projection=cert,directions={})
  for name,gradient in [('raw',v),('protected',protected)]:
   info=dict(source_out_CE_alignment=cosine(gradient,valgrad),case_alignment={s:cosine(gradient,cv) for s,cv in casegrad.items()});move=-.01*gradient/gradient.norm();offset=0
   with torch.no_grad():
    for q,original in zip(params,saved):n=q.numel();q.copy_(original+move[offset:offset+n].reshape(q.shape).to(q));offset+=n
   m.calibrate_features(raw,dom);info['finite']=stats();row['directions'][name]=info
   with torch.no_grad():
    for q,original in zip(params,saved):q.copy_(original)
   m.calibrate_features(raw,dom)
  rows.append(row)
  if len(rows)%10==0:print(len(rows),'batches checked',flush=True)
 m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert all(q.grad is None for q in m.parameters());assert all(not q.requires_grad for q in list(m.featurizer.parameters())+list(m.global_head.parameters()))
 for f,h in locks.items():assert digest(Path(f))==h
 write(OUT/'results.json',dict(plan=plan,cohorts=cohorts,baseline=baseline,rows=rows,input_sha256=locks,state_restored=True,grad_fields_unassigned=True))
if __name__=='__main__':run()
