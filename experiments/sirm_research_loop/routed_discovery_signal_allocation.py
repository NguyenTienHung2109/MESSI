"""Correct/error discovery-gradient allocation at saved routed bank; no optimizer."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import expert_ce
from .routed_discovery_run import ROOT,DEST
from .routed_conditional_utility import load
from .source_probe_penalty_direction import outputs
from .class_balanced_model import equal_class_cost
from .class_coherent_discovery import class_coherent_cost
from .discovery_error_gradient_diagnosis import vector
from .shrinkage import digest,write
OUT=DEST/'routed_discovery_signal_allocation'

def cosine(a,b):return float(a.dot(b)/(a.norm()*b.norm())) if a.norm()>0 and b.norm()>0 else None

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();m,p,path=load(DEST,'selected');before={k:v.clone() for k,v in m.state_dict().items()};locks={str(path):digest(path)};source={};cache=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';audit=json.loads((cache/'extraction_audit.json').read_text())
 for split in ['source_in','source_out']:
  f=cache/(split+'_tokens.npz');assert digest(f)==next(r['cache_sha256'] for r in audit['rows'] if r['split']==split);locks[str(f)]=digest(f);a=dict(np.load(f));source[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']),a['ids'])
 train,ty,td,_=source['source_in'];val,vy,vd,ids=source['source_out'];m.calibrate_features(train,td);params=list(m.residuals.parameters());m.eval()
 with torch.no_grad():vpi=m.router(m.bank_features(val)).softmax(1)
 vg,vdelta,vf=outputs(m,val,vpi);wrong=vg.argmax(1)!=vy;correct=~wrong;vl=F.cross_entropy(vf,vy,reduction='none');weights=torch.tensor([1/(3*int((vd==d).sum())) for d in vd]);valgrad={name:vector((vl*weights*mask).sum(),params) for name,mask in [('all',torch.ones_like(wrong)),('global_correct',correct),('global_error',wrong)]};casegrads={};caseinfo=[]
 for prefix in ['59f2c51e','598c5b04','59eab95e','59bfe004','5a096976']:
  i=next(i for i,id in enumerate(ids) if prefix in id);casegrads[prefix]=vector(vl[i],params);caseinfo.append(dict(id=str(ids[i]),label=int(vy[i]),global_prediction=int(vg[i].argmax()),native_prediction=int(vf[i].argmax()),native_CE=float(vl[i].detach())))
 h2=DEST/'source_conditional_H2';ha=json.loads((h2/'extraction_audit.json').read_text());locks[str(h2/'extraction_audit.json')]=digest(h2/'extraction_audit.json');plan=dict(source_only=True,target_read=False,fitting=False,parameter_updates=False,checkpoint_sha256=digest(path),selected_step=p['selected_step'],batches='All40 exact scheduled H2 source minibatches at steps200..395, evaluated at ONE saved selected bank. Not historical gradients at those steps.',objective='Actual routed equal-present-class hard winner cost, routing detached, residual batch-centering; gradient norm matched to original image-weighted class-coherent unscaled discovery. Original lambda_discover retained. Split selected contributions by frozen-global correct/error while holding class winner/tie choice and denominators fixed. Group gradients add exactly; norms need not add.',validation='Native saved router probabilities, original source-in mean calibrated at saved parameters. Equal-domain original source-out CE and five predefined cases. Dot-product/cosine diagnose infinitesimal direction only, not Adam updates or accuracy effects.',limits='Source-out already inspected. Batch centering couples examples. Forty batches reuse images/labels. No target/generalizable invariant-factor conclusion from signs alone.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[]
 for br in ha['rows']:
  f=h2/f"H2_step{br['step']}.pt";assert digest(f)==br['sha256'];locks[str(f)]=digest(f);a=torch.load(f,map_location='cpu',weights_only=False);raw,y=a['raw'],a['y']
  with torch.no_grad():pi=m.router(m.bank_features(raw)).softmax(1)
  g,delta,full=outputs(m,raw,pi,True);original=class_coherent_cost(expert_ce(g[:,None]+delta,y),y);per=expert_ce(g[:,None]+pi[:,:6,None].detach()*delta,y);loss=equal_class_cost(per,y);old=vector(original,params);v=vector(loss,params);assert v.norm()>0;scale=(old.norm()/v.norm()).float();labels=y.unique(sorted=True);terms=[];blocks=[]
  for label in labels:
   k=y==label;means=per[k].mean(0);ties=(means==means.min()).detach();w=ties.float()/ties.sum();term=(per*w).sum(1)*k/(len(labels)*int(k.sum()));terms.append(term);blocks.append(dict(label=int(label),n=int(k.sum()),global_errors=int((k&(g.argmax(1)!=y)).sum()),winners=ties.nonzero().flatten().tolist()))
  terms=torch.stack(terms).sum(0);assert torch.allclose(terms.sum(),loss,atol=1e-6,rtol=1e-5);error=g.argmax(1)!=y;dc=vector(m.config['lambda_discover']*scale*(terms*~error).sum(),params);de=vector(m.config['lambda_discover']*scale*(terms*error).sum(),params);total=vector(m.config['lambda_discover']*scale*loss,params);err=float((dc+de-total).abs().max());assert torch.allclose(dc+de,total,atol=1e-7,rtol=2e-4);cls=vector(F.cross_entropy(full,y),params);base=cls+total
  denom=float(total.square().sum());groups={}
  for name,gv in [('global_correct',dc),('global_error',de)]:groups[name]=dict(gradient_norm=float(gv.norm()),projection_fraction_of_total=float(gv.dot(total))/denom,cosine_to_classification=cosine(gv,cls),validation_alignment={name:cosine(gv,vg) for name,vg in valgrad.items()},case_alignment={key:cosine(gv,cg) for key,cg in casegrads.items()})
  rows.append(dict(batch_step=int(br['step']),n=len(y),global_errors=int(error.sum()),class_blocks=blocks,gradient_norm_scale=float(scale),reference_gradient_norm=float(old.norm()),matched_discovery_gradient_norm=float(total.norm()),correct_error_cosine=cosine(dc,de),discovery_classification_cosine=cosine(total,cls),total_validation_alignment={name:cosine(total,vg) for name,vg in valgrad.items()},total_case_alignment={key:cosine(total,cg) for key,cg in casegrads.items()},classification_plus_discovery_validation_alignment={name:cosine(base,vg) for name,vg in valgrad.items()},groups=groups,additive_max_error=err))
  if len(rows)%10==0:print(len(rows),'batches done',flush=True)
 m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert all(q.grad is None for q in m.parameters())
 for f,h in locks.items():assert digest(Path(f))==h
 assert len(rows)==40;write(OUT/'results.json',dict(plan=plan,rows=rows,cases=caseinfo,input_sha256=locks,source_only=True,target_read=False,state_tensors_restored=True,grad_fields_unassigned=True))
if __name__=='__main__':run()
