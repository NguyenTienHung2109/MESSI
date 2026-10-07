"""Matched two-event source-error replay directions at fixed saved banks."""
import json,itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .routed_discovery_run import ROOT,DEST,PRIOR
from .routed_conditional_utility import load
from .source_probe_penalty_direction import outputs
from .class_balanced_margin_diagnosis import metrics
from .discovery_error_gradient_diagnosis import vector
from .label_expected_utility_helpers import predict_label,expected_utility
from .teacher_distribution_helpers import teacher,gates_pi
from .source_residual_rank_diagnosis import summarize
from .shrinkage import digest,write
OUT=DEST/'source_error_contrast_directions';RADIUS=.01

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();locks={};supportfile=DEST/'source_error_contrast_support/results.json';support=json.loads(supportfile.read_text());locks[str(supportfile)]=digest(supportfile);rank=json.loads((DEST/'source_residual_rank_diagnosis/results.json').read_text());source={}
 for split in ['source_in','source_out']:
  path=ROOT/f'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/{split}_tokens.npz';assert digest(path)==rank['input_sha256'][str(path)];source[split]=dict(np.load(path));locks[str(path)]=digest(path)
 meta=ROOT/'results/sirm_research_loop/iteration_08/source_annotation_audit/selected_source_results.json';metadata={x['local_id']:x for x in json.loads(meta.read_text())['records']};locks[str(meta)]=digest(meta);proof=json.loads((DEST/'source_conditional_H2/extraction_audit.json').read_text());caches={}
 for r in proof['rows']:
  f=DEST/'source_conditional_H2'/f"H2_step{r['step']}.pt";assert digest(f)==r['sha256'];locks[str(f)]=digest(f);caches[r['step']]=torch.load(f,map_location='cpu',weights_only=False)
 plan=dict(source_only=True,target_read=False,primary_training=False,optimizer_updates=False,bank_updates=False,minimum2_distinct_images_and_events=True,selection='All9 pooled same-error contrasts supported by2 distinct source-in images/events per at least2 domains. Choose first2 in saved step/row order per eligible domain, distinct IDs/sequences; no validation choice. Choose raw CE-minimal shared expert on selected replay examples.',centering='Each chosen example retains its original96-image H2 batch-centering context, recomputed differentiably at saved weights. Do not center a homogeneous4-example error cohort; inference expert means recalibrate on unchanged original source-in.',directions=['selected_expert_raw_CE','selected_expert_binary_error_contrast'],control='Same examples/expert, fixed parameter L2 radius. Binary contrast is softplus(-(global_y-global_guess+delta_y-delta_guess)), not a trained factor objective or an inference gate. Per-domain raw CE gradient cosines diagnose alignment.',radius=RADIUS,evaluation='Source-out native and original feature-posterior composition, with fixed and recomputed posterior utility routing. Source-in means recalibrated after disposable perturbations. Preserve frozen backbone/global/router and restore all tensors/grad fields.',limits='Saved-state local directions, not Adam updates or400-step trajectories. Source-out previously selected/inspected. Known-label selection cannot be deployed as a router; positive outcomes alone do not demonstrate semantic/invariant factors.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[];postroot=DEST/'source_label_expected_utility';postrecords=json.loads((postroot/'results.json').read_text());locks[str(postroot/'results.json')]=digest(postroot/'results.json')
 si=source['source_in'];so=source['source_out'];train=torch.tensor(np.concatenate([si['cls'],si['patch_max']],1));td=torch.tensor(si['domains']);val=torch.tensor(np.concatenate([so['cls'],so['patch_max']],1));vy=torch.tensor(so['y']);vd=torch.tensor(so['domains']);vids=so['ids'];assert not set(si['ids'])&set(vids)
 for state in rank['states']:
  m,p,path=load(PRIOR if state.startswith('unscaled') else DEST,'selected');assert digest(path)==support['input_sha256'][str(path)];locks[str(path)]=digest(path);before={k:v.clone() for k,v in m.state_dict().items()};assert all(not v.requires_grad for module in [m.featurizer,m.global_head] for v in module.parameters());m.calibrate_features(train,td);mean=m.expert_source_mean.clone();params=list(m.residuals.parameters());copies=[p.detach().clone() for p in params];postrec=next(r for r in postrecords['states'][state] if r['input']=='CLS_plus_patch_max' and r['cohort']=='original');postfile=postroot/postrec['fit_file'];assert digest(postfile)==postrec['fit_sha256'];locks[str(postfile)]=digest(postfile);posterior=predict_label(dict(np.load(postfile)),val.double().numpy())
  def assessment(fixed=None):
   native,_,_,_=metrics(m,val,vy,vd)
   with torch.no_grad():
    g,d,_=outputs(m,val,torch.zeros((len(val),7)));g=g.numpy().astype(float);d=d.numpy().astype(float)
   pi=teacher(expected_utility(g,d,posterior)) if fixed is None else fixed;out=summarize(g,d,pi,so['y'],so['domains'],vids);out['Q']=gates_pi(g,d,so['y'],so['domains'],pi)
   return dict(native=native,expected_composition=out),pi
  baseline,vpi=assessment()
  # Frozen global predictions used for group membership, independent of bank.
  global_pred={}
  with torch.no_grad():
   for step,b in caches.items():global_pred[step]=m.global_head(b['raw'][:,:384]).argmax(1)
  for group in support['states'][state]['H2_pooled']['shared_contrasts_distinct_image_min2']:
   label,guess=group['key'];selected=[]
   for domain in group['domains']:
    ids=set();events=set()
    for step,b in caches.items():
     eligible=(b['y']==label)&(b['domains']==domain)&(global_pred[step]==guess)
     for index in torch.nonzero(eligible).flatten().tolist():
      id=b['ids'][index];event=metadata[id]['official_sequence']
      if id in ids or event is None or event in events:continue
      selected.append(dict(domain=domain,step=step,index=index,image_id=id,official_sequence=event));ids.add(id);events.add(event)
      if len(ids)==2:break
     if len(ids)==2:break
    assert len(ids)==2 and len(events)==2
   gg=[];dd=[];domains=[]
   for step in sorted(set(x['step'] for x in selected)):
    b=caches[step];g,d,_=outputs(m,b['raw'],torch.zeros((len(b['y']),7)),True)
    for item in selected:
     if item['step']!=step:continue
     i=item['index'];gg.append(g[i]);dd.append(d[i]);domains.append(item['domain'])
   g=torch.stack(gg);delta=torch.stack(dd);dom=torch.tensor(domains);y=torch.full((len(dom),),label,dtype=torch.long);per=F.cross_entropy((g[:,None]+delta).reshape(-1,10),y[:,None].expand(-1,6).reshape(-1),reduction='none').reshape(-1,6);expert=int(per.mean(0).argmin());raw=per[:,expert].mean();contrast=g[:,label]-g[:,guess]+delta[:,expert,label]-delta[:,expert,guess];binary=F.softplus(-contrast).mean();grads={int(d):vector(per[dom==d,expert].mean(),params) for d in dom.unique()};alignment=[]
   for aa,bb in itertools.combinations(grads,2):
    a,b=grads[aa],grads[bb];alignment.append(dict(domains=[aa,bb],cosine=float(F.cosine_similarity(a,b,dim=0)) if a.norm()>0 and b.norm()>0 else None,gradient_norms=[float(a.norm()),float(b.norm())]))
   vectors={'selected_expert_raw_CE':vector(raw,params),'selected_expert_binary_error_contrast':vector(binary,params)};policies={}
   for name,v in vectors.items():
    assert v.norm()>0 and torch.isfinite(v).all();move=(-RADIUS*v/v.norm()).float();pos=0
    with torch.no_grad():
     for param,copy in zip(params,copies):n=param.numel();param.copy_(copy+move[pos:pos+n].reshape_as(param));pos+=n
     actual=float(torch.cat([(p-copy).flatten().double() for p,copy in zip(params,copies)]).norm());assert abs(actual-RADIUS)<2e-6
    m.calibrate_features(train,td);new,_=assessment();fixed,_=assessment(vpi);policies[name]=dict(gradient_norm=float(v.norm()),actual_radius=actual,recomputed_routing=new,fixed_expected_routing=fixed)
    with torch.no_grad():
     for param,copy in zip(params,copies):param.copy_(copy)
     m.expert_source_mean.copy_(mean)
   rows.append(dict(state=state,label=label,global_prediction=guess,selected_examples=selected,expert=expert,training_raw_CE=float(raw.detach()),training_binary_CE=float(binary.detach()),raw_domain_gradient_alignment=alignment,raw_binary_gradient_cosine=float(F.cosine_similarity(vectors['selected_expert_raw_CE'],vectors['selected_expert_binary_error_contrast'],dim=0)),baseline=baseline,policies=policies));print(state,label,guess,'expert',expert,'cos',[round(x['cosine'],3) for x in alignment],flush=True)
  m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and all(p.grad is None for p in m.parameters())
 for f,h in locks.items():assert digest(Path(f))==h
 assert len(rows)==18
 write(OUT/'results.json',dict(plan=plan,rows=rows,input_sha256=locks,source_only=True,target_read=False,bank_tensors_restored=True,grad_fields_unassigned=True))
if __name__=='__main__':run()
