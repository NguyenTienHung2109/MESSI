"""Source-in repair-margin constraints, never selected from validation cases."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import expert_ce,router_teacher,component_irm,center_logits
from .repair_margin_guard_helpers import project
from .routed_conditional_utility import load
from .routed_discovery_run import ROOT,DEST,PRIOR
from .source_probe_penalty_direction import outputs
from .label_expected_utility_helpers import predict_label,expected_utility
from .teacher_distribution_helpers import teacher
from .class_balanced_model import equal_class_cost,gradient_norm
from .class_coherent_discovery import class_coherent_cost
from .class_balanced_margin_diagnosis import margin,classmean
from .discovery_error_gradient_diagnosis import vector
from .shrinkage import digest,write
OUT=DEST/'source_expected_utility_repair_guard';RADIUS=.01

def route(g,d,posterior):return torch.tensor(teacher(expected_utility(g.detach().double().numpy(),d.detach().double().numpy(),posterior)),dtype=torch.float64)

def assess(m,raw,y,dom,pi,q,posterior=None):
 g,d,_=outputs(m,raw,pi);g=g.double();d=d.double()
 if posterior is not None:pi=route(g,d,posterior)
 f=g+(pi[:,:6,None]*d).sum(1);correct=f.argmax(1).eq(y);gc=g.argmax(1).eq(y);mar=margin(f,y)
 penalty,info=component_irm(g,d,pi,y,dom,q,2.,'binary')
 # Fixed Q mask derivatives regardless of validation ESS; do not call these an eligible penalty.
 prob=f.detach().softmax(1)-F.one_hot(y,10);contribution=(pi[:,:6,None]*d).detach()
 pairs=[(i,j,k) for i,j in itertools.combinations(range(3),2) for k in range(6) if q[i,k]*q[j,k]>0]
 cells=sorted({(d,k) for i,j,k in pairs for d in [i,j]})
 G={f'{i},{k}':float((prob[dom==i]*contribution[dom==i,k]).sum(1).mean()) for i,k in cells}
 metrics=dict(CE=float(torch.stack([F.cross_entropy(f[dom==i],y[dom==i]) for i in range(3)]).mean().detach()),accuracy=float(torch.stack([correct[dom==i].double().mean() for i in range(3)]).mean()),repairs=int((~gc&correct).sum()),harms=int((gc&~correct).sum()),bird_correct=int((correct&(y==0)).sum()),residual_RMS=float(d.detach().square().mean().sqrt()),correction_RMS=float((pi[:,:6,None]*d.detach()).sum(1).square().mean().sqrt()),wrong_margin=float(classmean(mar.detach(),y,~gc)),correct_margin=float(classmean(mar.detach(),y,gc)),fixed_mask_G_mean_abs=float(np.mean([abs(v) for v in G.values()])),fixed_mask_derivatives=G,eligible_validation_penalty=float(penalty.detach()) if info['active'] else None,validation_pair_count=info['pair_count'])
 return metrics,correct,mar.detach(),pi

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists()
 parent=DEST/'source_expected_utility_controls';proof=json.loads((parent/'H2_EMA_replay.json').read_text());fits=json.loads((DEST/'source_label_expected_utility/results.json').read_text());h2proof=json.loads((DEST/'source_conditional_H2/extraction_audit.json').read_text());hashes={r['step']:r['sha256'] for r in h2proof['rows']}
 plan=dict(source_only=True,target_read=False,primary_training=False,probe_refitting=False,optimizer_updates=False,states=list(proof['checkpoint_sha256']),cohorts=['H2','joint'],input='CLS_plus_patch_max',guard='Negative mean true-label margin within each original source-in domain/label group whose frozen-global-wrong examples are repaired by current locked posterior composition. Minimum2 repaired images/group. Membership and posterior routing fixed at saved state; differentiate equal-domain residual centering. Euclidean residual-gradient projection onto all no-margin-decrease halfspaces, no validation labels in guards. This first-order mean constraint does not guarantee per-image or finite-step preservation.',windows='Every saved selected-bank H2/joint feature window with positive eligible-pair count, determined before validation outcomes.',directions=['binary_SIRM','diagnostic_base','residual_output_contraction','SIRM_repair_guard','base_repair_guard','contraction_repair_guard'],radius=RADIUS,lambda_SIRM=1.,ratio_scope='Residual parameters at full lambda1, no ramp. Diagnostic base uses detached expected routing and unchanged donor losses/norm match. Router/backbone/global fixed; these are not trained optimizer ratios.',evaluation='Fixed Q and fixed original validation routing plus recomputed expected routing after each move; source-in mean recalibrated. Fixed-mask mean absolute domain scale derivative is computed over all pair-participating cells irrespective of validation ESS; eligible penalty is separate and nullable.',limits='Saved-bank EMA initializes at step200. Finite .01-radius residual moves are disposable, not a trained400-step OFF/ON comparison or proof of target/invariant factors.',code_sha256=digest(Path(__file__)),H2_replay_sha256=digest(parent/'H2_EMA_replay.json'))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';ta=json.loads((token/'extraction_audit.json').read_text());data={};cachelocks={}
 for split in ['source_in','source_out']:
  p=token/f'{split}_tokens.npz';assert digest(p)==next(r['cache_sha256'] for r in ta['rows'] if r['split']==split);a=np.load(p);cachelocks[str(p)]=digest(p);data[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']),a['ids'])
 train,ty,td,tids=data['source_in'];val,vy,vd,vids=data['source_out'];assert not set(tids)&set(vids);vdom=torch.tensor([0 if int(d)==0 else 1 if int(d)==2 else 2 for d in vd]);rows=[];locks={}
 for state in plan['states']:
  m,p,path=load(PRIOR if state.startswith('unscaled') else DEST,'selected');locks[state]=digest(path);assert locks[state]==proof['checkpoint_sha256'][state]
  assert not any(p.requires_grad for p in m.featurizer.parameters()) and not any(p.requires_grad for p in m.global_head.parameters())
  original={k:v.clone() for k,v in m.state_dict().items()};m.calibrate_features(train,td);mean=m.expert_source_mean.clone();params=list(m.residuals.parameters());copies=[p.detach().clone() for p in params]
  for cohort in plan['cohorts']:
   rec=next(r for r in fits['states'][state] if r['input']==plan['input'] and r['cohort']==cohort);file=DEST/'source_label_expected_utility'/rec['fit_file'];assert digest(file)==rec['fit_sha256'];model=dict(np.load(file));vp=predict_label(model,val.double().numpy())
   with torch.no_grad():
    vg,vdelta,_=outputs(m,val,torch.zeros((len(val),7)));vpi=route(vg,vdelta,vp)
    tdom=torch.tensor([0 if int(d)==0 else 1 if int(d)==2 else 2 for d in td]);tp=predict_label(model,train.double().numpy());tg,tdelta,_=outputs(m,train,torch.zeros((len(train),7)));tpi=route(tg,tdelta,tp)
   with torch.no_grad():tz=m.bank_features(train);tg=m.global_head(train[:,:384]).double()
   rawdelta=center_logits(torch.stack([e(tz) for e in m.residuals],1)).double();tdelta=rawdelta-torch.stack([rawdelta[tdom==i].mean(0) for i in range(3)]).mean(0)[None];tf=tg+(tpi[:,:6,None]*tdelta).sum(1);tm=margin(tf,ty);repair=(tg.argmax(1)!=ty)&(tf.argmax(1)==ty);guards=[];guardcohorts=[];skipped=[];guardmasks=[]
   for domain in range(3):
    for label in ty[tdom==domain].unique(sorted=True):
     mask=repair&tdom.eq(domain)&ty.eq(label);count=int(mask.sum())
     if count<2:
      if count:skipped.append(dict(domain=domain,label=int(label),count=count))
      continue
     guard=vector(-tm[mask].mean(),params);assert torch.isfinite(guard).all() and guard.norm()>0;guards.append(guard);guardmasks.append(mask)
     guardcohorts.append(dict(domain=domain,label=int(label),count=count,source_in_ids=tids[mask.numpy()].tolist(),baseline_mean_margin=float(tm[mask].mean().detach())))
   del rawdelta,tdelta,tf,tm
   for gate in proof['rows']:
    if gate['state']!=state or gate['cohort']!=cohort or gate['input']!=plan['input'] or gate['eligible_pairs']==0:continue
    step=gate['step'];cache=DEST/'source_conditional_H2'/f'H2_step{step}.pt';assert digest(cache)==hashes[step];cachelocks[str(cache)]=digest(cache);a=torch.load(cache,map_location='cpu',weights_only=False);raw,y,dom=a['raw'],a['y'],a['domains'];dom=torch.tensor([0 if int(d)==0 else 1 if int(d)==2 else 2 for d in dom]);posterior=predict_label(model,raw.double().numpy());g,d,_=outputs(m,raw,torch.zeros((len(raw),7)),True);g=g.double();d=d.double();pi=route(g,d,posterior);q=torch.tensor(gate['Q'],dtype=torch.float64);full=g+(pi[:,:6,None]*d).sum(1);sirm,info=component_irm(g,d,pi,y,dom,q,2.,'binary');assert info['active'] and info['pair_count']==gate['eligible_pairs']
    per=expert_ce(g[:,None]+d,y);reference=class_coherent_cost(per,y);chosen=expert_ce(g[:,None]+pi[:,:6,None]*d,y) if state.startswith('routed') else per;balanced=equal_class_cost(chosen,y);oldnorm=gradient_norm(reference,params);newnorm=gradient_norm(balanced,params);discovery=balanced*(oldnorm/newnorm).detach().to(balanced.dtype) if newnorm>0 else reference
    target=router_teacher(F.cross_entropy(g,y,reduction='none')[:,None]-per,.1,.002);loads=(-per/m.config['discovery_temperature']).softmax(1).mean(0);balance=(6*loads.square().sum()-1)*(1-target[:,-1]).mean();base=F.cross_entropy(full,y)+m.config['lambda_discover']*discovery+m.config['lambda_balance']*balance
    vecs={'binary_SIRM':vector(sirm,params),'diagnostic_base':vector(base,params),'residual_output_contraction':vector(d.square().mean(),params)};projection={}
    for old,new in [('binary_SIRM','SIRM_repair_guard'),('diagnostic_base','base_repair_guard'),('residual_output_contraction','contraction_repair_guard')]:vecs[new],projection[new]=project(vecs[old],guards)
    baseline,bc,bm,_=assess(m,val,vy,vdom,vpi,q);policies={}
    for name,v in vecs.items():
     assert torch.isfinite(v).all() and v.norm()>0;move=(-RADIUS*v/v.norm()).float();offset=0
     with torch.no_grad():
      for param,copy in zip(params,copies):n=param.numel();param.copy_(copy+move[offset:offset+n].reshape_as(param));offset+=n
      actual=float(torch.cat([(p-copy).flatten().double() for p,copy in zip(params,copies)]).norm());assert abs(actual-RADIUS)<2e-6
     m.calibrate_features(train,td);moved={}
     for label,recompute in [('fixed_routing',False),('recomputed_routing',True)]:
      metrics,correct,mar,mpi=assess(m,val,vy,vdom,vpi,q,vp if recompute else None);changed=torch.nonzero(correct!=bc).flatten().tolist()
      changes={k:metrics[k]-baseline[k] for k in baseline if k!='fixed_mask_derivatives' and metrics[k] is not None and baseline[k] is not None}
      moved[label]=dict(metrics=metrics,delta=changes,case_changes=[dict(image_id=str(vids[i]),was_correct=bool(bc[i]),now_correct=bool(correct[i]),baseline_margin=float(bm[i]),new_margin=float(mar[i])) for i in changed],margin_deltas_all=(mar-bm).tolist(),routing_L1_mean=float((mpi-vpi).abs().sum(1).mean()))
     with torch.no_grad():
      tg,tdelta,_=outputs(m,train,tpi);trainfull=tg.double()+(tpi[:,:6,None]*tdelta.double()).sum(1);mt=margin(trainfull,ty);trainrecompute=tg.double()+(route(tg,tdelta,tp)[:,:6,None]*tdelta.double()).sum(1);mtr=margin(trainrecompute,ty)
      guard_after=[dict(domain=c['domain'],label=c['label'],count=c['count'],fixed_mean_margin=float(mt[mask].mean()),fixed_margin_delta=float(mt[mask].mean())-c['baseline_mean_margin'],recomputed_mean_margin=float(mtr[mask].mean()),recomputed_margin_delta=float(mtr[mask].mean())-c['baseline_mean_margin'],fixed_repair_losses=int((mt[mask]<=0).sum()),recomputed_repair_losses=int((mtr[mask]<=0).sum())) for c,mask in zip(guardcohorts,guardmasks)]
     policies[name]=dict(gradient_norm=float(v.norm()),actual_radius=actual,assessments=moved,source_in_guard_after=guard_after)
     with torch.no_grad():
      for param,copy in zip(params,copies):param.copy_(copy)
      m.expert_source_mean.copy_(mean)
    rows.append(dict(state=state,cohort=cohort,step=step,posterior_sha256=rec['fit_sha256'],pair_count=info['pair_count'],training_G=info['G'],ESS=info['ess'].tolist(),Q=q.tolist(),raw_penalty=float(sirm.detach()),gradient_ratio_to_diagnostic_base=float(vecs['binary_SIRM'].norm()/vecs['diagnostic_base'].norm()),baseline=baseline,guard_cohorts=guardcohorts,guard_skipped_below_min2=skipped,projection=projection,policies=policies))
   print(state,cohort,'complete',flush=True)
  m.load_state_dict(original);assert all(torch.equal(v,m.state_dict()[k]) for k,v in original.items()) and all(p.grad is None for p in m.parameters()) and digest(path)==locks[state]
 assert len(rows)==sum(r['eligible_pairs']>0 for r in proof['rows'] if r['input']==plan['input'] and r['cohort'] in plan['cohorts'])==19
 for path,h in cachelocks.items():assert digest(Path(path))==h
 write(OUT/'results.json',dict(plan=plan,rows=rows,validation_ids=vids.tolist(),checkpoint_sha256=locks,cache_sha256=cachelocks,source_only=True,target_read=False,bank_tensors_restored=True,grad_fields_unassigned=True))
if __name__=='__main__':run()
