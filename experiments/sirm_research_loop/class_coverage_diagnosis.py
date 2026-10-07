"""Source-only class coverage and matched finite-gradient diagnostic."""
import json,hashlib,gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from domainbed.deit_transformer import DeiTFeaturizer
from domainbed.scripts.train_subset_irm_pacs import make_schedule
from experiments.sirm_discover_gate.data import Adapter
from experiments.sirm_h1_h6.run import ScheduledImages
from experiments.sirm_discover.math import center_logits,expert_ce,residual_utilities,soft_null_q
from .routed_discovery_run import ROOT,DEST,PRIOR,ANCHOR
from .spatial_bank_model import SpatialBankGateSIRM,pooled_raw_features
from .class_balanced_model import equal_class_cost
from .class_balanced_margin_diagnosis import graph,metrics,margin,classmean
from .discovery_error_gradient_diagnosis import vector
from .shrinkage import digest,write
OUT=DEST/'source_class_coverage';STEPS=[200,215,250,300,350,395];RADIUS=.01

def load(state):
 path=PRIOR/'experiment/runs/L38/OFF'/('source_best.pkl' if state=='selected' else 'last.pkl');p=torch.load(path,map_location='cpu',weights_only=False);c=p['config'];m=SpatialBankGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c);m.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()});m.configure_stage();m.eval();return m,p,path

def extract():
 m,p,path=load('selected');phi=m.featurizer.cuda().eval();before={k:v.cpu().clone() for k,v in phi.state_dict().items()};manifest=json.loads((DEST/'experiment/subset_manifest.json').read_text());adapter=Adapter('TerraIncognita');adapter.classes=manifest['class_names'];source=dict(manifest,sources={str(d):manifest['sources'][str(d)] for d in [0,2,3]});datasets=adapter.source_data(ROOT/'domainbed/data',source,p['config'],'in',train=True);schedule={d:make_schedule(list(range(len(ds))),400,32,0,d) for d,ds in datasets.items()};lock=json.loads((DEST/'experiment/runs/L38/OFF/source_schedules.json').read_text());meta=json.loads((ROOT/'results/sirm_research_loop/iteration_08/source_annotation_audit/selected_source_results.json').read_text());metadata={x['local_id']:x for x in meta['records']};records=[]
 for d,ds in datasets.items():assert list(ds.indices)==lock[str(d)]['split_keys'] and hashlib.sha256(schedule[d].tobytes()).hexdigest()==lock[str(d)]['schedule_sha256']
 for step in STEPS:
  images=[];labels=[];domains=[];weights=[];ids=[];events=[];original_ids=[];guarantee=[]
  for d,ds in sorted(datasets.items()):
   entries=manifest['sources'][str(d)]['images'];ys=np.array([entries[i]['label'] if 'label' in entries[i] else manifest['class_names'].index(entries[i]['id'].split('/')[1]) for i in ds.indices]);counts=np.bincount(ys,minlength=10);assert (counts>=2).all();prob=counts/counts.sum();q=20/32/10+12/32*prob;selected=[(int(i),pos,False) for pos,i in enumerate(schedule[d][step][:12])]
   for c in range(10):
    seed=int(hashlib.sha256(f'coverage-v1/{step}/{d}/{c}'.encode()).hexdigest()[:16],16);rng=np.random.default_rng(seed);draw=rng.choice(np.flatnonzero(ys==c),2,replace=False);selected.extend((int(i),12+2*c+j,True) for j,i in enumerate(draw))
   scheduled=ScheduledImages(ds,0,d)
   for i,pos,guaranteed in selected:
    x,y=scheduled[(i,step,pos)];id=entries[ds.indices[i]]['id'];assert y==ys[i] and id in metadata;images.append(x);labels.append(y);domains.append(d);weights.append(float(prob[y]/q[y]));ids.append(id);events.append(metadata[id]['official_sequence']);original_ids.append(metadata[id]['original_image_id']);guarantee.append(guaranteed)
  with torch.no_grad():_,raw=pooled_raw_features(phi,torch.stack(images).cuda())
  payload=dict(raw=raw.cpu(),y=torch.tensor(labels),domains=torch.tensor(domains),weights=torch.tensor(weights),ids=ids,official_sequences=events,original_image_ids=original_ids,guaranteed_class_draw=guarantee,step=step);file=OUT/f'coverage_step{step}.pt';torch.save(payload,file);records.append(dict(step=step,sha256=digest(file)))
  write(OUT/'progress.json',dict(status='extracting_class_coverage',step=step))
 assert all(torch.equal(v,phi.state_dict()[k].cpu()) for k,v in before.items());assert all(q.grad is None for q in phi.parameters());phi.cpu();del m,phi;gc.collect();torch.cuda.empty_cache();write(OUT/'extraction_audit.json',dict(source_only=True,target_read=False,backbone_unchanged=True,schedules_verified=True,rows=records))

def weighted_class_cost(losses,y,w):
 return torch.stack([(losses[y==c]*w[y==c,None]).sum(0).div(w[y==c].sum()).amin() for c in y.unique()]).mean()

def objectives(m,raw,y,dom,w,coverage):
 g,d,f=graph(m,raw,y,'batch');loss=expert_ce(g[:,None]+d,y);ce=F.cross_entropy(f,y,reduction='none');p=list(m.residuals.parameters());o={'unscaled_discovery':equal_class_cost(loss,y),'importance_CE':torch.stack([(ce[dom==x]*w[dom==x]).sum()/int((dom==x).sum()) for x in dom.unique()]).mean()}
 if coverage:o.update(importance_discovery=weighted_class_cost(loss,y,w),plain_CE=torch.stack([ce[dom==x].mean() for x in dom.unique()]).mean())
 with torch.no_grad():
  _,pi=m.bank_features(raw),m.router(m.bank_features(raw)).softmax(1);_,_,u,_=residual_utilities(g,d,pi,y);Q={}
  for label,ww in [('plain',torch.ones_like(w)),('importance',w)]:
   U=torch.stack([(u[dom==x]*ww[dom==x,None]).sum(0)/int((dom==x).sum()) for x in dom.unique()]);q,_,_,v=soft_null_q(U,torch.ones_like(U,dtype=torch.bool),temperature=.05,utility_floor=.002,interaction_floor=.002);Q[label]=dict(U=U.tolist(),V=v.tolist(),shared_experts=torch.nonzero(q.gt(0).sum(0)>=2).flatten().tolist())
 return {k:vector(v,p) for k,v in o.items()},Q

def run():
 OUT.mkdir(exist_ok=True);plan=dict(source_only=True,target_read=False,primary_training=False,optimizer_updates=False,method_change=False,states=['unscaled14_selected375','unscaled14_last399'],steps=STEPS,radius=RADIUS,cohort='Per domain32: retain first12 exact scheduled natural image/views, add2 distinct source-in image draws per each10 labels. Extra seeds SHA coverage-v1/step/domain/class; deterministic H2 positions12..31.',importance='p(c|domain)/q(c|domain), q=.625/10+.375*p. Fixed component counts make weighted sum/32 an unbiased empirical-domain CE estimator under repeated draws. Within-class weighted normalized mean has finite-sample/self-normalization bias and cannot be asserted exactly equivalent to original stochastic discovery.',finite_controls='Residual-only gradients, detached router; equal parameter-L2 radius. Original source-in calibration before/after perturbations, source-out evaluation, all tensors restored. Norm controls are not Adam or full trajectory equivalence.',limits='No target read. Source-out previously used for selection, and rare original images/events repeat. Coverage is not independent diversity. Fresh utility/Q at saved states is not actual training EMA. Unweighted coverage changes class/domain weighting; report separately, not as matched objective.')
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());extract();token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';audit=json.loads((token/'extraction_audit.json').read_text());data={};cachelocks={}
 for split in ['source_in','source_out']:
  path=token/f'{split}_tokens.npz';assert digest(path)==next(x['cache_sha256'] for x in audit['rows'] if x['split']==split);a=np.load(path);data[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']),a['ids']);cachelocks[split]=digest(path)
 train,ty,td,tids=data['source_in'];val,vy,vd,vids=data['source_out'];assert not set(tids)&set(vids);naturalproof=json.loads((DEST/'source_conditional_H2/extraction_audit.json').read_text());naturalhash={x['step']:x['sha256'] for x in naturalproof['rows']};results=[];locks={}
 for state in ['selected','last']:
  m,p,path=load(state);locks[state]=digest(path);original={k:v.clone() for k,v in m.state_dict().items()};m.calibrate_features(train,td);mean=m.expert_source_mean.clone();params=list(m.residuals.parameters());copies=[q.detach().clone() for q in params];base,bmar,bc,bwrong=metrics(m,val,vy,vd);g,d,f=graph(m,val,vy,'saved');wm=vector(-classmean(margin(f,vy),vy,bwrong),params);cm=vector(-classmean(margin(f,vy),vy,~bwrong),params)
  for step in STEPS:
   for cohort in ['natural','coverage']:
    cache=DEST/'source_conditional_H2'/f'H2_step{step}.pt' if cohort=='natural' else OUT/f'coverage_step{step}.pt';a=torch.load(cache,map_location='cpu',weights_only=False)
    if cohort=='natural':assert digest(cache)==naturalhash[step]
    raw,y,dom=a['raw'],a['y'],a['domains'];w=a['weights'] if cohort=='coverage' else torch.ones(len(y));vecs,Q=objectives(m,raw,y,dom,w,cohort=='coverage');coverage={str(d):{str(c):dict(count=int((dom.eq(d)&y.eq(c)).sum()),unique_ids=len({a['ids'][i] for i in torch.nonzero(dom.eq(d)&y.eq(c)).flatten().tolist()})) for c in range(10)} for d in [0,2,3]};policies={}
    for key,v in vecs.items():
     assert v.norm()>0;move=(-v/v.norm()*RADIUS).float();pos=0
     with torch.no_grad():
      for q,copy in zip(params,copies):n=q.numel();q.copy_(copy+move[pos:pos+n].reshape_as(q));pos+=n
      actual=float(torch.cat([(q-copy).flatten().double() for q,copy in zip(params,copies)]).norm());assert abs(actual-RADIUS)<2e-6
     m.calibrate_features(train,td);s,mar,correct,wrong=metrics(m,val,vy,vd);policies[key]=dict(gradient_norm=float(v.norm()),actual_radius=actual,predicted_wrong_margin_change=float(RADIUS*wm.dot(v)/v.norm()),predicted_correct_margin_change=float(RADIUS*cm.dot(v)/v.norm()),metrics=s,delta={k:s[k]-base[k] for k in base},new_correct_to_wrong=int((bc&~correct).sum()),new_wrong_to_correct=int((~bc&correct).sum()),changed_correctness_ids=[str(vids[i]) for i in torch.nonzero(bc!=correct).flatten().tolist()])
     with torch.no_grad():
      for q,copy in zip(params,copies):q.copy_(copy)
      m.expert_source_mean.copy_(mean)
    results.append(dict(state=state,checkpoint_step=p.get('selected_step',p.get('step')),step=step,cohort=cohort,coverage=coverage,Q=Q,baseline=base,policies=policies,cache_sha256=digest(cache)));write(OUT/'progress.json',dict(status='finite_coverage_controls',state=state,step=step,cohort=cohort,records=len(results)))
  m.load_state_dict(original);assert all(torch.equal(v,m.state_dict()[k]) for k,v in original.items()) and all(q.grad is None for q in m.parameters());assert digest(path)==locks[state]
 write(OUT/'results.json',dict(plan=plan,source_cache_sha256=cachelocks,checkpoint_sha256=locks,code_sha256=digest(Path(__file__)),all_tensors_restored=True,grad_fields_unassigned=True,records=results));write(OUT/'progress.json',dict(status='complete',records=len(results),scientific_goal_achieved=False))
if __name__=='__main__':
 torch.set_num_threads(4);run()
