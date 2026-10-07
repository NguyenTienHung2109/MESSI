"""Locked source-only H2 utility decomposition; no optimizer or gate changes."""
import json,hashlib,gc
from pathlib import Path
import numpy as np
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from domainbed.scripts.train_subset_irm_pacs import make_schedule
from experiments.sirm_discover_gate.data import Adapter
from experiments.sirm_h1_h6.run import ScheduledImages
from experiments.sirm_discover.math import center_logits,residual_utilities,soft_null_q
from .routed_discovery_run import DEST,PRIOR,ROOT,ANCHOR
from .spatial_bank_model import SpatialBankGateSIRM,pooled_raw_features
from .shrinkage import digest,write
OUT=DEST/'source_conditional_H2';STEPS=list(range(200,400,5))
def load(root,state):
 path=root/'experiment/runs/L38/OFF'/('source_best.pkl' if state=='selected' else 'last.pkl');p=torch.load(path,map_location='cpu',weights_only=False);c=p['config'];m=SpatialBankGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c);m.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()});m.configure_stage();m.eval();return m,p,path

def extract():
 OUT.mkdir(exist_ok=True);m,p,path=load(DEST,'selected');manifest=json.loads((DEST/'experiment/subset_manifest.json').read_text());c=p['config'];phi=m.featurizer.cuda().eval();before={k:v.cpu().clone() for k,v in phi.state_dict().items()}
 adapter=Adapter('TerraIncognita');adapter.classes=manifest['class_names'];source=dict(manifest,sources={str(d):manifest['sources'][str(d)] for d in [0,2,3]});datasets=adapter.source_data(ROOT/'domainbed/data',source,c,'in',train=True);lock=json.loads((DEST/'experiment/runs/L38/OFF/source_schedules.json').read_text());schedules={d:make_schedule(list(range(len(ds))),400,32,0,d) for d,ds in datasets.items()}
 for d,ds in datasets.items():assert list(ds.indices)==lock[str(d)]['split_keys'] and hashlib.sha256(schedules[d].tobytes()).hexdigest()==lock[str(d)]['schedule_sha256']
 oldroot=ROOT/'results/sirm_research_loop/iteration_13/source_class_allocation';oldaudit=json.loads((oldroot/'extraction_audit.json').read_text());old={r['step']:r for r in oldaudit['rows']};rows=[]
 for step in STEPS:
  file=OUT/f'H2_step{step}.pt'
  if file.exists():a=torch.load(file,map_location='cpu',weights_only=False)
  else:
   images=[];ys=[];domains=[];ids=[]
   for d,ds in sorted(datasets.items()):
    scheduled=ScheduledImages(ds,0,d)
    for pos,index in enumerate(schedules[d][step]):
     x,y=scheduled[(int(index),step,pos)];images.append(x);ys.append(y);domains.append(d);ids.append(manifest['sources'][str(d)]['images'][ds.indices[int(index)]]['id'])
   with torch.no_grad():_,raw=pooled_raw_features(phi,torch.stack(images).cuda())
   a=dict(raw=raw.cpu(),y=torch.tensor(ys),domains=torch.tensor(domains),ids=ids,step=step,source_only=True);torch.save(a,file)
  assert a['step']==step and len(a['y'])==96 and set(a['domains'].tolist())=={0,2,3};err=None
  if step in old:
   oldfile=oldroot/f'H2_step{step}.pt';assert digest(oldfile)==old[step]['sha256'];b=torch.load(oldfile,map_location='cpu',weights_only=False);assert torch.equal(a['y'],b['y']) and torch.equal(a['domains'],b['domains']);err=float((a['raw']-b['raw']).abs().max());assert err<5e-5
  rows.append(dict(step=step,sha256=digest(file),previous_cache_max_abs_difference=err));write(OUT/'progress.json',dict(status='extracting_source_H2',step=step))
 assert all(torch.equal(v,phi.state_dict()[k].cpu()) for k,v in before.items());assert all(q.grad is None for q in phi.parameters());phi.cpu();del phi,m;gc.collect();torch.cuda.empty_cache()
 write(OUT/'extraction_audit.json',dict(source_only=True,target_read=False,training=False,anchor_sha256=digest(ANCHOR),schedule_verified=True,backbone_unchanged=True,previous_six_caches_verified=True,rows=rows))

def group(u,y,dom,ids,mask):
 counts=[int((mask&dom.eq(d)).sum()) for d in [0,2,3]];unique=[len({ids[i] for i in torch.nonzero(mask&dom.eq(d)).flatten().tolist()}) for d in [0,2,3]];means=[u[mask&dom.eq(d)].mean(0).tolist() if n else None for d,n in zip([0,2,3],counts)];q=None;shared=None
 if all(n>=2 for n in counts):
  U=torch.tensor(means);Q,_,_,V=soft_null_q(U,torch.ones_like(U,dtype=torch.bool),temperature=.05,utility_floor=.002,interaction_floor=.002);q=Q.tolist();shared=torch.nonzero((Q>0).sum(0)>=2).flatten().tolist()
 return dict(counts=counts,unique_image_counts=unique,valid_original_min2=all(n>=2 for n in counts),valid_distinct_min2=all(n>=2 for n in unique),U=means,Q=q,shared_Q_experts=shared)

@torch.no_grad()
def analyze():
 torch.set_num_threads(4);proof=json.loads((OUT/'extraction_audit.json').read_text());rows=[];locks={};aggregate=[]
 for root,label in [(PRIOR,'unscaled'),(DEST,'routed')]:
  for state in ['selected','last']:
   m,p,path=load(root,state);before={k:v.clone() for k,v in m.state_dict().items()};locks[label+'/'+state]=digest(path);all_u=[];all_y=[];all_d=[];all_ids=[];all_wrong=[]
   for record in proof['rows']:
    cache=OUT/f"H2_step{record['step']}.pt";assert digest(cache)==record['sha256'];a=torch.load(cache,map_location='cpu',weights_only=False);raw,y,dom,ids=a['raw'],a['y'],a['domains'],a['ids'];z=m.bank_features(raw);g=m.global_head(raw[:,:384]);pi=m.router(z).softmax(1);delta=center_logits(torch.stack([e(z) for e in m.residuals],1));delta-=delta.mean(0,keepdim=True);f,_,u,_=residual_utilities(g,delta,pi,y);wrong=g.argmax(1).ne(y);all_u.append(u);all_y.append(y);all_d.append(dom);all_ids+=ids;all_wrong.append(wrong)
    groups={'pooled':group(u,y,dom,ids,torch.ones(len(y),dtype=torch.bool)),'global_errors':group(u,y,dom,ids,wrong),'global_correct':group(u,y,dom,ids,~wrong)}
    for cls in range(10):groups[f'class_{cls}']=group(u,y,dom,ids,y.eq(cls))
    rows.append(dict(bank=label,state=state,checkpoint_step=p.get('selected_step',p.get('step')),batch_step=record['step'],groups=groups,repairs=int((wrong&f.argmax(1).eq(y)).sum()),harms=int((~wrong&f.argmax(1).ne(y)).sum())))
   u,y,dom,wrong=map(torch.cat,[all_u,all_y,all_d,all_wrong]);groups={'pooled':group(u,y,dom,all_ids,torch.ones(len(y),dtype=torch.bool)),'global_errors':group(u,y,dom,all_ids,wrong),'global_correct':group(u,y,dom,all_ids,~wrong)}
   for cls in range(10):groups[f'class_{cls}']=group(u,y,dom,all_ids,y.eq(cls))
   aggregate.append(dict(bank=label,state=state,groups=groups,scope='3840 scheduled occurrences, repeated images; means pooled across batches, not original EMA or independent observations.'))
   assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and all(q.grad is None for q in m.parameters());assert digest(path)==locks[label+'/'+state]
 result=dict(source_only=True,target_read=False,training=False,parameter_updates=False,Q_changed=False,threshold_changes=False,checkpoint_sha256=locks,code_sha256=digest(Path(__file__)),centering='actual batch centering over96 training images',scope='Four saved states replayed on40 exact scheduled source H2 refresh batches; not historical states or historical EMA. Groups use known source labels/frozen-global errors for diagnosis, not deployable gates. Counts are occurrences and distinct IDs both reported. Min2 is diagnostic component count, not a replacement for utility EMA min32.',rows=rows,aggregate=aggregate)
 write(OUT/'results.json',result);write(OUT/'progress.json',dict(status='complete',saved_states=4,batches=40,rows=len(rows),target_read=False))
if __name__=='__main__':
 torch.set_num_threads(4);extract();analyze()
