"""Source-only saved-state discovery allocation and class-weight gradient counterfactuals."""
import gc,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from domainbed.deit_transformer import DeiTFeaturizer
from domainbed.scripts.train_subset_irm_pacs import make_schedule
from experiments.sirm_discover_gate.data import Adapter
from experiments.sirm_h1_h6.run import ScheduledImages
from experiments.sirm_discover.math import center_logits,expert_ce
from .spatial_bank_model import SpatialBankGateSIRM,pooled_raw_features
from .class_coherent_discovery import class_coherent_cost
from .discovery_error_gradient_diagnosis import vector
from .spatial_bank_run import ROOT,DEST,PRIOR
from .shrinkage import digest,write
OUT=DEST/'source_class_allocation';STEPS=[200,215,250,300,350,395]
def load(arm,state):
 path=DEST/'experiment/runs/L38'/arm/('source_best.pkl' if state=='selected' else 'last.pkl');p=torch.load(path,map_location='cpu',weights_only=False);c=p['config']
 m=SpatialBankGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c);m.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()});m.configure_stage();m.eval()
 assert all(not q.requires_grad for name in ['featurizer','global_head'] for q in getattr(m,name).parameters())
 return m,p,path

def extract():
 m,p,path=load('OFF','selected');manifest=json.loads((DEST/'experiment/subset_manifest.json').read_text());c=p['config'];phi=m.featurizer.cuda().eval()
 adapter=Adapter('TerraIncognita');adapter.classes=manifest['class_names'];source_manifest=dict(manifest,sources={str(d):manifest['sources'][str(d)] for d in [0,2,3]})
 datasets=adapter.source_data(ROOT/'domainbed/data',source_manifest,c,'in',train=True);schedules={d:make_schedule(list(range(len(ds))),400,32,0,d) for d,ds in datasets.items()};lock=json.loads((DEST/'experiment/runs/L38/OFF/source_schedules.json').read_text())
 for d,ds in datasets.items():assert list(ds.indices)==lock[str(d)]['split_keys'] and hashlib.sha256(schedules[d].tobytes()).hexdigest()==lock[str(d)]['schedule_sha256']
 rows=[]
 for step in STEPS:
  images=[];labels=[];domains=[]
  for d,ds in sorted(datasets.items()):
   scheduled=ScheduledImages(ds,0,d)
   for position,index in enumerate(schedules[d][step]):
    x,y=scheduled[(int(index),step,position)];images.append(x);labels.append(y);domains.append(d)
  with torch.no_grad():cls,raw=pooled_raw_features(phi,torch.stack(images).cuda());raw=raw.cpu();cls=cls.cpu()
  y=torch.tensor(labels);dom=torch.tensor(domains);oldpath=ROOT/'results/sirm_research_loop/iteration_08/pair_shared_parameter_audit'/f'H2_step{step}.pt';old=torch.load(oldpath,map_location='cpu',weights_only=False)
  assert torch.equal(y,old['y']) and torch.equal(dom,old['domains']);err=float((cls-old['z']).abs().max());assert err<5e-5
  file=OUT/f'H2_step{step}.pt';payload={'raw':raw,'y':y,'domains':dom,'scheduled_step':step,'target_read':False,'checkpoint_sha256':digest(path)}
  if file.exists():
   b=torch.load(file,weights_only=False);assert all(torch.equal(payload[k],b[k]) for k in ['raw','y','domains'])
  else:torch.save(payload,file)
  rows.append({'step':step,'sha256':digest(file),'legacy_CLS_sha256':digest(oldpath),'max_abs_CLS_difference':err,'labels_domains_exact':True})
  write(OUT/'progress.json',{'status':'extracting_exact_source_H2','step':step})
 phi.cpu();del m,phi;gc.collect();torch.cuda.empty_cache();write(OUT/'extraction_audit.json',{'target_read':False,'frozen_anchor_sha256':p['config']['strong_anchor_sha256'],'schedule_verified':True,'rows':rows})

def cos(a,b):return float(F.cosine_similarity(a,b,dim=0)) if a.norm() and b.norm() else None

def assess(m,raw,y,domains,centering,reference=None,class_reference=None):
 z=m.bank_features(raw)
 with torch.no_grad():g=m.global_head(raw[:,:384]);pi=m.router(z).softmax(1)
 delta=center_logits(torch.stack([e(z) for e in m.residuals],1));delta=delta-(delta.mean(0,keepdim=True) if centering=='batch' else m.expert_source_mean[None])
 losses=expert_ce(g[:,None]+delta,y);full=g+(pi[:,:6,None]*delta).sum(1);params=list(m.residuals.parameters());labels=y.unique(sorted=True);objectives={};groups=[];vectors={};wrong=g.argmax(1).ne(y)
 for label in labels:
  mask=y.eq(label);means=losses[mask].mean(0);cost=means.amin();n=int(mask.sum());original=n/len(y)*cost;balanced=cost/len(labels)
  objectives[int(label)]=(original,balanced);v=vector(m.config['lambda_discover']*original,params);vectors[int(label)]=v
  groups.append({'label':int(label),'n':n,'class_image_weight':n/len(y),'equal_class_weight':1/len(labels),'global_errors':int((mask&wrong).sum()),'domain_counts':{str(int(d)):int((mask&domains.eq(d)).sum()) for d in domains.unique()},'mean_expert_CE':means.detach().tolist(),'winner_experts':means.eq(means.amin()).nonzero().flatten().tolist(),'weighted_original_gradient_norm':float(v.norm()),'cosine_to_source_out_CE':cos(v,reference) if reference is not None else None,'cosine_to_same_class_source_out_CE':cos(v,class_reference[int(label)]) if class_reference is not None and int(label) in class_reference else None})
 original=sum(t[0] for t in objectives.values());balanced=sum(t[1] for t in objectives.values());assert torch.allclose(original,class_coherent_cost(losses,y),atol=1e-6)
 vs={'original':vector(m.config['lambda_discover']*original,params),'equal_class':vector(m.config['lambda_discover']*balanced,params)};assert torch.allclose(sum(vectors.values()),vs['original'],atol=1e-7,rtol=1e-5)
 cls=torch.stack([F.cross_entropy(full[domains.eq(d)],y[domains.eq(d)]) for d in domains.unique()]).mean();cv=vector(cls,params)
 grads={k:{'loss':float((original if k=='original' else balanced).detach()),'weighted_residual_gradient_norm':float(v.norm()),'cosine_to_local_equal_domain_CE':cos(v,cv),'cosine_to_source_out_CE':cos(v,reference) if reference is not None else None,'cosine_to_original':cos(v,vs['original'])} for k,v in vs.items()}
 for row in groups:
  v=vectors[row['label']];row['projection_fraction_of_original_gradient']=float(v.dot(vs['original'])/vs['original'].dot(vs['original'])) if vs['original'].norm() else None
 classrefs={}
 if reference is None:
  for label in labels:
   domains_here=domains[y.eq(label)].unique();loss=torch.stack([F.cross_entropy(full[y.eq(label)&domains.eq(d)],y[y.eq(label)&domains.eq(d)]) for d in domains_here]).mean();classrefs[int(label)]=vector(loss,params)
 return {'n':len(y),'centering':centering,'classes':groups,'gradients':grads,'residual_classification_gradient_norm':float(cv.norm()),'all_class_gradient_additivity_max_error':float((sum(vectors.values())-vs['original']).abs().max())},cv,classrefs

def main():
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);OUT.mkdir(exist_ok=True)
 final=json.loads((DEST/'final_analysis_audit.json').read_text());assert final['analysis_complete']
 plan=json.loads((DEST/'plan.json').read_text());assert all(digest(ROOT/f)==h for k in ['base_code_hashes','variant_code_hashes'] for f,h in plan[k].items())
 write(OUT/'plan.json',{'code_sha256':digest(Path(__file__)),'source_only':True,'target_read':False,'training':False,'fitting':False,'updates':False,'steps':STEPS,'hypothesis':'Outer class-image weighting may allocate discovery signal away from difficult classes; compare original and equal-class gradient direction at saved selected/common399 states, before proposing any objective change.','scope':'Fixed source-in/out original views and six exact scheduled source H2 batches, selected and399 CLS/spatial banks. Saved-state gradients, not historical training gradients. Equal-class weighting is diagnostic only, not a selected model. Shared held-out source selection limits independence; gradients residual-only, router probabilities detached as controlled diagnostic.'})
 extract();cache={};audit=json.loads((PRIOR/'frozen_token_source_diagnosis/extraction_audit.json').read_text())
 for split in ['source_in','source_out']:
  file=PRIOR/'frozen_token_source_diagnosis'/f'{split}_tokens.npz';assert digest(file)==next(r['cache_sha256'] for r in audit['rows'] if r['split']==split)
  with np.load(file) as a:cache[split]=[torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains'])]
 records=[];locks={}
 for arm in ['CLS_CONTROL','OFF']:
  for state in ['selected','last']:
   m,p,path=load(arm,state);before={k:v.clone() for k,v in m.state_dict().items()};locks[f'{arm}/{state}']={'sha256':digest(path),'step':p.get('selected_step',p.get('step'))}
   val,ref,classref=assess(m,*cache['source_out'],'saved');records.append({'arm':arm,'state':state,'split':'source_out','view':'original','metrics':val})
   train,_,_=assess(m,*cache['source_in'],'saved',ref,classref);records.append({'arm':arm,'state':state,'split':'source_in','view':'original','metrics':train})
   for step in STEPS:
    b=torch.load(OUT/f'H2_step{step}.pt',weights_only=False)
    for centering in ['saved','batch']:
     metric,_,_=assess(m,b['raw'],b['y'],b['domains'],centering,ref,classref);records.append({'arm':arm,'state':state,'split':'source_in_scheduled','view':'H2','batch_step':step,'metrics':metric})
    write(OUT/'progress.json',{'status':'saved_state_gradient_diagnosis','arm':arm,'state':state,'batch_step':step,'records':len(records)})
   assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert all(q.grad is None for q in m.parameters());del m,before;gc.collect()
 write(OUT/'results.json',{'source_only':True,'target_read':False,'training':False,'fitting':False,'parameter_updates':False,'state_tensors_unchanged':True,'grad_fields_none':True,'checkpoint_locks':locks,'code_sha256':digest(Path(__file__)),'records':records})
 write(OUT/'progress.json',{'status':'complete','records':len(records),'remaining':'Interpret source allocation and gradient alignment; no candidate promoted.'});print('Completed',len(records),'source-only saved-state records',flush=True)
if __name__=='__main__':main()
