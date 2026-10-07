"""Exact locked source H2 schedule replay at saved states, no parameter update."""
import hashlib,json
from pathlib import Path
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from domainbed.scripts.train_subset_irm_pacs import make_schedule
from experiments.sirm_discover_gate.data import Adapter
from experiments.sirm_h1_h6.run import ScheduledImages
from experiments.sirm_discover.math import center_logits
from .centered_model import CenteredGateSIRM
from .hard_discovery import hard_winner_loss
from .pair_shared_discovery import paired_shared_loss
from .shrinkage import ROOT,digest,write
B=ROOT/'results/sirm_research_loop/iteration_08'
def flatten(gs):return torch.cat([g.detach().double().flatten() for g in gs])
def run():
 torch.set_num_threads(4);out=B/'pair_shared_parameter_audit';out.mkdir(exist_ok=True);bank=B/'experiment/runs/L38/OFF';paths={'selected':bank/'source_best.pkl','last':bank/'last.pkl'};steps=[200,215,250,300,350,395];mp=B/'experiment/subset_manifest.json';manifest=json.loads(mp.read_text());plan={'code_sha256':digest(Path(__file__)),'candidate_code_sha256':digest(Path(__file__).with_name('pair_shared_discovery.py')),'checkpoint_hashes':{s:digest(p) for s,p in paths.items()},'manifest_sha256':digest(mp),'steps':steps,'scope':'Saved selected375/final399 residual parameter gradients on exact locked H2 source batches; source0/2/3 only, no target, optimizer, .grad assignment, selection or historical-gradient claim. Differentiable equal-source batch centering, same as training.','strength_control_candidate':'Per-batch detached residual-parameter hard/pair gradient norm ratio, keeping lambda_discover unchanged; same norm at own current state, not identical gradient direction or trajectory.'};write(out/'plan.json',plan);models={}
 for s,path in paths.items():
  payload=torch.load(path,map_location='cpu',weights_only=False);c=payload['config'];phi=DeiTFeaturizer(pretrained=False,model_name=c['backbone']);model=CenteredGateSIRM(phi,384,10,3,c);model.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});model.configure_stage();models[s]=model
 phi=models['selected'].featurizer.cuda().eval();adapter=Adapter('TerraIncognita');adapter.classes=manifest['class_names'];source_manifest=dict(manifest,sources={str(d):manifest['sources'][str(d)] for d in [0,2,3]});datasets=adapter.source_data(ROOT/'domainbed/data',source_manifest,c,'in',train=True);schedules={d:make_schedule(list(range(len(ds))),400,32,0,d) for d,ds in datasets.items()};lock=json.loads((bank/'source_schedules.json').read_text())
 for d,ds in datasets.items():assert list(ds.indices)==lock[str(d)]['split_keys'] and hashlib.sha256(schedules[d].tobytes()).hexdigest()==lock[str(d)]['schedule_sha256']
 records=[]
 for step in steps:
  images=[];labels=[];domains=[]
  for d,ds in sorted(datasets.items()):
   scheduled=ScheduledImages(ds,0,d)
   for position,index in enumerate(schedules[d][step]):
    x,y=scheduled[(int(index),step,position)];images.append(x);labels.append(y);domains.append(d)
  with torch.no_grad():z=phi(torch.stack(images).cuda()).cpu()
  y=torch.tensor(labels);dom=torch.tensor(domains);torch.save({'z':z,'y':y,'domains':dom,'scheduled_step':step},out/f'H2_step{step}.pt')
  for state,m in models.items():
   g=m.global_head(z).detach();raw=center_logits(torch.stack([e(z) for e in m.residuals],1));delta=raw-raw.mean(0,keepdim=True);hard,_=hard_winner_loss(g,delta,y,.25);pair,info=paired_shared_loss(g,delta,y,dom);params=list(m.residuals.parameters());gh=flatten(torch.autograd.grad(hard,params,retain_graph=True));gp=flatten(torch.autograd.grad(pair,params,retain_graph=True));nh=gh.norm();np_=gp.norm();scale=(nh/np_).detach() if np_>0 else np_.new_tensor(0.);matched=flatten(torch.autograd.grad(pair*scale.to(pair.dtype),params));assert torch.allclose(matched.norm(),nh,rtol=1e-5,atol=1e-8);assert all(p.grad is None for p in m.parameters());records.append({'state':state,'scheduled_step':step,'hard_grad_norm':float(nh),'pair_grad_norm':float(np_),'pair_hard_ratio':float(np_/nh),'matched_scale':float(scale),'gradient_cosine':float(torch.nn.functional.cosine_similarity(gh,gp,dim=0)),'matched_grad_norm':float(matched.norm()),'matched_relative_norm_error':float(abs(matched.norm()-nh)/nh),'pair_count':len(info['pairs']),'paired_classes':info.get('paired_classes',0),'fallback':info['fallback']})
  write(out/'progress.json',{'status':'source_schedule_replay','step':step,'records':len(records)})
 assert all(digest(p)==plan['checkpoint_hashes'][s] for s,p in paths.items());write(out/'results.json',{'plan':plan,'records':records,'summary':{'ratio_median':float(torch.tensor([r['pair_hard_ratio'] for r in records]).median()),'ratio_min':min(r['pair_hard_ratio'] for r in records),'ratio_max':max(r['pair_hard_ratio'] for r in records),'max_matched_relative_norm_error':max(r['matched_relative_norm_error'] for r in records)}});write(out/'progress.json',{'status':'complete','records':len(records),'parameter_updates':False,'target_read':False});print(records)
if __name__=='__main__':run()
