"""Candidate15 real-H2 disposable update versus iteration14 equal weighting; no primary training."""
import copy,json,hashlib
from pathlib import Path
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from domainbed.scripts.train_subset_irm_pacs import make_schedule
from experiments.sirm_discover_gate.data import Adapter
from experiments.sirm_h1_h6.run import ScheduledImages
from .spatial_bank_model import SpatialBankGateSIRM
from .class_balanced_model import ClassBalancedSpatialGateSIRM
from .routed_discovery_model import RoutedDiscoverySpatialGateSIRM
from .anchor_train import load_anchor
from .spatial_bank_run import ROOT,DEST,ANCHOR
from .shrinkage import digest,write
OUT=ROOT/'results/sirm_research_loop/iteration_15'
def paramhash(m):
 h=hashlib.sha256()
 for name,p in m.named_parameters():h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 old=json.loads((ROOT/'results/sirm_research_loop/iteration_14/plan.json').read_text());assert json.loads((DEST/'final_analysis_audit.json').read_text())['analysis_complete'];assert json.loads((DEST/'source_class_allocation/completion_audit.json').read_text())['complete'];payload=torch.load(ANCHOR,map_location='cpu',weights_only=False);cache=torch.load(DEST/'source_feature_cache.pt',weights_only=False);models=[];initial=[];configs=[]
 for kind in ['original','balanced_OFF','balanced_ON']:
  c=copy.deepcopy(old['hparams']['ON' if kind=='balanced_ON' else 'OFF']['sirm_discover_config'])
  if kind!='original':c.update(protocol='sirm_research_iter15_routed_discovery_norm_match',discovery_class_weighting='equal_class_routed_original_gradient_norm')
  torch.manual_seed(0);m=(ClassBalancedSpatialGateSIRM if kind=='original' else RoutedDiscoverySpatialGateSIRM)(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c);load_anchor(m,payload);m.configure_stage();initial.append(paramhash(m));m=m.cuda();m._source_features=cache['features'];m._source_domains=cache['domains'];m.runtime_calibration=True;models.append(m);configs.append(c)
 assert len(set(initial))==1
 expected=json.loads((DEST/'experiment/runs/L38/OFF/spatial_initialization.json').read_text())['parameter_sha256'];assert initial[0]==expected
 assert all(torch.equal(v,models[2].state_dict()[k]) for k,v in models[1].state_dict().items())
 frozen={k:v.cpu().clone() for k,v in models[0].state_dict().items() if k.startswith(('featurizer.','global_head.'))}
 manifest=json.loads((DEST/'experiment/subset_manifest.json').read_text());adapter=Adapter('TerraIncognita');adapter.classes=manifest['class_names'];source=dict(manifest,sources={str(d):manifest['sources'][str(d)] for d in [0,2,3]});datasets=adapter.source_data(ROOT/'domainbed/data',source,configs[0],'in',train=True);lock=json.loads((DEST/'experiment/runs/L38/OFF/source_schedules.json').read_text());batches=[]
 for d,ds in sorted(datasets.items()):
  schedule=make_schedule(list(range(len(ds))),400,32,0,d);assert list(ds.indices)==lock[str(d)]['split_keys'] and hashlib.sha256(schedule.tobytes()).hexdigest()==lock[str(d)]['schedule_sha256'];scheduled=ScheduledImages(ds,0,d);xy=[scheduled[(int(i),200,pos)] for pos,i in enumerate(schedule[200])];batches.append((torch.stack([x for x,y in xy]).cuda(),torch.tensor([y for x,y in xy]).cuda()))
 rows=[];norms=[]
 for m in models:
  rows.append(m.update(batches));m.eval();assert all(torch.equal(v,m.state_dict()[k].cpu()) for k,v in frozen.items());assert all(not p.requires_grad and p.grad is None for mod in [m.featurizer,m.global_head] for p in mod.parameters());norms.append(m.last_diagnostics.get('discovery_gradient_norm_match'))
 assert rows[1]==rows[2] and all(torch.equal(v,models[2].state_dict()[k]) for k,v in models[1].state_dict().items())
 assert norms[1] and norms[1]['original_gradient_norm']>0 and norms[1]['routed_gradient_norm']>0
 assert abs(norms[1]['matched_gradient_norm']/norms[1]['original_gradient_norm']-1)<2e-5
 write(OUT/'prelaunch_integration_audit.json',{'passed':True,'source_only':True,'target_read':False,'primary_training_started':False,'diagnostic_updates':1,'source_scheduled_H2_batch_step':200,'source_images':96,'anchor_sha256':digest(ANCHOR),'initial_parameter_hash':initial[0],'exact_iteration13_initialization':True,'balanced_OFF_ON_all_state_exact_initial_and_after_update':True,'frozen_anchor_unchanged':True,'norm_match':norms[1],'observed_rows':rows,'code_sha256':digest(Path(__file__)),'model_code_sha256':digest(Path(__file__).with_name('routed_discovery_model.py'))})
 for m in models[1:]:
  info=m.last_diagnostics['discovery_gradient_norm_match'];reference=m.last_diagnostics['discovery_reference_class_blocks'];blocks=m.last_diagnostics['discovery_class_blocks']
  original=sum(min(b['mean_expert_CE'])*b['count']/96 for b in reference);routed=sum(min(b['mean_expert_CE']) for b in blocks)/len(blocks)
  assert abs(original-info['original_loss'])<2e-6 and abs(routed-info['routed_loss'])<2e-6
  assert m.last_diagnostics['discovery_router_gradient_absent']
 write(OUT/'prelaunch_logging_audit.json',{'passed':True,'scope':'Actual96-image H2 objective replay from separate unscaled-reference and routed class blocks; discovery-router gradients independently checked absent in update.','original_reference_loss':original,'routed_loss':routed,'observed_diagnostics':models[1].last_diagnostics})
 print('Real scheduled96-image H2 update passed; exact original initialization, frozen anchor and OFF/ON equality; norm match',norms[1],flush=True)
if __name__=='__main__':main()
