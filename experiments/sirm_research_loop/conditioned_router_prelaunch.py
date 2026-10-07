"""Disposable CPU integration audit; no primary training or target reads."""
import copy,json
from pathlib import Path
from unittest.mock import patch
import torch
from .routed_conditional_utility import load
from .conditioned_router_model import ConditionedRouterSpatialGateSIRM
from .shrinkage import ROOT,digest,write
PRIOR=ROOT/'results/sirm_research_loop/iteration_18';DEST=PRIOR.parent/'iteration_19'
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 DEST.mkdir(exist_ok=True);assert not (DEST/'prelaunch_integration_audit.json').exists()
 audit=json.loads((PRIOR/'final_analysis_audit.json').read_text());assert audit['analysis_complete']
 for rel,h in audit['sha256'].items():assert digest(PRIOR/rel)==h,rel
 from . import routed_conditional_utility as loader
 from .confidence_router_model import ConfidenceRouterSpatialGateSIRM
 loader.SpatialBankGateSIRM=ConfidenceRouterSpatialGateSIRM
 old,p,path=load(PRIOR,'selected');c=dict(p['config']);c.update(protocol='sirm_iter19_conditioned_confidence_router',router_optimizer_lr=.01,router_descriptor_std_floor=.01)
 m=ConditionedRouterSpatialGateSIRM(copy.deepcopy(old.featurizer),384,10,3,c)
 state={k:v for k,v in old.state_dict().items() if not k.startswith('router.')};missing=m.load_state_dict(state,strict=False);assert not missing.unexpected_keys;assert set(missing.missing_keys)=={'router.head.weight','router.head.bias','router.global_logit_unit','router.descriptor_mean','router.descriptor_std','router.descriptor_calibration_step'}
 m.step.fill_(199);m.configure_stage();m.calibrate_features(*m._descriptor_source_cache);frozen={k:v.clone() for k,v in m.state_dict().items() if k.startswith(('featurizer.','global_head.'))}
 cache=ROOT/'results/sirm_research_loop/iteration_15/source_conditional_H2/H2_step200.pt';a=torch.load(cache,map_location='cpu',weights_only=False);batch=[(a['raw'][i*32:(i+1)*32],a['y'][i*32:(i+1)*32]) for i in range(3)]
 for mode in [True,False]:
  m.train(mode);z=m.bank_features(a['raw']);out=m.router(z);assert out.shape==(96,7) and torch.isfinite(out).all();assert torch.equal(out,torch.zeros_like(out))
  deriv=torch.autograd.grad(out.sum(),list(m.residuals.parameters()),allow_unused=True);assert all(v is None for v in deriv)
  recon=z*m.bank_source_feature_std*2**.5+m.bank_source_feature_mean;err=float((recon-a['raw']).abs().max());assert err<1e-4
 # deepcopy must point at its own experts, not the original owner's modules.
 rows=[];states=[]
 for _ in range(2):
  q=copy.deepcopy(m);assert q.router.owner is q;q.train();q.step.fill_(199);q.configure_stage()
  with patch('experiments.sirm_research_loop.spatial_bank_model.pooled_raw_features',lambda featurizer,x:(x[:,:384],x)):rows.append(q.update(batch))
  assert q.last_diagnostics['router_objective_residual_gradient_absent'];assert q.last_diagnostics['router_probability_replay_max_error']<1e-6
  assert all(not pp.requires_grad and pp.grad is None for mod in [q.featurizer,q.global_head] for pp in mod.parameters());assert all(torch.equal(v,q.state_dict()[k]) for k,v in frozen.items())
  assert set(id(pp) for group in q.optimizer.param_groups for pp in group['params'])==set(id(pp) for pp in q.parameters())
  states.append(copy.deepcopy(q.state_dict()))
 assert rows[0]==rows[1] and all(torch.equal(v,states[1][k]) for k,v in states[0].items())
 assert any(not torch.equal(v,m.state_dict()[k]) for k,v in states[0].items() if k.startswith('router.head.'))
 assert digest(path)==audit['sha256'][str(path.relative_to(PRIOR))]
 write(DEST/'prelaunch_integration_audit.json',dict(passed=True,source_only=True,target_read=False,primary_training_started=False,scope='Two disposable deterministic updates on saved selected200 nonzero experts at step199, plus train/eval descriptor tests. Not historical gradients or a trained new method.',frozen_anchor_unchanged=True,descriptor_residual_gradients_absent=True,auxiliary_residual_gradients_absent=True,uniform_initial_router=True,deepcopy_owner_correct=True,optimizer_parameter_identity_verified=True,raw_inverse_max_error=err,rows=rows,cache_sha256=digest(cache),model_code_sha256=digest(Path(__file__).with_name('conditioned_router_model.py')),code_sha256=digest(Path(__file__))))
 print('Passed conditioned confidence-router disposable integration',flush=True)
if __name__=='__main__':main()
