"""Source-only disposable actual update audit, no primary training."""
import copy,json
from pathlib import Path
from unittest.mock import patch
import torch
from .routed_conditional_utility import load
from .correct_guard_model import CorrectGuardSpatialGateSIRM
from .direct_router_risk_model import DirectRouterRiskSpatialGateSIRM
from .shrinkage import ROOT,write,digest
PRIOR=ROOT/'results/sirm_research_loop/iteration_16';DEST=PRIOR.parent/'iteration_17'
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 DEST.mkdir(exist_ok=True);assert not (DEST/'prelaunch_integration_audit.json').exists()
 old=json.loads((PRIOR/'final_analysis_audit.json').read_text())
 for f,h in old['sha256'].items():assert digest(PRIOR/f)==h,f
 cache=ROOT/'results/sirm_research_loop/iteration_15/source_conditional_H2/H2_step200.pt';a=torch.load(cache,weights_only=False,map_location='cpu');batch=[(a['raw'][i*32:(i+1)*32],a['y'][i*32:(i+1)*32]) for i in range(3)]
 # At nonzero expert weights, isolate auxiliary objective versus old update.
 m,p,path=load(PRIOR,'selected');saved=copy.deepcopy(m.state_dict());frozen={k:v.clone() for k,v in saved.items() if k.startswith(('featurizer.','global_head.'))};rows=[];states=[]
 for cls in [CorrectGuardSpatialGateSIRM,DirectRouterRiskSpatialGateSIRM,DirectRouterRiskSpatialGateSIRM]:
  q=copy.deepcopy(m);q.__class__=cls;q._cap_recording=False;q._cap_forward=None;q.step.fill_(199);q.configure_stage()
  with patch('experiments.sirm_research_loop.spatial_bank_model.pooled_raw_features',lambda featurizer,x:(x[:,:384],x)):
   rows.append(q.update(batch))
  states.append(copy.deepcopy(q.state_dict()));assert all(torch.equal(v,q.state_dict()[k]) for k,v in frozen.items());assert all(not v.requires_grad and v.grad is None for module in [q.featurizer,q.global_head] for v in module.parameters())
  if cls is DirectRouterRiskSpatialGateSIRM:
   assert q.last_diagnostics['router_objective_residual_gradient_absent'];assert q.last_diagnostics['router_probability_replay_max_error']<1e-6
   assert abs(q.last_diagnostics['router_aux_raw_loss']-q.last_diagnostics['router_objective_raw_CE'])<1e-6
   assert q.last_diagnostics['router_aux_post_cap_parameter_grad_norm']<=q.last_diagnostics['router_cls_parameter_grad_norm']*(1+1e-5)+1e-10
 assert rows[1]==rows[2] and all(torch.equal(v,states[2][k]) for k,v in states[1].items())
 assert rows[0]['loss_router']!=rows[1]['loss_router'];assert rows[0]['loss_discover']==rows[1]['loss_discover'];assert rows[0]['loss_cls']==rows[1]['loss_cls'];assert rows[0]['loss_balance']==rows[1]['loss_balance']
 assert any(not torch.equal(v,states[1][k]) for k,v in states[0].items() if k.startswith('router.'))
 assert digest(path)==old['sha256'][str(path.relative_to(PRIOR))]
 write(DEST/'prelaunch_integration_audit.json',dict(passed=True,source_only=True,target_read=False,primary_training_started=False,scope='Three disposable CPU updates at saved selected200 expert state with step199; same cached scheduled96-image H2 batch. Not fresh initialization or historical gradients.',frozen_anchor_unchanged=True,new_update_deterministic=True,nonrouter_scalar_losses_equal=True,auxiliary_residual_gradients_absent=True,router_cap_preserved=True,original_bank_hash_unchanged=True,rows=rows,code_sha256=digest(Path(__file__)),model_code_sha256=digest(Path(__file__).with_name('direct_router_risk_model.py'))))
 print('Passed actual update audit',rows,flush=True)
if __name__=='__main__':main()
