"""Counterfactual strength, tie symmetry, immutable global and actual SIRM path."""
import unittest
from unittest.mock import patch
import torch
from experiments.sirm_discover.math import multiple_choice_loss
from .class_balanced_model import equal_class_cost,ClassBalancedGateSIRM
from .class_coherent_discovery import class_coherent_cost
from .test_class_coherent import ClassCoherentChecks
from .test_centered import CenteringChecks

class ClassBalancedChecks(unittest.TestCase):
 def model(self,on=False):
  base=ClassCoherentChecks().model(on);c=dict(base.config,discovery_class_weighting='equal_class_original_gradient_norm');m=ClassBalancedGateSIRM(torch.nn.Linear(4,4),4,3,3,c);m.load_state_dict(base.state_dict());return m
 def test_rare_class_weight_and_ties(self):
  v=torch.tensor([[1.,3.],[1.,3.],[1.,3.],[4.,2.]],dtype=torch.double,requires_grad=True);y=torch.tensor([0,0,0,1]);a=class_coherent_cost(v,y);b=equal_class_cost(v,y);self.assertEqual(float(a.detach()),1.25);self.assertEqual(float(b.detach()),1.5)
  grad=torch.autograd.grad(b,v)[0];self.assertTrue(torch.allclose(grad,torch.tensor([[1/6,0.],[1/6,0.],[1/6,0.],[0.,.5]],dtype=torch.double)))
  ties=torch.ones(4,2,dtype=torch.double,requires_grad=True);grad=torch.autograd.grad(equal_class_cost(ties,y),ties)[0];self.assertTrue(torch.allclose(grad,torch.tensor([[1/12,1/12]]*3+[[.25,.25]],dtype=torch.double)))
 def test_permutation_and_singletons(self):
  torch.manual_seed(123);v=torch.randn(11,6,dtype=torch.double);y=torch.arange(11)%3;p=torch.randperm(11);self.assertTrue(torch.allclose(equal_class_cost(v,y),equal_class_cost(v[p],(2-y)[p])));self.assertTrue(torch.allclose(equal_class_cost(v,torch.arange(11)),v.amin(1).mean()))
 def test_norm_matched_update_frozen_pair_and_original_balance(self):
  a=self.model();b=self.model(True);CenteringChecks().nonzero_heads(a);b.load_state_dict(a.state_dict());frozen={k:v.clone() for k,v in a.state_dict().items() if k.startswith(('featurizer.','global_head.'))}
  for step in range(3):
   torch.manual_seed(321+step);batches=[(torch.randn(8,4),torch.tensor([0]*5+[1]*2+[2])) for _ in range(3)]
   with patch.object(a.optimizer,'step',wraps=a.optimizer.step) as spy:ra=a.update(batches);self.assertEqual(spy.call_count,1)
   rb=b.update(batches);self.assertEqual(ra,rb);self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
   info=a.last_diagnostics['discovery_gradient_norm_match'];self.assertAlmostEqual(info['original_gradient_norm'],info['matched_gradient_norm'],places=7);self.assertGreater(info['equal_class_gradient_norm'],0)
   self.assertAlmostEqual(ra['loss_discover'],a.last_diagnostics['discovery_class_loss_replay'],places=6)
  self.assertTrue(all(torch.equal(v,a.state_dict()[k]) for k,v in frozen.items()));self.assertTrue(all(not p.requires_grad and p.grad is None for mod in [a.featurizer,a.global_head] for p in mod.parameters()))
 def test_eligible_sirm_path_is_preserved(self):
  m=self.model(True);CenteringChecks().nonzero_heads(m);m.step.fill_(251);m.q_learned.zero_();m.q_learned[0,0]=m.q_learned[1,0]=1.
  torch.manual_seed(987);r=m.update([(torch.randn(12,4),torch.arange(12)%3) for _ in range(3)]);self.assertTrue(r['sirm_active']);self.assertGreater(r['loss_sirm'],0);self.assertEqual(r['effective_lambda_sirm'],1.)
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
