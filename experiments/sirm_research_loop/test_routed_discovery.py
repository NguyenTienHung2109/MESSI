"""Candidate disposable updates: same anchor, one step, real penalty path."""
import unittest
from unittest.mock import patch
import torch
from .routed_discovery_model import RoutedDiscoveryGateSIRM
from .test_class_balanced import ClassBalancedChecks
from .test_centered import CenteringChecks

class RoutedChecks(ClassBalancedChecks):
    def model(self,on=False):
        base=super().model(on);c=dict(base.config,discovery_class_weighting='equal_class_routed_original_gradient_norm');m=RoutedDiscoveryGateSIRM(torch.nn.Linear(4,4),4,3,3,c);m.load_state_dict(base.state_dict());return m
    def test_norm_matched_update_frozen_pair_and_original_balance(self):
        a=self.model();b=self.model(True);CenteringChecks().nonzero_heads(a);b.load_state_dict(a.state_dict());frozen={k:v.clone() for k,v in a.state_dict().items() if k.startswith(('featurizer.','global_head.'))}
        for step in range(3):
            torch.manual_seed(321+step);batches=[(torch.randn(8,4),torch.tensor([0]*5+[1]*2+[2])) for _ in range(3)]
            with patch.object(a.optimizer,'step',wraps=a.optimizer.step) as spy:ra=a.update(batches);self.assertEqual(spy.call_count,1)
            rb=b.update(batches);self.assertEqual(ra,rb);self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
            info=a.last_diagnostics['discovery_gradient_norm_match'];self.assertAlmostEqual(info['original_gradient_norm'],info['matched_gradient_norm'],places=7);self.assertGreater(info['routed_gradient_norm'],0);self.assertTrue(a.last_diagnostics['discovery_routing_detached'])
            self.assertAlmostEqual(ra['loss_discover'],a.last_diagnostics['discovery_class_loss_replay'],places=6)
        self.assertTrue(all(torch.equal(v,a.state_dict()[k]) for k,v in frozen.items()));self.assertTrue(all(not p.requires_grad and p.grad is None for mod in [a.featurizer,a.global_head] for p in mod.parameters()))
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
