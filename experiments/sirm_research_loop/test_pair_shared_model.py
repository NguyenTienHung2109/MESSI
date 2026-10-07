import copy,unittest
import torch
from torch import nn
from .pair_shared_model import PairSharedGateSIRM
from .test_centered import CenteringChecks
class PairModelChecks(unittest.TestCase):
    def model(self,on):
        c=copy.deepcopy(CenteringChecks().model().config);c.update(discovery_assignment='pair_shared',discovery_strength_control='match_hard_parameter_gradient',balance_assignment='original_soft',lambda_sirm=1. if on else 0.)
        torch.manual_seed(918);return PairSharedGateSIRM(nn.Linear(4,4),4,3,3,c)
    def test_paired_updates_frozen_and_strength_matched(self):
        a=self.model(False);b=self.model(True);b.load_state_dict(a.state_dict());before={k:v.clone() for k,v in a.state_dict().items() if k.startswith(('featurizer.','global_head.'))}
        for step in range(4):
            torch.manual_seed(400+step);batches=[(torch.randn(8,4),torch.arange(8)%3) for _ in range(3)];aa=a.update(batches);bb=b.update(batches);self.assertEqual(aa,bb);self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()));d=a.last_diagnostics;self.assertAlmostEqual(d['discovery_hard_parameter_grad_norm'],d['discovery_matched_parameter_grad_norm'],places=6);self.assertGreater(d['discovery_pair_count'],0)
        self.assertTrue(all(torch.equal(v,a.state_dict()[k]) for k,v in before.items()));self.assertTrue(all(p.grad is None for name in ['featurizer','global_head'] for p in getattr(a,name).parameters()))
    def test_synthetic_supported_sirm_executes(self):
        m=self.model(True);CenteringChecks().nonzero_heads(m);m.step.fill_(251);m.q_learned.zero_();m.q_learned[0,0]=m.q_learned[1,0]=1.
        torch.manual_seed(171);r=m.update([(torch.randn(12,4),torch.arange(12)%3) for _ in range(3)]);self.assertTrue(r['sirm_active']);self.assertGreater(r['loss_sirm'],0);self.assertEqual(r['effective_lambda_sirm'],1.);self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
