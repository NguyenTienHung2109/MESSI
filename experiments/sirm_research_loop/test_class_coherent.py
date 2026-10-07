import copy
import unittest
from unittest.mock import patch
import torch
from torch import nn
from experiments.sirm_discover.math import multiple_choice_loss
from .class_coherent_discovery import class_coherent_cost, class_coherent_loss
from .class_coherent_model import ClassCoherentGateSIRM
from .test_hard_router_cap import HardRouterCapChecks
from .test_centered import CenteringChecks


class ClassCoherentChecks(unittest.TestCase):
    def test_conflicting_image_winners_select_one_class_expert(self):
        losses = torch.tensor([[.1, 1.], [1., .2]], dtype=torch.double, requires_grad=True)
        loss = class_coherent_cost(losses, torch.tensor([0, 0]))
        self.assertAlmostEqual(float(loss), .55)
        self.assertTrue(torch.equal(torch.autograd.grad(loss, losses)[0], torch.tensor([[.5, 0.], [.5, 0.]], dtype=torch.double)))

    def test_singletons_equal_image_hard(self):
        losses = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
        a = class_coherent_cost(losses, torch.arange(4)); b = losses.amin(1).mean()
        self.assertTrue(torch.allclose(a, b, atol=1e-14))
        self.assertTrue(torch.equal(torch.autograd.grad(a, losses, retain_graph=True)[0], torch.autograd.grad(b, losses)[0]))

    def test_count_weighting_and_ties(self):
        losses = torch.tensor([[1., 1.], [1., 1.], [3., 3.]], dtype=torch.double, requires_grad=True)
        loss = class_coherent_cost(losses, torch.tensor([0, 0, 1]))
        self.assertAlmostEqual(float(loss), 5 / 3)
        self.assertTrue(torch.allclose(torch.autograd.grad(loss, losses)[0], torch.full_like(losses, 1 / 6)))

    def test_label_and_sample_permutation(self):
        torch.manual_seed(44); losses = torch.randn(12, 6, dtype=torch.double)
        y = torch.arange(12) % 3; p = torch.randperm(12)
        self.assertTrue(torch.allclose(class_coherent_cost(losses, y), class_coherent_cost(losses[p], (2-y)[p]), atol=1e-14))

    def test_original_balance_values_and_gradients(self):
        torch.manual_seed(45); g = torch.randn(12, 3, dtype=torch.double)
        delta = torch.randn(12, 6, 3, dtype=torch.double, requires_grad=True); y = torch.arange(12) % 3
        _, a = class_coherent_loss(g, delta, y, .25); _, b = multiple_choice_loss(g, delta, y, .25)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(torch.autograd.grad(a.mean(0).square().sum(), delta, retain_graph=True)[0], torch.autograd.grad(b.mean(0).square().sum(), delta)[0]))

    def model(self, on=False):
        old = HardRouterCapChecks().model(on=on); c = copy.deepcopy(old.config)
        c['discovery_scope'] = 'label_class_batch'
        m = ClassCoherentGateSIRM(nn.Linear(4,4), 4, 3, 3, c)
        m.load_state_dict(old.state_dict()); return m

    def test_one_optimizer_step_frozen_pair_and_actual_formula(self):
        a = self.model(); b = self.model(True); b.load_state_dict(a.state_dict())
        CenteringChecks().nonzero_heads(a); b.load_state_dict(a.state_dict())
        before = {k:v.clone() for k,v in a.state_dict().items() if k.startswith(('featurizer.', 'global_head.'))}
        for step in range(3):
            torch.manual_seed(46+step); batches = [(torch.randn(8,4), torch.arange(8)%3) for _ in range(3)]
            with patch.object(a.optimizer, 'step', wraps=a.optimizer.step) as spy:
                ra = a.update(batches)
                self.assertEqual(spy.call_count, 1)
            rb = b.update(batches); self.assertEqual(ra, rb)
            d = a.last_diagnostics
            self.assertNotIn('discovery_hard_loss_replay', d)
            self.assertAlmostEqual(ra['loss_discover'], d['discovery_class_loss_replay'], places=6)
            self.assertEqual(sum(block['count'] for block in d['discovery_class_blocks']), 24)
            self.assertLessEqual(d['router_aux_post_cap_parameter_grad_norm'], d['router_cls_parameter_grad_norm']*1.00001+1e-10)
            self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        self.assertTrue(all(torch.equal(v,a.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for mod in [a.featurizer,a.global_head] for p in mod.parameters()))

    def test_eligible_sirm_not_disabled_by_adapter(self):
        m = self.model(True); CenteringChecks().nonzero_heads(m); m.step.fill_(251)
        m.q_learned.zero_(); m.q_learned[0,0] = m.q_learned[1,0] = 1.
        torch.manual_seed(171); r = m.update([(torch.randn(12,4),torch.arange(12)%3) for _ in range(3)])
        self.assertTrue(r['sirm_active']); self.assertGreater(r['loss_sirm'],0)
        self.assertEqual(r['effective_lambda_sirm'],1.)

if __name__ == '__main__':
    torch.set_num_threads(2); unittest.main()
