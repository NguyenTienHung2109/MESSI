import copy
import unittest
import torch
from torch import nn
from .hard_router_cap_model import HardRouterCapGateSIRM, bounded_auxiliary
from .hard_discovery import HardWinnerGateSIRM
from .test_hard_discovery import HardDiscoveryChecks
from .test_centered import CenteringChecks


class HardRouterCapChecks(unittest.TestCase):
    def test_active_cap_is_detached_and_router_only(self):
        p = nn.Parameter(torch.tensor([1., 2.])); q = nn.Parameter(torch.tensor([3.]))
        cls = p.square().sum() + q.square().sum(); aux = 10 * p.square().sum()
        loss, scale, cn, an, bn = bounded_auxiliary(cls, aux, [p])
        self.assertAlmostEqual(float(scale), .1, places=6); self.assertFalse(scale.requires_grad)
        self.assertAlmostEqual(float(cn), float(bn), places=6)
        self.assertEqual(torch.autograd.grad(loss, q, allow_unused=True), (None,))
        self.assertTrue(p.grad is None and q.grad is None)

    def test_zero_classification_zeroes_auxiliary(self):
        p = nn.Parameter(torch.tensor([1.]))
        loss, scale, cn, an, bn = bounded_auxiliary(p.sum() * 0, p.square().sum(), [p])
        self.assertEqual(float(scale), 0.); self.assertEqual(float(bn), 0.)
        self.assertEqual(float(torch.autograd.grad(loss, p)[0]), 0.)

    def test_no_clipping_when_aux_smaller(self):
        p = nn.Parameter(torch.tensor([1.]))
        aux = p.square().sum()
        loss, scale, cn, an, bn = bounded_auxiliary(10 * aux, aux, [p])
        self.assertEqual(float(scale), 1.); self.assertTrue(torch.equal(loss, aux))

    def test_zero_auxiliary_keeps_identity_scale(self):
        p = nn.Parameter(torch.tensor([1.]))
        loss, scale, cn, an, bn = bounded_auxiliary(p.square().sum(), p.sum() * 0, [p])
        self.assertEqual(float(scale), 1.); self.assertEqual(float(an), 0.); self.assertEqual(float(bn), 0.)

    def model(self, mode='cap_aux_to_classification', on=False):
        old = HardDiscoveryChecks().model(on)
        c = copy.deepcopy(old.config); c['router_strength_control'] = mode
        torch.manual_seed(918); m = HardRouterCapGateSIRM(nn.Linear(4, 4), 4, 3, 3, c)
        m.load_state_dict(old.state_dict()); return m

    def test_disabled_cap_exact_update_matches_prior(self):
        old = HardDiscoveryChecks().model(False); new = self.model('none')
        CenteringChecks().nonzero_heads(old); new.load_state_dict(old.state_dict())
        torch.manual_seed(812); batches = [(torch.randn(8, 4), torch.arange(8) % 3) for _ in range(3)]
        a = old.update(batches); b = new.update(batches)
        for k in a: self.assertAlmostEqual(a[k], b[k], places=7) if isinstance(a[k], float) else self.assertEqual(a[k], b[k])
        for k, v in old.state_dict().items(): self.assertTrue(torch.equal(v, new.state_dict()[k]), k)

    def test_pair_frozen_and_norm_bound_each_update(self):
        a = self.model(); b = self.model(on=True); b.load_state_dict(a.state_dict())
        before = {k: v.clone() for k, v in a.state_dict().items() if k.startswith(('featurizer.', 'global_head.'))}
        for step in range(4):
            torch.manual_seed(813 + step); batches = [(torch.randn(8, 4), torch.arange(8) % 3) for _ in range(3)]
            result_a = a.update(batches); result_b = b.update(batches)
            self.assertEqual(result_a, result_b)
            d = a.last_diagnostics
            self.assertEqual(d['discovery_assignment'], 'hard_winner')
            self.assertAlmostEqual(a.last_diagnostics['discovery_hard_loss_replay'], result_a['loss_discover'], places=7)
            self.assertLessEqual(d['router_aux_post_cap_parameter_grad_norm'], d['router_cls_parameter_grad_norm'] * 1.00001 + 1e-10)
            for k, v in a.state_dict().items(): self.assertTrue(torch.equal(v, b.state_dict()[k]), k)
        for k, v in before.items(): self.assertTrue(torch.equal(v, a.state_dict()[k]), k)
        self.assertTrue(all(p.grad is None for mod in [a.featurizer, a.global_head] for p in mod.parameters()))

    def test_cap_does_not_change_same_state_residual_gradients(self):
        a = self.model('none'); b = self.model()
        CenteringChecks().nonzero_heads(a); b.load_state_dict(a.state_dict())
        torch.manual_seed(922); batches = [(torch.randn(8, 4), torch.arange(8) % 3) for _ in range(3)]
        a.update(batches); b.update(batches)
        self.assertLess(b.last_diagnostics['router_aux_detached_scale'], 1.)
        for pa, pb in zip(a.residuals.parameters(), b.residuals.parameters()):
            self.assertTrue(torch.equal(pa.grad, pb.grad))

    def test_supported_sirm_still_executes_with_cap(self):
        m = self.model(on=True); CenteringChecks().nonzero_heads(m); m.step.fill_(251)
        m.q_learned.zero_(); m.q_learned[0, 0] = m.q_learned[1, 0] = 1.
        torch.manual_seed(171)
        r = m.update([(torch.randn(12, 4), torch.arange(12) % 3) for _ in range(3)])
        self.assertTrue(r['sirm_active']); self.assertGreater(r['loss_sirm'], 0.)
        self.assertEqual(r['effective_lambda_sirm'], 1.)
        self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))


if __name__ == '__main__':
    torch.set_num_threads(2); unittest.main()
