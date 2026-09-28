import itertools
import math
import types
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from domainbed.predictive_support import (PredictiveSupport, feasible_supports,
                                         mmd_unbiased, predictive_terms)
from domainbed.scripts.train_predictive_support import SourceSampler


class ToyFeatures(nn.Module):
    n_outputs = 4
    def forward(self, x):
        return x


def toy_model(experts=3):
    return PredictiveSupport((4, 1, 1), 2, 3, dict(num_experts=experts,
        support_classifier_norm=1., support_warmup_steps=1,
        support_lambda_sub=2., lr=.01, expert_mlp_ratio=1), featurizer=ToyFeatures())


class SupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_feasible_set_includes_overlap_and_global(self):
        candidates, pairs = feasible_supports(torch.ones(3, 2), 6, 3)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(bool(candidates.any(2).all()))
        self.assertTrue(bool(((candidates.sum(1) == 0) | (candidates.sum(1) >= 2)).all()))
        costs = sum((candidates[:, i] & candidates[:, j]).sum(1) for i, j in pairs)
        self.assertTrue(bool(costs.eq(3).all()))
        overlap = torch.zeros(3, 6, dtype=torch.bool)
        overlap[:, :3] = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
        self.assertTrue(bool((candidates == overlap).all(2).all(1).any()))
        self.assertTrue(bool(candidates.sum(1).eq(3).any()))
        # Independent exhaustive oracle (5^6 before filtering).
        options = [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
        expected = sum(sum(math.comb(sum(col), 2) for col in cols) == 3 and
                       all(any(col[k] for col in cols) for k in range(3))
                       for cols in itertools.product(options, repeat=6))
        self.assertEqual(len(candidates), expected)

    def test_missing_classes_not_zero_discrepancy_pairs(self):
        presence = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.bool)
        candidates, pairs = feasible_supports(presence, 2, 2)
        self.assertEqual(pairs, [(0, 1), (1, 2)])
        self.assertTrue(bool((sum((candidates[:, i] & candidates[:, j]).sum(1)
                                  for i, j in pairs) == 2).all()))
        with self.assertRaises(ValueError):
            feasible_supports(torch.eye(3), 3, 1)

    def test_unbiased_mmd_can_be_negative(self):
        x = torch.tensor([[0.], [1.]])
        self.assertLess(mmd_unbiased(x, x, 1.).item(), 0.)
        with self.assertRaises(ValueError):
            mmd_unbiased(x[:1], x, 1.)

    def test_prediction_bound_and_admissibility_gradient(self):
        torch.manual_seed(5)
        z = F.normalize(torch.randn(32, 4, 7), dim=-1)
        w = F.normalize(torch.randn(3, 7), dim=-1) * 2
        logits = z @ w.T
        router = torch.randn(32, 4, requires_grad=True)
        y = torch.randint(3, (32,))
        support = torch.tensor([1, 0, 1, 0], dtype=torch.bool).expand(32, -1)
        loc, adm, _ = predictive_terms(logits, router, y, support)
        pi = router.softmax(1)
        mixed_ce = F.cross_entropy((pi[..., None] * logits).sum(1), y, reduction='none')
        self.assertTrue(bool((mixed_ce <= loc + (math.log(3) + 4) * adm + 1e-6).all()))
        gradient = torch.autograd.grad(adm.sum(), router)[0]
        expected = pi - support * pi / (pi * support).sum(1, keepdim=True)
        torch.testing.assert_close(gradient, expected)
        # Extreme routing must not be clamped or lose the corrective gradient.
        extreme = torch.tensor([[-1000., 1000.]], requires_grad=True)
        _, a, _ = predictive_terms(torch.zeros(1, 2, 2), extreme, torch.tensor([0]),
                                   torch.tensor([[True, False]]))
        self.assertTrue(torch.isfinite(a).all())
        self.assertLess(torch.autograd.grad(a.sum(), extreme)[0][0, 0], -.99)

    def test_alignment_does_not_update_unselected_expert(self):
        z = torch.randn(8, 3, 4, requires_grad=True)
        loss = mmd_unbiased(z[:4, 1], z[4:, 1], 1.)
        loss.backward()
        self.assertEqual(z.grad[:, 0].abs().sum(), 0)
        self.assertEqual(z.grad[:, 2].abs().sum(), 0)
        self.assertGreater(z.grad[:, 1].abs().sum(), 0)

    def test_alternating_update_checkpoint_and_domain_free_inference(self):
        torch.manual_seed(4)
        model = toy_model()
        model.configure_support(torch.ones(3, 2, dtype=torch.bool))
        batches = [(torch.randn(6, 4), torch.tensor([0, 0, 0, 1, 1, 1])) for _ in range(3)]
        self.assertTrue(model.update(batches)['warmup'])
        with self.assertRaises(RuntimeError):
            model.update(batches)
        cells = {(k, c): x[y == c] for k, (x, y) in enumerate(batches) for c in range(2)}
        info = model.update_structure(batches, cells)
        self.assertGreater(info['structure_candidates'], 0)
        pairs = [(m, i, j) for m in range(3) for i in range(3) for j in range(i + 1, 3)
                 if model.support[i, m] and model.support[j, m]]
        m, i, j = pairs[0]
        stats = model.update(batches, alignment_batches=[(m, i, j, 0, cells[i, 0], cells[j, 0])])
        self.assertFalse(stats['warmup'])
        self.assertTrue(math.isfinite(stats['loss']))
        self.assertLessEqual(model.moe_head.classifier.weight.norm(dim=1).max(), 1.000001)
        model.eval()
        prediction = model.predict(batches[0][0])
        clone = toy_model(); clone.load_state_dict(model.state_dict()); clone.eval()
        torch.testing.assert_close(prediction, clone.predict(batches[0][0]))
        clone.support.logical_not_()
        torch.testing.assert_close(prediction, clone.predict(batches[0][0]))
        self.assertIsNone(model.moe_head.classifier.bias)
        mix, logits, router, z = model.components(batches[0][0])
        torch.testing.assert_close(mix, model.moe_head.classifier((router.softmax(1)[..., None] * z).sum(1)))

    def test_sampler_preserves_prediction_law_and_selected_alignment(self):
        class Data:
            targets = [0] * 18 + [1] * 2
            def __getitem__(self, i):
                return torch.tensor([float(i)]), self.targets[i]
        sampler = SourceSampler([Data()] * 3, [list(range(20))] * 3, 2, 3, 'cpu')
        batches = sampler.prediction(1000)
        for _, y in batches:
            self.assertLess(abs(y.float().mean().item() - .1), .04)
        support = torch.tensor([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.bool)
        draws = sampler.alignment(support, 600, 2)
        counts = {}
        for m, i, j, c, x, y in draws:
            self.assertTrue(support[i, m] and support[j, m])
            self.assertEqual(len(x), 2)
            self.assertEqual(len(y), 2)
            self.assertTrue(bool(((x[:, 0] >= 18).long() == c).all()))
            key = (m, i, j)
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), 3)
        self.assertTrue(all(abs(n - 200) < 45 for n in counts.values()))
        # Alignment classes are uniform, rather than following 90/10 prediction prior.
        self.assertLess(abs(sum(row[3] for row in draws) / len(draws) - .5), .07)

    def test_structure_rejects_sparse_observations(self):
        model = toy_model()
        model.configure_support(torch.ones(3, 2, dtype=torch.bool))
        batches = [(torch.randn(4, 4), torch.tensor([0, 0, 1, 1])) for _ in range(3)]
        with self.assertRaises(ValueError):
            model.update_structure(batches, {(0, 0): torch.randn(1, 4)})
        self.assertTrue(model.training)
        self.assertFalse(bool(model.structure_ready))

    def synthetic_structure(self, kind):
        model = toy_model()
        model.configure_support(torch.ones(3, 2, dtype=torch.bool))
        # Each expert has an identical predictive coordinate and a nuisance
        # coordinate equal exactly on its known domain pair. There is no router
        # advantage, so exact search identifies the discrepancy-supported S.
        supports = torch.tensor([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.bool)
        def components(self, x):
            d, y = x[:, 0].long(), x[:, 1]
            nuisance = (~supports[d]).float()
            if kind == 'global':
                nuisance[:, 0] = 0
                nuisance[:, 1:] = d[:, None].float()
            if kind == 'no_gap':
                nuisance.zero_()
            z = torch.stack([y[:, None].expand(-1, 3) * 2 - 1, nuisance], -1)
            z = F.normalize(z, dim=-1)
            # Equal bounded predictive risks isolate discrepancy in this fixture.
            logits = torch.stack([1-y, y], -1)[:, None].expand(-1, 3, -1)
            router = torch.zeros(len(x), 3)
            return logits.mean(1), logits, router, z
        model.components = types.MethodType(components, model)
        # Use all experts admissible initially, then compare valid structure scores.
        # admissibility favors overlap over global; remove that confound by making
        # lambda_sub large enough for a globally shared expert to win when needed.
        model.lambda_sub = 100.
        batches = [(torch.tensor([[k, c] for c in (0, 0, 1, 1)], dtype=torch.float),
                    torch.tensor([0, 0, 1, 1])) for k in range(3)]
        cells = {(k, c): x[y == c] for k, (x, y) in enumerate(batches) for c in range(2)}
        return model, batches, cells, supports

    def test_overlapping_ground_truth_recovery_and_coupling(self):
        model, batches, cells, expected = self.synthetic_structure('overlap')
        model.update_structure(batches, cells)
        self.assertTrue(torch.equal(model.support, expected))
        model.update_structure(batches, cells, mode='no_discrepancy')
        self.assertFalse(torch.equal(model.support, expected))

    def test_global_support_and_no_gap_case(self):
        model, batches, cells, _ = self.synthetic_structure('global')
        model.update_structure(batches, cells)
        self.assertTrue(bool(model.support[:, 0].all()))
        self.assertEqual(int(model.support[:, 1:].sum()), 0)
        model, batches, cells, _ = self.synthetic_structure('no_gap')
        before = model.update_structure(batches, cells)
        # With no representation separation, a permuted solution has equal score;
        # no semantic recovery guarantee is asserted.
        model.support.copy_(model.support.flip(1))
        after = model.update_structure(batches, cells, mode='fixed')
        self.assertAlmostEqual(before['structure_score'], after['structure_score'], places=5)

    def test_collapsed_local_expert_pays_risk_despite_router_shortcut(self):
        y = torch.tensor([0, 1])
        logits = torch.tensor([[[8., -8.], [-8., 8.]], [[8., -8.], [-8., 8.]]])
        router = torch.tensor([[10., -10.], [-10., 10.]])
        loc, _, _ = predictive_terms(logits, router, y, torch.ones(2, 2, dtype=torch.bool))
        mixture = (router.softmax(1)[..., None] * logits).sum(1)
        self.assertLess(F.cross_entropy(mixture, y), .001)
        self.assertGreater(loc.mean(), 10)


if __name__ == '__main__':
    unittest.main()
