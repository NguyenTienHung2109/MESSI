import math
import unittest
import numpy as np
import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from domainbed.predictive_compatibility import (
    make_cell, solve_head, solve_common, audit_head, gaussian_signal,
    uniform_risk_bound, entropy_lower)


def binary_cell(p):
    return make_cell([[1.], [-1.]], [1, 1], [p, 1-p])


class CompatibilityTests(unittest.TestCase):
    def test_convex_bounds_contain_analytic_optimum(self):
        for p in [.1, .5, .8, .99]:
            w, result = solve_head(binary_cell(p), 2, steps=500)
            logit = np.clip(math.log(p/(1-p)), -2, 2)
            exact = np.logaddexp(0, logit) - p*logit
            self.assertLessEqual(result['lower'], exact + 1e-10)
            self.assertGreaterEqual(result['upper'], exact - 1e-10)
            self.assertLess(result['gap'], 1e-5)
            self.assertLessEqual(float(w.norm(dim=1).max()), 1 + 1e-12)

    def test_conflicting_predictors(self):
        _, result = solve_common([binary_cell(.8), binary_cell(.2)], 2, steps=500)
        exact = math.log(2) + .8*math.log(.8) + .2*math.log(.2)
        self.assertLessEqual(result['lower'], exact+1e-10)
        self.assertGreaterEqual(result['upper'], exact-1e-10)
        self.assertLess(result['gap'], 1e-5)

    def test_equal_optimum_does_not_require_equal_risk(self):
        # Adding uninformative z=0 observations changes risk, not optimal W.
        a = binary_cell(.8)
        b = make_cell([[1.], [-1.], [0.]], [1, 1, 0], [.4, .1, .5])
        _, result = solve_common([a, b], 2, steps=1000)
        self.assertLess(result['upper'], 1e-5)
        self.assertGreater(abs(result['individual'][0]['upper']-result['individual'][1]['upper']), .05)

    def test_audit_accepts_signal_with_enough_data(self):
        cells = [binary_cell(.84)]*2
        w, _ = solve_common(cells, 2)
        large = audit_head(w, cells, [[100000, 100000]]*2, 1., .12, .05, .001)
        self.assertTrue(large['accepted'])
        small = audit_head(w, cells, [[20, 20]]*2, 1., .12, .05, .001)
        self.assertFalse(small['accepted'])
        chance = audit_head(torch.zeros_like(w), [binary_cell(.5)], [[100000, 100000]], 1., .12, .05, .001)
        self.assertFalse(chance['accepted'])

    def test_gaussian_ground_truth(self):
        means = [[1, 0], [1, 1], [0, 1]]
        self.assertAlmostEqual(gaussian_signal(means, np.eye(2), [0, 1])['gamma'], 1.)
        self.assertAlmostEqual(gaussian_signal(means, np.eye(2), [1, 2])['gamma'], 1.)
        self.assertFalse(gaussian_signal(means, np.eye(2), [0, 1, 2])['predictive'])
        self.assertFalse(gaussian_signal([[-1], [0], [1]], [[1]], [0, 1])['predictive'])

    def test_bounds_and_input_validation(self):
        self.assertGreater(uniform_risk_bound(100, 2, 1, 1, .01), uniform_risk_bound(10000, 2, 1, 1, .01))
        self.assertEqual(entropy_lower([100, 0], .01)[1], 0)
        with self.assertRaises(ValueError):
            make_cell([[2.]], [0])

    def test_disjoint_partitions(self):
        from domainbed.scripts.audit_predictive_compatibility import partition
        parts = partition(103, 0)
        combined = np.concatenate(list(parts.values()))
        self.assertEqual(len(np.unique(combined)), 103)
        self.assertEqual(sorted(combined), list(range(103)))
        np.testing.assert_array_equal(parts['audit'], partition(103, 0)['audit'])

    def test_fold_fallback_and_held_domain_exclusion(self):
        from domainbed.scripts.audit_predictive_compatibility import fit_fold
        args = SimpleNamespace(seed=0, alpha=.05, dimension=2, hidden=4, radius=1.,
                               map_steps=10, router_steps=10, solver_steps=50,
                               solver_tolerance=1e-4, learning_rate=.01,
                               batch_size=8, tau=.12, delta=.05)
        torch.manual_seed(0)
        bank = {d: (torch.randn(40, 3), torch.arange(40) % 2) for d in [0, 1]}
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaises(ValueError):
                fit_fold(bank, [0], 2, args, out/'invalid', 10)
            result, evaluate = fit_fold(bank, [0, 1], 2, args, out/'valid', 10)
            self.assertEqual(result['accepted_experts'], [])
            self.assertEqual(result['selected'], 'global')
            self.assertEqual(result['adaptation_domains'], [0, 1])
            evaluation = evaluate(*bank[0])
            self.assertEqual(evaluation['controls']['global_predictor'], evaluation['controls']['selected'])
            self.assertTrue((out/'valid/model.pt').exists())

    def test_jensen_for_distinct_expert_heads(self):
        from torch.nn import functional as F
        torch.manual_seed(4)
        logits = torch.randn(25, 3, 2)
        weights = torch.randn(25, 3).softmax(1)
        y = torch.arange(25) % 2
        mixed = F.cross_entropy((weights[:, :, None]*logits).sum(1), y, reduction='none')
        individual = torch.stack([F.cross_entropy(logits[:, i], y, reduction='none') for i in range(3)], 1)
        self.assertTrue((mixed <= (weights*individual).sum(1)+1e-6).all())

    def test_diagnostics_identical_constant_experts(self):
        from domainbed.compatibility_diagnostics import bank_behavior, prediction_metrics
        y = torch.tensor([0, 1, 0, 1])
        logits = torch.zeros(4, 2, 2)
        features = torch.ones(4, 2, 1)
        weights = torch.full((4, 2), .5)
        result = bank_behavior(features, logits, weights, y)
        self.assertAlmostEqual(result['jensen_slack'], 0.)
        self.assertEqual(result['removal_delta_learned_ce'], [0., 0.])
        self.assertEqual(result['probability_js'], [[0., 0.], [0., 0.]])
        self.assertEqual(result['feature_statistics'][0]['effective_rank'], 0.)
        self.assertAlmostEqual(result['router_effective_experts'], 2.)
        scores = prediction_metrics(logits[:, 0], y)
        self.assertAlmostEqual(scores['ece_15'], 0.)
        self.assertAlmostEqual(scores['brier'], .5)
        self.assertEqual(scores['confusion'], [[2, 0], [2, 0]])


if __name__ == '__main__':
    unittest.main()
