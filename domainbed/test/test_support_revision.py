"""Regression tests for the causal direction and predictive-information hinge."""
import math
import types
import unittest

import torch
from torch.nn import functional as F

from domainbed.predictive_support import PredictiveSupport, information_constraint
from domainbed.test.test_predictive_support import ToyFeatures


def revised_model(**overrides):
    hp = dict(num_experts=3, expert_mlp_ratio=1, lr=.01,
              support_classifier_norm=1., support_warmup_steps=1,
              support_predictive_objective='information_hinge',
              support_structure_router_weight=0., support_lambda_adm=1.,
              support_adm_ramp_steps=4)
    hp.update(overrides)
    model = PredictiveSupport((4, 1, 1), 2, 3, hp, featurizer=ToyFeatures())
    model.configure_support(torch.ones(3, 2, dtype=torch.bool), torch.ones(3, 2) * 10)
    return model


def known_support_fixture(kind='overlap'):
    model = revised_model()
    expected = torch.tensor([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.bool)
    model.router_bias_fixture = torch.zeros(3)
    def components(self, x):
        d, y = x[:, 0].long(), x[:, 1]
        nuisance = (~expected[d]).float() * .6
        if kind == 'global':
            nuisance[:, 0] = 0
            nuisance[:, 1:] = d[:, None].float() * .3
        if kind == 'no_gap':
            nuisance.zero_()
        z = F.normalize(torch.stack([(2*y[:, None]-1).expand(-1, 3) * .8, nuisance], -1), dim=-1)
        # Same bounded linear W for every normalized expert, as in the method.
        w = torch.tensor([[-1., 0.], [1., 0.]])
        logits = z @ w.T
        router = self.router_bias_fixture.expand(len(x), -1)
        return (router.softmax(1)[..., None] * logits).sum(1), logits, router, z
    model.components = types.MethodType(components, model)
    batches = [(torch.tensor([[k, c] for c in (0, 0, 1, 1)], dtype=torch.float),
                torch.tensor([0, 0, 1, 1])) for k in range(3)]
    cells = {(k, c): x[y == c] for k, (x, y) in enumerate(batches) for c in range(2)}
    return model, batches, cells, expected


class SupportRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_risk_and_entropy_use_support_reference_law(self):
        # Unequal raw domain sizes must not replace the uniform-domain Q.
        risks = torch.tensor([[.2, .8], [.8, .2], [5., .6]], requires_grad=True)
        support = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.bool)
        p = torch.tensor([[.9, .1], [.1, .9], [.1, .9]])
        loss, info = information_constraint(risks, support, p, .1)
        torch.testing.assert_close(info['risk'], torch.tensor([.5, .4]))
        self.assertAlmostEqual(float(info['entropy'][0]), math.log(2), places=6)
        self.assertLess(float(info['entropy'][1]), .33)
        loss.backward()
        self.assertEqual(float(risks.grad[:, 0].abs().sum()), 0.)
        self.assertEqual(float(risks.grad[0, 1]), 0.)
        torch.testing.assert_close(risks.grad[1:, 1], torch.tensor([.5, .5]))

    def test_constant_expert_pays_margin_and_inactive_experts_do_not(self):
        p = torch.full((3, 2), .5)
        risks = torch.full((3, 2), math.log(2), requires_grad=True)
        support = torch.tensor([[1, 0], [1, 0], [1, 0]], dtype=torch.bool)
        loss, info = information_constraint(risks, support, p, .1)
        self.assertAlmostEqual(float(loss), .1, places=6)
        loss.backward()
        self.assertEqual(float(risks.grad[:, 1].sum()), 0.)
        self.assertFalse(bool(info['active'][1]))

    def test_satisfied_predictive_constraint_has_zero_gradient(self):
        r = torch.full((3, 3), .2, requires_grad=True)
        loss, _ = information_constraint(r, torch.ones(3, 3, dtype=torch.bool), torch.full((3, 2), .5), .1)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(r.grad.abs().sum()), 0.)

    def test_router_perturbation_cannot_change_structure_scores(self):
        model, batches, cells, expected = known_support_fixture()
        before = model.update_structure(batches, cells)
        model.router_bias_fixture = torch.tensor([1000., -1000., 200.])
        after = model.update_structure(batches, cells)
        self.assertTrue(torch.equal(model.support, expected))
        self.assertEqual(before['structure_score'], after['structure_score'])
        self.assertEqual(before['structure_eda']['empirical_runner_up_gap'], after['structure_eda']['empirical_runner_up_gap'])
        for a, b in zip(before['structure_eda']['top_candidates'], after['structure_eda']['top_candidates']):
            self.assertEqual(a['support'], b['support'])
            self.assertEqual(a['score'], b['score'])
        self.assertEqual(after['structure_eda']['gap_router'], 0.)
        self.assertNotEqual(before['structure_eda']['selected_score_adm'], after['structure_eda']['selected_score_adm'])

    def test_gap_decomposition_and_discrepancy_intervention(self):
        model, batches, cells, expected = known_support_fixture()
        result = model.update_structure(batches, cells)
        eda = result['structure_eda']
        self.assertTrue(torch.equal(model.support, expected))
        self.assertAlmostEqual(eda['empirical_runner_up_gap'], eda['gap_predictive'] + eda['gap_mmd'] + eda['gap_router'], places=6)
        self.assertGreater(eda['gap_mmd'], 0.)
        result = model.update_structure(batches, cells, mode='no_discrepancy')
        self.assertEqual(result['structure_score'], 0.)
        self.assertGreater(result['structure_eda']['empirical_best_tie_count_at_1e8'], 1)

    def test_real_head_structure_is_invariant_to_router_weights(self):
        torch.manual_seed(7)
        model = revised_model()
        batches = [(torch.randn(4, 4), torch.tensor([0, 0, 1, 1])) for _ in range(3)]
        cells = {(k,c): x[y==c] for k,(x,y) in enumerate(batches) for c in range(2)}
        before = model.update_structure(batches, cells)
        with torch.no_grad():
            model.moe_head.router.weight.mul_(1000)
            model.moe_head.router.bias.copy_(torch.tensor([1000., -1000., 0.]))
        after = model.update_structure(batches, cells)
        self.assertEqual(before['support'], after['support'])
        self.assertEqual(before['structure_score'], after['structure_score'])

    def test_mixture_supervision_selects_within_admissible_set(self):
        logits = torch.tensor([[[3., -3.], [-3., 3.]], [[3., -3.], [-3., 3.]]])
        router = torch.zeros(2, 2, requires_grad=True)
        pi = router.softmax(1)
        adm = -(pi.sum(1)).log().mean()  # both experts are admissible
        adm_grad = torch.autograd.grad(adm, router, retain_graph=True)[0]
        mix = F.cross_entropy((pi[..., None] * logits).sum(1), torch.tensor([0, 1]))
        mix_grad = torch.autograd.grad(mix, router)[0]
        self.assertEqual(float(adm_grad.abs().sum()), 0.)
        self.assertLess(float(mix_grad[0, 0]), 0.)
        self.assertGreater(float(mix_grad[0, 1]), 0.)
        self.assertGreater(float(mix_grad[1, 0]), 0.)
        self.assertLess(float(mix_grad[1, 1]), 0.)

    def test_global_and_no_separation_cases(self):
        model, batches, cells, _ = known_support_fixture('global')
        model.update_structure(batches, cells)
        self.assertTrue(bool(model.support[:, 0].all()))
        self.assertEqual(int(model.support[:, 1:].sum()), 0)
        model, batches, cells, _ = known_support_fixture('no_gap')
        result = model.update_structure(batches, cells)
        self.assertEqual(result['structure_eda']['empirical_runner_up_gap'], 0.)

    def test_real_neural_update_trains_router_and_roundtrips_priors(self):
        torch.manual_seed(42)
        model = revised_model()
        batches = [(torch.randn(8, 4), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])) for _ in range(3)]
        old_router = model.moe_head.router.weight.detach().clone()
        warm = model.update(batches)
        self.assertTrue(warm['warmup'])
        self.assertEqual(warm['admissibility_coefficient'], 0.)
        self.assertFalse(torch.equal(old_router, model.moe_head.router.weight))
        cells = {(k,c): x[y==c] for k,(x,y) in enumerate(batches) for c in range(2)}
        model.update_structure(batches, cells)
        m,i,j = next((m,i,j) for m in range(3) for i in range(3) for j in range(i+1,3)
                     if model.support[i,m] and model.support[j,m])
        r = model.update(batches, alignment_batches=[(m,i,j,0,cells[i,0],cells[j,0])])
        self.assertEqual(r['admissibility_coefficient'], .25)
        self.assertAlmostEqual(r['loss'], r['mixture_ce'] + r['weighted_predictive'] + r['weighted_admissibility'] + r['weighted_mmd'], places=5)
        clone = revised_model(); clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(model.source_class_probabilities, clone.source_class_probabilities)
        torch.testing.assert_close(model.predict(batches[0][0]), clone.predict(batches[0][0]))

    def test_missing_training_counts_are_rejected(self):
        model = revised_model()
        with self.assertRaises(ValueError):
            model.configure_support(torch.ones(3, 2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model.configure_support(torch.ones(3, 2, dtype=torch.bool), torch.zeros(3, 2))


if __name__ == '__main__':
    unittest.main()
