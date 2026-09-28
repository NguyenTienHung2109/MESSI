import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.sirm_res import (
    marginal_residual_sirm_penalty,
    proper_subset_assignment,
    sparse_router_with_floor,
)


class TestSIRMResRouter(unittest.TestCase):
    def test_selected_weights_have_floor_and_sum_to_one(self):
        logits = torch.tensor([[4.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 3.0]])
        weights, selected = sparse_router_with_floor(logits, top_k=2, q_min=0.2)
        self.assertTrue(torch.equal(selected.sum(1), torch.tensor([2, 2])))
        torch.testing.assert_close(weights.sum(1), torch.ones(2))
        self.assertTrue(bool((weights[selected] >= 0.2).all()))
        self.assertTrue(bool((weights[~selected] == 0).all()))

    def test_proper_subset_removes_universal_residual_when_possible(self):
        risks = torch.tensor([[0.1, 0.2, 1.0], [0.1, 0.3, 1.0], [0.1, 0.4, 1.0]])
        valid = torch.ones_like(risks, dtype=torch.bool)
        q = proper_subset_assignment(risks, valid, top_r=1)
        torch.testing.assert_close(q.sum(1), torch.ones(3))
        self.assertFalse(bool(q[:, 0].gt(0).all()))


class TestSIRMResLoss(unittest.TestCase):
    def test_zero_residual_heads_make_final_exactly_global(self):
        torch.manual_seed(0)
        global_head = nn.Linear(5, 3)
        residual_heads = nn.ModuleList([nn.Linear(5, 3), nn.Linear(5, 3)])
        for head in residual_heads:
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        features = torch.randn(7, 5)
        global_logits = global_head(features)
        residual = torch.stack([head(features) for head in residual_heads], dim=1)
        weights, _ = sparse_router_with_floor(torch.randn(7, 2), 2, .2)
        final_logits = global_logits + (weights.unsqueeze(-1) * residual).sum(1)
        torch.testing.assert_close(final_logits, global_logits, rtol=0, atol=0)

    def test_marginal_penalty_isolates_unassigned_expert_and_router(self):
        torch.manual_seed(3)
        residual_a, residual_b = nn.Linear(4, 3, bias=False), nn.Linear(4, 3, bias=False)
        router = nn.Linear(4, 2, bias=False)
        global_head = nn.Linear(4, 3, bias=False)
        x = torch.randn(12, 4)
        targets = torch.randint(0, 3, (12,))
        domains = torch.arange(3).repeat_interleave(4)
        global_logits = global_head(x)
        residual_logits = torch.stack([residual_a(x), residual_b(x)], dim=1)
        routing, _ = sparse_router_with_floor(router(x), top_k=2, q_min=.2)
        # Only residual 0 has proper two-domain Q support; residual 1 is not
        # assigned, so it must receive no S-IRM-res gradient.
        q = torch.tensor([[.5, 0.0], [.5, 0.0], [0.0, 0.0]])
        penalty, info = marginal_residual_sirm_penalty(
            global_logits, residual_logits, targets, domains, routing, q,
            min_effective_samples=2,
        )
        penalty.backward()
        self.assertTrue(info["active"])
        self.assertGreater(residual_a.weight.grad.abs().sum().item(), 0)
        self.assertTrue(residual_b.weight.grad is None or residual_b.weight.grad.abs().sum().item() == 0)
        self.assertIsNone(router.weight.grad)
        self.assertIsNone(global_head.weight.grad)

    def test_synthetic_global_then_residual_optimization(self):
        """Small deterministic 3-domain integration check (not a benchmark)."""
        torch.manual_seed(4)
        domain = torch.arange(3).repeat_interleave(24)
        shared = torch.randn(72, 1)
        subset = torch.randn(72, 1)
        labels = (shared[:, 0] + (domain.ne(2).float() * subset[:, 0]) > 0).long()
        features = torch.cat([shared, subset], dim=1)
        global_head = nn.Linear(2, 2)
        global_opt = torch.optim.SGD(global_head.parameters(), lr=.2)
        for _ in range(40):
            global_opt.zero_grad()
            loss = torch.stack([F.cross_entropy(global_head(features[domain == d]), labels[domain == d]) for d in range(3)]).mean()
            loss.backward(); global_opt.step()
        anchor = {k: v.detach().clone() for k, v in global_head.state_dict().items()}
        residual = nn.Linear(2, 2)
        nn.init.zeros_(residual.weight); nn.init.zeros_(residual.bias)
        residual_opt = torch.optim.SGD(residual.parameters(), lr=.15)
        before = F.cross_entropy(global_head(features), labels).item()
        for _ in range(30):
            residual_opt.zero_grad()
            loss = F.cross_entropy(global_head(features).detach() + residual(features), labels)
            loss.backward(); residual_opt.step()
        after = F.cross_entropy(global_head(features).detach() + residual(features), labels).item()
        self.assertLess(after, before)
        self.assertGreater(residual.weight.detach().abs().sum().item(), 0)
        for key, value in global_head.state_dict().items():
            torch.testing.assert_close(value, anchor[key], rtol=0, atol=0)
