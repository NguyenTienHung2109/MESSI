import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.subset_irm import (
    ExpertClassifierHeads,
    MESSISubsetIRM,
    SourceRiskEMA,
    expert_domain_risk,
    expert_predictive_loss,
    per_sample_topk,
    q_capacity_loss,
    q_gated_sinkhorn_divergence,
    router_distillation_loss,
    sinkhorn_divergence,
    subset_irm_penalty,
    top_r_assignment,
)
from domainbed.scripts.train_subset_irm_pacs import (
    target_checkpoint_diagnostics,
)


class TestExpertHeadsAndRouting(unittest.TestCase):
    def test_six_identical_heads_shapes_and_device(self):
        shared = nn.Linear(5, 3)
        heads = ExpertClassifierHeads(6, 5, 3, shared)
        features = torch.randn(4, 6, 5)
        logits = heads(features)
        self.assertEqual(len(heads.heads), 6)
        self.assertEqual(logits.shape, (4, 6, 3))
        self.assertEqual(logits.device, next(heads.parameters()).device)
        for head in heads.heads:
            torch.testing.assert_close(head.weight, shared.weight)
            torch.testing.assert_close(head.bias, shared.bias)

    def test_auxiliary_prediction_is_original_shared_feature_mix(self):
        torch.manual_seed(0)
        h = torch.randn(5, 6, 4)
        pi = torch.softmax(torch.randn(5, 6), dim=1)
        shared = nn.Linear(4, 3)
        original = shared((pi.unsqueeze(-1) * h).sum(dim=1))
        _ = ExpertClassifierHeads(6, 4, 3, shared)(h)
        auxiliary = shared((pi.unsqueeze(-1) * h).sum(dim=1))
        torch.testing.assert_close(auxiliary, original, rtol=0, atol=0)

    def test_expert_logit_mixture_shape(self):
        logits = torch.randn(7, 6, 3)
        gamma = per_sample_topk(torch.softmax(torch.randn(7, 6), dim=1), 2)
        self.assertEqual((gamma.unsqueeze(-1) * logits).sum(1).shape, (7, 3))

    def test_topk_is_exact_normalized_per_sample_and_deterministic(self):
        probabilities = torch.tensor([
            [0.50, 0.30, 0.10, 0.05, 0.03, 0.02],
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
            [0.01, 0.02, 0.70, 0.20, 0.04, 0.03],
        ])
        first = per_sample_topk(probabilities, 2)
        second = per_sample_topk(probabilities, 2)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close((first > 0).sum(1), torch.full((3,), 2))
        torch.testing.assert_close(first.sum(1), torch.ones(3))
        self.assertTrue(bool(first[first == 0].eq(0).all()))
        selected_sets = [tuple(row.nonzero().flatten().tolist()) for row in first]
        self.assertGreater(len(set(selected_sets)), 1)

    def test_topk_masks_nonselected_per_sample_expert_gradients(self):
        experts = nn.ModuleList([nn.Linear(3, 2, bias=False) for _ in range(6)])
        heads = ExpertClassifierHeads(6, 2, 2)
        x = torch.tensor([[1.0, -2.0, 0.5]])
        probabilities = torch.tensor([[0.60, 0.25, 0.05, 0.04, 0.03, 0.03]])
        gamma = per_sample_topk(probabilities, 2)
        features = torch.stack([expert(x) for expert in experts], dim=1)
        logits = heads(features)
        loss = F.cross_entropy((gamma.unsqueeze(-1) * logits).sum(1),
                               torch.tensor([1]))
        loss.backward()
        for index in range(6):
            expert_norm = experts[index].weight.grad.abs().sum().item()
            head_norm = heads.heads[index].weight.grad.abs().sum().item()
            if index in (0, 1):
                self.assertGreater(expert_norm, 0)
                self.assertGreater(head_norm, 0)
                self.assertTrue(torch.isfinite(experts[index].weight.grad).all())
            else:
                self.assertEqual(expert_norm, 0)
                self.assertEqual(head_norm, 0)


class TestRiskAndExpertLoss(unittest.TestCase):
    def _batch(self, dtype=torch.float32, device="cpu"):
        logits = torch.tensor([
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 2.0], [1.0, 0.0]],
            [[1.5, 0.0], [0.0, 1.5]],
            [[0.0, 1.5], [1.5, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ], dtype=dtype, device=device)
        labels = torch.tensor([0, 1, 0, 1, 0], device=device)
        domains = torch.tensor([0, 0, 1, 1, 1], device=device)
        gamma = torch.tensor([
            [1.0, 0.0], [0.5, 0.5], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]
        ], dtype=dtype, device=device)
        return logits, labels, domains, gamma

    def test_weighted_risk_matches_manual_and_handles_mixed_domains(self):
        logits, labels, domains, gamma = self._batch()
        risk, ess, valid, support = expert_domain_risk(
            logits, labels, domains, gamma, 2, min_effective_samples=1
        )
        ce = F.cross_entropy(logits[:, 0], labels, reduction="none")
        mask = domains.eq(1)
        manual = (gamma[mask, 0] * ce[mask]).sum() / gamma[mask, 0].sum()
        torch.testing.assert_close(risk[1, 0], manual)
        self.assertTrue(torch.isfinite(risk).all())
        self.assertTrue(torch.isfinite(ess).all())
        self.assertEqual(support.shape, (2, 2))
        self.assertTrue(bool(valid.any()))

    def test_one_hot_zero_support_and_insufficient_ess(self):
        logits, labels, domains, _ = self._batch()
        gamma = torch.zeros(5, 2)
        gamma[:, 0] = 1
        risk, ess, valid, support = expert_domain_risk(
            logits, labels, domains, gamma, 2, min_effective_samples=4
        )
        self.assertTrue(torch.isfinite(risk).all())
        self.assertTrue(torch.isfinite(ess).all())
        self.assertTrue(risk[:, 1].eq(0).all())
        self.assertTrue(support[:, 1].eq(0).all())
        self.assertFalse(bool(valid.any()))

    def test_fp32_cpu_and_autocast_gpu_compatibility(self):
        logits, labels, domains, gamma = self._batch()
        cpu = expert_domain_risk(logits, labels, domains, gamma, 2, 1)[0]
        self.assertEqual(cpu.dtype, torch.float32)
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        logits, labels, domains, gamma = self._batch(device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            gpu = expert_domain_risk(logits, labels, domains, gamma, 2, 1)[0]
        self.assertTrue(torch.isfinite(gpu).all())
        torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-4, atol=1e-5)

    def test_expert_loss_gradient_paths_with_detached_responsibility(self):
        torch.manual_seed(3)
        backbone = nn.Linear(3, 4, bias=False)
        experts = nn.ModuleList([nn.Linear(4, 2, bias=False) for _ in range(3)])
        heads = ExpertClassifierHeads(3, 2, 2)
        router = nn.Linear(4, 3, bias=False)
        x = torch.tensor([[1.0, 2.0, -1.0]])
        z = backbone(x)
        pi = torch.softmax(router(z), dim=1)
        # Force experts 0/1 active and expert 2 inactive while preserving the
        # router graph before responsibility detachment.
        gamma = per_sample_topk(pi + torch.tensor([[2.0, 1.0, 0.0]]), 2)
        features = torch.stack([expert(z) for expert in experts], dim=1)
        logits = heads(features)
        loss = expert_predictive_loss(logits, torch.tensor([1]), gamma, True)
        loss.backward()
        self.assertGreater(backbone.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(router.weight.grad)
        selected = gamma[0].nonzero().flatten().tolist()
        for index in range(3):
            expert_norm = experts[index].weight.grad.abs().sum().item()
            head_norm = heads.heads[index].weight.grad.abs().sum().item()
            if index in selected:
                self.assertGreater(expert_norm, 0)
                self.assertGreater(head_norm, 0)
            else:
                self.assertEqual(expert_norm, 0)
                self.assertEqual(head_norm, 0)


class TestAssignmentAndDistillation(unittest.TestCase):
    def test_risk_ema_and_topr_assignment_are_source_only(self):
        risks = torch.tensor([[0.1, 1.0, 2.0], [1.2, 0.2, 0.8]])
        valid = torch.ones_like(risks, dtype=torch.bool)
        ema = SourceRiskEMA(2, 3, beta=0.9)
        ema.update(risks, valid)
        q = top_r_assignment(ema.risk, ema.valid, top_r=2, temperature=1)
        self.assertEqual(q.shape[0], 2)  # only the two supplied source domains
        torch.testing.assert_close(q.sum(1), torch.ones(2))
        torch.testing.assert_close((q > 0).sum(1), torch.full((2,), 2))
        self.assertTrue(q[0, 0] > q[0, 1])
        self.assertTrue(q[1, 1] > q[1, 2])
        self.assertFalse(q.requires_grad)
        self.assertTrue(q[q == 0].eq(0).all())

    def test_assignment_ignores_invalid_entries(self):
        risk = torch.tensor([[0.1, float("nan"), 0.4]])
        valid = torch.tensor([[True, True, False]])
        q = top_r_assignment(risk, valid, top_r=2)
        torch.testing.assert_close(q, torch.tensor([[1.0, 0.0, 0.0]]))

    def test_router_distillation_gradients_and_zero_teacher_entries(self):
        router = nn.Linear(4, 3)
        expert_head = nn.Linear(4, 2)
        x = torch.randn(6, 4)
        pi = torch.softmax(router(x), dim=1)
        q = torch.tensor([[0.7, 0.3, 0.0], [0.0, 0.4, 0.6]])
        domains = torch.tensor([0, 0, 0, 1, 1, 1])
        loss = router_distillation_loss(pi, domains, q)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(router.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(expert_head.weight.grad)

        close_pi = q[domains].clamp_min(1e-7)
        close_pi = close_pi / close_pi.sum(1, keepdim=True)
        far_pi = torch.full_like(close_pi, 1 / 3)
        self.assertLess(
            router_distillation_loss(close_pi, domains, q).item(),
            router_distillation_loss(far_pi, domains, q).item(),
        )


class TestTargetCheckpointDiagnostics(unittest.TestCase):
    def test_uses_last_checkpoint_and_target_out_peak(self):
        records = [
            {"step": 0, "env1_in_acc": 0.6, "env1_out_acc": 0.7},
            {"step": 1, "env1_in_acc": 0.8, "env1_out_acc": 0.65},
        ]

        diagnostics = target_checkpoint_diagnostics(records, target_env=1)

        self.assertEqual(diagnostics["domainbed_oracle"]["step"], 1)
        self.assertEqual(
            diagnostics["target_out_peak_diagnostic"]["step"], 0
        )

    def test_rejects_empty_checkpoint_history(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            target_checkpoint_diagnostics([], target_env=1)


class TestQCapacity(unittest.TestCase):
    def test_matches_hinge_capacity_formula_and_has_gradient(self):
        probabilities = torch.tensor([
            [0.7, 0.3, 0.0], [0.5, 0.5, 0.0],
            [0.1, 0.7, 0.2], [0.3, 0.5, 0.2],
        ], requires_grad=True)
        domains = torch.tensor([0, 0, 1, 1])
        loss, info = q_capacity_loss(
            probabilities, domains, num_domains=2, rho_max=0.35
        )
        expected_mean = torch.tensor([0.4, 0.5, 0.1])
        expected = torch.relu(expected_mean - 0.35).square().mean()
        torch.testing.assert_close(info["mean_mass"], expected_mean)
        torch.testing.assert_close(loss, expected)
        self.assertEqual(info["violating_expert_count"], 2)
        loss.backward()
        self.assertTrue(torch.isfinite(probabilities.grad).all())
        self.assertGreater(probabilities.grad.abs().sum().item(), 0)

    def test_capacity_updates_router_not_expert_head(self):
        torch.manual_seed(7)
        router = nn.Linear(4, 3)
        expert_head = nn.Linear(4, 2)
        x = torch.randn(8, 4)
        domains = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        probabilities = torch.softmax(router(x), dim=1)
        loss, _ = q_capacity_loss(
            probabilities, domains, num_domains=2, rho_max=0.2
        )
        loss.backward()
        self.assertGreater(router.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(expert_head.weight.grad)

    def test_zero_when_no_expert_exceeds_capacity(self):
        probabilities = torch.full((6, 3), 1 / 3, requires_grad=True)
        domains = torch.tensor([0, 0, 0, 1, 1, 1])
        loss, info = q_capacity_loss(
            probabilities, domains, num_domains=2, rho_max=0.4
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(info["violating_expert_count"], 0)
        loss.backward()
        self.assertTrue(probabilities.grad.eq(0).all())


class TestQGatedSinkhorn(unittest.TestCase):
    def test_sinkhorn_divergence_identity_and_shift(self):
        torch.manual_seed(4)
        first = torch.randn(5, 4, requires_grad=True)
        identity, diagnostics = sinkhorn_divergence(
            first, first, epsilon=0.1, sinkhorn_iters=40
        )
        shifted, _ = sinkhorn_divergence(
            first, first.detach() + torch.tensor([0.5, 0.0, 0.0, 0.0]),
            epsilon=0.1, sinkhorn_iters=40,
        )
        self.assertLess(identity.item(), 1e-6)
        self.assertGreater(shifted.item(), identity.item())
        self.assertLess(diagnostics["marginal_error"].item(), 1e-4)
        shifted.backward()
        self.assertTrue(torch.isfinite(first.grad).all())

    def test_q_gate_updates_only_expert_shared_by_both_domains(self):
        torch.manual_seed(5)
        features = torch.randn(12, 2, 4, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 1, 1, 1] * 2)
        domains = torch.tensor([0] * 6 + [1] * 6)
        with torch.no_grad():
            features[6:, 0] += torch.tensor([0.8, 0.0, 0.0, 0.0])
        q = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0]], requires_grad=True
        )
        loss, info = q_gated_sinkhorn_divergence(
            features, labels, domains, q, num_classes=2,
            epsilon=0.1, sinkhorn_iters=40, min_samples=2,
        )
        self.assertTrue(info["active"])
        self.assertEqual(info["active_pair_count"], 2)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertGreater(features.grad[:, 0].abs().sum().item(), 0)
        self.assertEqual(features.grad[:, 1].abs().sum().item(), 0)
        self.assertIsNone(q.grad)

    def test_no_q_overlap_returns_differentiable_zero(self):
        features = torch.randn(8, 2, 3, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1] * 2)
        domains = torch.tensor([0] * 4 + [1] * 4)
        q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss, info = q_gated_sinkhorn_divergence(
            features, labels, domains, q, num_classes=2,
            min_samples=2,
        )
        self.assertFalse(info["active"])
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(features.grad.eq(0).all())


class TestSubsetIRMPenalty(unittest.TestCase):
    def _case(self, conflicting=False):
        labels = torch.tensor([0, 1, 0, 1])
        domains = torch.tensor([0, 0, 1, 1])
        shared = torch.tensor([
            [[5.0, -5.0]], [[-5.0, 5.0]],
            [[5.0, -5.0]], [[-5.0, 5.0]],
        ])
        if conflicting:
            shared[2:] = -shared[2:]
        logits = shared.clone().requires_grad_(True)
        gamma = torch.ones(4, 1)
        q = torch.ones(2, 1)
        return logits, labels, domains, gamma, q

    def test_shared_mechanism_smaller_than_conflicting_and_optimizable(self):
        shared = self._case(False)
        conflict = self._case(True)
        p_shared, info_shared = subset_irm_penalty(*shared, 2)
        p_conflict, info_conflict = subset_irm_penalty(*conflict, 2)
        self.assertTrue(info_shared["active"])
        self.assertTrue(info_conflict["active"])
        self.assertLess(p_shared.item(), p_conflict.item())

        parameter = nn.Parameter(conflict[0].detach().clone())
        optimizer = torch.optim.SGD([parameter], lr=0.01)
        before = None
        for _ in range(3):
            optimizer.zero_grad()
            penalty, _ = subset_irm_penalty(
                parameter, *conflict[1:], min_effective_samples=2
            )
            if before is None:
                before = penalty.item()
            penalty.backward()
            optimizer.step()
        after = subset_irm_penalty(
            parameter, *conflict[1:], min_effective_samples=2
        )[0].item()
        self.assertLess(after, before)

    def test_inactive_cases_return_finite_zero(self):
        logits, labels, domains, gamma, q = self._case(False)
        one_domain = domains.zero_()
        penalty, info = subset_irm_penalty(
            logits, labels, one_domain, gamma, q, 2
        )
        self.assertFalse(info["active"])
        self.assertEqual(penalty.item(), 0)
        penalty, info = subset_irm_penalty(
            logits, labels, domains, gamma * 0, q, 2
        )
        self.assertFalse(info["active"])
        self.assertTrue(torch.isfinite(penalty))

    def test_finite_first_second_order_and_no_router_gradient(self):
        logits, labels, domains, _, q = self._case(True)
        router_logits = torch.randn(4, 1, requires_grad=True)
        gamma = torch.softmax(router_logits, dim=1)
        penalty, _ = subset_irm_penalty(
            logits, labels, domains, gamma, q, 2,
            detach_responsibility=True,
        )
        first = torch.autograd.grad(
            penalty, logits, create_graph=True, retain_graph=True
        )[0]
        second = torch.autograd.grad(first.sum(), logits, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.isfinite(second).all())
        penalty.backward()
        self.assertIsNone(router_logits.grad)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMESSIIntegration(unittest.TestCase):
    def test_encoder_classifier_skip_adds_to_top2_moe_logits(self):
        hparams = {
            "model": "deit_tiny_patch16_224",
            "pretrained": False,
            "legacy_non_distilled_deit": False,
            "moe_dim": "auto",
            "num_experts": 6,
            "expert_mlp_ratio": 2,
            "lr": 3e-5,
            "weight_decay": 1e-6,
            "lambda_inv": 0.0,
            "lambda_sp": 0.0,
            "lambda_bal": 0.0,
            "lambda_div": 0.0,
            "alpha": 4.0,
            "subset_irm_enabled": True,
            "subset_irm_prediction_mode": "expert_logit_mix",
            "subset_irm_router_topk": 2,
            "subset_irm_topk_warmup_steps": 0,
            "subset_irm_use_expert_heads": True,
            "subset_irm_encoder_skip_enabled": True,
            "subset_irm_encoder_skip_scale": 0.75,
            "subset_irm_lambda_expert": 0.0,
            "subset_irm_lambda_route": 0.0,
            "subset_irm_lambda_sirm": 0.0,
        }
        model = MESSISubsetIRM((3, 224, 224), 7, 3, hparams).eval()
        torch.testing.assert_close(
            model.encoder_skip_classifier.weight,
            model.moe_head.classifier.weight,
        )
        torch.testing.assert_close(
            model.encoder_skip_classifier.bias,
            model.moe_head.classifier.bias,
        )

        x = torch.randn(2, 3, 224, 224, device="cuda")
        output = model._subset_forward(x)
        logits, gamma, moe_logits, skip_logits = (
            output[0], output[2], output[6], output[7]
        )
        torch.testing.assert_close(
            logits, moe_logits + 0.75 * skip_logits
        )
        torch.testing.assert_close(
            gamma.gt(0).sum(dim=1),
            torch.full((2,), 2, device="cuda"),
        )
        F.cross_entropy(logits, torch.tensor([0, 1], device="cuda")).backward()
        self.assertGreater(
            model.encoder_skip_classifier.weight.grad.abs().sum().item(), 0
        )

        model.encoder_skip_enabled = False
        without_skip = model._subset_forward(x)
        torch.testing.assert_close(without_skip[0], without_skip[6])

    def test_loss_schedule_changes_only_at_configured_steps(self):
        hparams = {
            "model": "deit_tiny_patch16_224",
            "pretrained": False,
            "legacy_non_distilled_deit": False,
            "moe_dim": "auto",
            "num_experts": 6,
            "expert_mlp_ratio": 2,
            "lr": 3e-5,
            "weight_decay": 1e-6,
            "lambda_inv": 0.0,
            "lambda_sp": 0.0,
            "lambda_bal": 0.02,
            "lambda_div": 0.0,
            "alpha": 4.0,
            "subset_irm_enabled": True,
            "subset_irm_prediction_mode": "expert_logit_mix",
            "subset_irm_router_topk": 2,
            "subset_irm_topk_warmup_steps": 500,
            "subset_irm_use_expert_heads": True,
            "subset_irm_lambda_expert": 0.1,
            "subset_irm_lambda_route": 0.06,
            "subset_irm_route_anneal_steps": 500,
            "subset_irm_route_ramp_steps": 500,
            "subset_irm_lambda_sirm": 1.0,
            "subset_irm_sirm_anneal_steps": 500,
            "subset_irm_sirm_ramp_steps": 500,
            "subset_irm_post_warmup_lambda_sp": 0.0,
            "subset_irm_post_warmup_lambda_bal": 0.0,
        }
        model = MESSISubsetIRM((3, 224, 224), 7, 3, hparams)
        model.subset_step.fill_(499)
        before = model.update([
            (torch.randn(2, 3, 224, 224, device="cuda"),
             torch.tensor([0, 1], device="cuda")) for _ in range(3)
        ])
        self.assertIn("loss_cls", before)
        self.assertNotIn("loss_mix", before)
        self.assertEqual(before["effective_lambda_route"], 0.0)
        self.assertEqual(before["effective_lambda_sirm"], 0.0)
        self.assertEqual(before["effective_lambda_bal"], 0.02)
        model.subset_step.fill_(750)
        middle = model.update([
            (torch.randn(2, 3, 224, 224, device="cuda"),
             torch.tensor([0, 1], device="cuda")) for _ in range(3)
        ])
        self.assertEqual(middle["effective_lambda_route"], 0.03)
        self.assertEqual(middle["effective_lambda_sirm"], 0.5)
        self.assertEqual(middle["effective_lambda_bal"], 0.0)

    def test_auxiliary_mode_matches_legacy_prediction(self):
        hparams = {
            "model": "deit_tiny_patch16_224",
            "pretrained": False,
            "legacy_non_distilled_deit": False,
            "moe_dim": "auto",
            "num_experts": 6,
            "expert_mlp_ratio": 2,
            "lr": 3e-5,
            "weight_decay": 1e-6,
            "lambda_inv": 0.0,
            "lambda_sp": 0.0,
            "lambda_bal": 0.0,
            "lambda_div": 0.0,
            "alpha": 4.0,
            "subset_irm_enabled": True,
            "subset_irm_prediction_mode": "shared_feature_mix",
            "subset_irm_use_expert_heads": True,
            "subset_irm_lambda_expert": 0.0,
            "subset_irm_lambda_route": 0.0,
            "subset_irm_lambda_sirm": 0.0,
        }
        model = MESSISubsetIRM((3, 224, 224), 7, 3, hparams).eval()
        x = torch.randn(2, 3, 224, 224, device="cuda")
        with torch.no_grad():
            z = model.featurizer(x)
            legacy = model.moe_head(z)[0]
            subset = model._subset_forward(x)[0]
        torch.testing.assert_close(subset, legacy, rtol=1e-6, atol=1e-6)
        self.assertEqual(len(model.expert_heads.heads), 6)
        self.assertTrue(all(
            next(head.parameters()).device.type == "cuda"
            for head in model.expert_heads.heads
        ))


if __name__ == "__main__":
    unittest.main()
