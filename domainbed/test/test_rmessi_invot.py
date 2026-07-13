import math
import unittest
from unittest import mock

import torch

from domainbed import algorithms, datasets, deit_transformer, hparams_registry
from domainbed.losses import gmoe_utils


class TestReliableInvOTLoss(unittest.TestCase):
    def _toy_inputs(self):
        torch.manual_seed(0)
        h_stack = torch.randn(6, 2, 3)
        pi = torch.softmax(torch.randn(6, 2), dim=-1)
        y = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
        domain_ids = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long)
        return h_stack, pi, y, domain_ids

    def test_algorithm_registered_and_hparams_exist(self):
        self.assertIs(algorithms.get_algorithm_class('rMESSI_InvOT'), algorithms.rMESSI_InvOT)
        hparams = hparams_registry.default_hparams('rMESSI_InvOT', 'PACS')
        self.assertEqual(hparams['reliability_mode'], 'none')
        self.assertIn('r_n_min', hparams)
        self.assertEqual(hparams['reliable_log_detail'], 'compact')

    def test_none_matches_legacy_invot(self):
        h_stack, pi, y, domain_ids = self._toy_inputs()
        legacy = gmoe_utils.loss_inv_OT(
            h_stack, pi, y, 1, domain_ids, 2,
            alpha=4.0, epsilon=0.1, sinkhorn_iters=3,
        )
        reliable, diag = gmoe_utils.loss_inv_OT_reliable(
            h_stack, pi, y, 1, domain_ids, 2,
            alpha=4.0, epsilon=0.1, sinkhorn_iters=3,
            reliability_mode='none',
        )
        self.assertTrue(torch.allclose(legacy, reliable, atol=1e-6))
        for key in (
            'num_valid_slots', 'slot_density', 'r_mean', 'r_nonzero_frac',
            'w_sum', 'coverage_mean', 'mask_keep_frac',
            'ot_cost_weighted_mean',
        ):
            self.assertIn(key, diag)
        self.assertNotIn('r_min', diag)
        self.assertNotIn('ema_E_valid_frac', diag)
        self.assertEqual(diag['num_valid_slots'], 2.0)
        self.assertEqual(diag['slot_density'], 1.0)
        self.assertEqual(diag['r_mean'], 1.0)
        self.assertEqual(diag['r_nonzero_frac'], 1.0)
        self.assertGreater(diag['w_sum'], 0.0)
        self.assertTrue(math.isfinite(diag['ot_cost_weighted_mean']))

    def test_mask_skips_low_count_slots(self):
        h_stack = torch.randn(2, 2, 3)
        pi = torch.softmax(torch.randn(2, 2), dim=-1)
        y = torch.tensor([0, 0], dtype=torch.long)
        domain_ids = torch.tensor([0, 1], dtype=torch.long)
        loss, diag = gmoe_utils.loss_inv_OT_reliable(
            h_stack, pi, y, 1, domain_ids, 2,
            sinkhorn_iters=2, reliability_mode='mask', r_n_min=2,
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(diag['num_valid_slots'], 0.0)
        self.assertEqual(diag['slot_density'], 0.0)
        self.assertEqual(diag['r_nonzero_frac'], 0.0)
        self.assertEqual(diag['w_sum'], 0.0)
        self.assertEqual(diag['mask_keep_frac'], 0.0)
        self.assertTrue(math.isfinite(diag['ot_cost_weighted_mean']))

    def test_coverage_weight_is_bounded_and_matches_formula(self):
        h_stack, pi, y, domain_ids = self._toy_inputs()
        _, diag = gmoe_utils.loss_inv_OT_reliable(
            h_stack, pi, y, 1, domain_ids, 2,
            sinkhorn_iters=2, reliability_mode='coverage',
            r_n_min=2, r_tau=4.0,
        )
        expected = math.sqrt((2.0 * 4.0) / ((2.0 + 4.0) * (4.0 + 4.0)))
        self.assertGreaterEqual(diag['r_mean'], 0.0)
        self.assertLessEqual(diag['r_mean'], 1.0)
        self.assertAlmostEqual(diag['r_mean'], expected, places=6)
        self.assertAlmostEqual(diag['coverage_mean'], expected, places=6)
        self.assertGreaterEqual(diag['slot_density'], 0.0)
        self.assertLessEqual(diag['slot_density'], 1.0)
        self.assertGreaterEqual(diag['r_nonzero_frac'], 0.0)
        self.assertLessEqual(diag['r_nonzero_frac'], 1.0)

    def test_ema_coverage_uses_evidence_and_handles_no_valid_slots(self):
        h_stack, pi, y, domain_ids = self._toy_inputs()
        evidence = torch.zeros(2, 1)
        loss, diag = gmoe_utils.loss_inv_OT_reliable(
            h_stack, pi, y, 1, domain_ids, 2,
            sinkhorn_iters=2, reliability_mode='ema_coverage',
            ema_evidence=evidence, r_emin=4.0,
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(math.isfinite(loss.item()))
        self.assertEqual(diag['num_valid_slots'], 0.0)
        self.assertEqual(diag['ema_E_valid_frac'], 0.0)
        self.assertEqual(diag['ema_extra_pairs'], 0.0)

        evidence[:, 0] = torch.tensor([4.0, 8.0])
        loss, diag = gmoe_utils.loss_inv_OT_reliable(
            h_stack, pi, y, 1, domain_ids, 2,
            sinkhorn_iters=2, reliability_mode='ema_coverage',
            ema_evidence=evidence, r_emin=4.0, r_tau=4.0,
        )
        expected = math.sqrt((4.0 * 8.0) / ((4.0 + 4.0) * (8.0 + 4.0)))
        self.assertTrue(math.isfinite(loss.item()))
        self.assertEqual(diag['num_valid_slots'], 2.0)
        self.assertAlmostEqual(diag['r_mean'], expected, places=6)
        self.assertEqual(diag['ema_E_valid_frac'], 1.0)
        self.assertEqual(diag['ema_extra_pairs'], 0.0)

    def test_full_log_detail_includes_legacy_diagnostics(self):
        h_stack, pi, y, domain_ids = self._toy_inputs()
        _, diag = gmoe_utils.loss_inv_OT_reliable(
            h_stack, pi, y, 1, domain_ids, 2,
            sinkhorn_iters=2, reliability_mode='coverage',
            r_n_min=2, r_tau=4.0, reliable_log_detail='full',
        )
        for key in (
            'r_min', 'r_max', 'r_sum', 'num_candidate_pairs',
            'num_valid_pairs', 'num_skipped_low_count',
            'mean_responsibility', 'mean_ot_cost',
        ):
            self.assertIn(key, diag)
        self.assertIn('w_sum', diag)

    def test_ema_state_updates_only_encountered_domains(self):
        algo = object.__new__(algorithms.rMESSI_InvOT)
        algo.reliability_mode = 'ema_coverage'
        algo.num_classes = 3
        algo.r_ema_beta = 0.5
        algo.r_ema_evidence = torch.zeros(3, 3)

        y = torch.tensor([0, 0, 2, 1], dtype=torch.long)
        domain_ids = torch.tensor([0, 0, 0, 2], dtype=torch.long)
        algorithms.rMESSI_InvOT._update_reliability_state(algo, y, domain_ids)
        self.assertTrue(torch.equal(algo.r_ema_evidence[1], torch.zeros(3)))
        self.assertEqual(algo.r_ema_evidence[0, 0].item(), 2.0)
        self.assertEqual(algo.r_ema_evidence[0, 2].item(), 1.0)
        self.assertEqual(algo.r_ema_evidence[2, 1].item(), 1.0)

        algorithms.rMESSI_InvOT._update_reliability_state(algo, y, domain_ids)
        self.assertTrue(torch.equal(algo.r_ema_evidence[1], torch.zeros(3)))
        self.assertEqual(algo.r_ema_evidence[0, 0].item(), 3.0)
        self.assertEqual(algo.r_ema_evidence[0, 1].item(), 0.0)
        self.assertEqual(algo.r_ema_evidence[0, 2].item(), 1.5)
        self.assertEqual(algo.r_ema_evidence[2, 0].item(), 0.0)
        self.assertEqual(algo.r_ema_evidence[2, 1].item(), 1.5)
        self.assertEqual(algo.r_ema_evidence[2, 2].item(), 0.0)

    def test_ema_state_decays_unobserved_classes_for_encountered_domain(self):
        algo = object.__new__(algorithms.rMESSI_InvOT)
        algo.reliability_mode = 'ema_coverage'
        algo.num_classes = 3
        algo.r_ema_beta = 0.5
        algo.r_ema_evidence = torch.tensor([
            [4.0, 6.0, 8.0],
            [5.0, 7.0, 9.0],
        ])

        y = torch.tensor([0, 0], dtype=torch.long)
        domain_ids = torch.tensor([0, 0], dtype=torch.long)
        algorithms.rMESSI_InvOT._update_reliability_state(algo, y, domain_ids)

        self.assertTrue(torch.equal(algo.r_ema_evidence[1], torch.tensor([5.0, 7.0, 9.0])))
        self.assertTrue(torch.equal(algo.r_ema_evidence[0], torch.tensor([4.0, 3.0, 4.0])))

    def test_ema_buffer_shape_on_algorithm_init(self):
        def fake_base_init(self, input_shape, num_classes, num_domains, hparams):
            torch.nn.Module.__init__(self)
            self.num_classes = num_classes
            self.num_domains = num_domains
            self.num_experts = hparams.get('num_experts', 2)

        hparams = {
            'num_experts': 2,
            'reliability_mode': 'ema_coverage',
        }
        with mock.patch.object(algorithms.GMoEVariantBase, '__init__', fake_base_init):
            algo = algorithms.rMESSI_InvOT((3, 224, 224), 5, 7, hparams)

        self.assertEqual(tuple(algo.r_ema_evidence.shape), (7, 5))
        self.assertFalse(algo.r_ema_evidence.requires_grad)

    def test_ema_pre_update_semantics_requires_next_batch_to_use_new_evidence(self):
        h_stack = torch.randn(2, 2, 3)
        pi = torch.softmax(torch.randn(2, 2), dim=-1)
        y = torch.tensor([0, 0], dtype=torch.long)
        domain_ids = torch.tensor([0, 1], dtype=torch.long)
        evidence = torch.tensor([[3.5], [3.5]])

        loss, diag = gmoe_utils.loss_inv_OT_reliable(
            h_stack, pi, y, 1, domain_ids, 2,
            sinkhorn_iters=2, reliability_mode='ema_coverage',
            ema_evidence=evidence, r_emin=4.0, r_n_min=2,
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(diag['num_valid_slots'], 0.0)

        algo = object.__new__(algorithms.rMESSI_InvOT)
        algo.reliability_mode = 'ema_coverage'
        algo.num_classes = 1
        algo.r_ema_beta = 0.9
        algo.r_ema_evidence = evidence.clone()
        algorithms.rMESSI_InvOT._update_reliability_state(algo, y, domain_ids)
        self.assertTrue(torch.allclose(algo.r_ema_evidence, torch.tensor([[4.15], [4.15]])))

        loss, diag = gmoe_utils.loss_inv_OT_reliable(
            h_stack, pi, y, 1, domain_ids, 2,
            sinkhorn_iters=2, reliability_mode='ema_coverage',
            ema_evidence=algo.r_ema_evidence, r_emin=4.0, r_n_min=2,
        )
        self.assertTrue(math.isfinite(loss.item()))
        self.assertEqual(diag['num_valid_slots'], 2.0)

    def test_ema_coverage_rescues_sparse_current_pair_and_logs_pair_level_counts(self):
        h_stack = torch.randn(4, 3, 2)
        pi = torch.softmax(torch.randn(4, 3), dim=-1)
        y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        domain_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        evidence = torch.tensor([[4.0, 8.0], [4.0, 8.0]])

        loss, diag = gmoe_utils.loss_inv_OT_reliable(
            h_stack, pi, y, 2, domain_ids, 2,
            sinkhorn_iters=2, reliability_mode='ema_coverage',
            ema_evidence=evidence, r_emin=4.0, r_tau=4.0, r_n_min=2,
            reliable_log_detail='full',
        )

        self.assertTrue(math.isfinite(loss.item()))
        self.assertEqual(diag['num_valid_slots'], 6.0)
        self.assertEqual(diag['ema_E_valid_frac'], 1.0)
        self.assertEqual(diag['ema_keep_frac'], 1.0)
        self.assertEqual(diag['ema_extra_pairs'], 2.0)
        self.assertEqual(diag['ema_extra_frac'], 1.0)
        self.assertEqual(diag['mask_keep_frac'], 0.0)
        self.assertGreaterEqual(diag['r_mean'], 0.0)
        self.assertLessEqual(diag['r_mean'], 1.0)

    def test_ema_coverage_requires_current_samples_on_both_sides(self):
        h_stack = torch.randn(2, 2, 3)
        pi = torch.softmax(torch.randn(2, 2), dim=-1)
        y = torch.tensor([0, 0], dtype=torch.long)
        domain_ids = torch.tensor([0, 0], dtype=torch.long)
        evidence = torch.tensor([[100.0], [100.0]])

        loss, diag = gmoe_utils.loss_inv_OT_reliable(
            h_stack, pi, y, 1, domain_ids, 2,
            sinkhorn_iters=2, reliability_mode='ema_coverage',
            ema_evidence=evidence, r_emin=4.0,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(diag['num_valid_slots'], 0.0)
        self.assertEqual(diag['ema_E_valid_frac'], 0.0)

    def test_reserved_reliability_modes_raise(self):
        h_stack, pi, y, domain_ids = self._toy_inputs()
        for mode in ('ema_coverage_conf', 'full'):
            with self.assertRaises(NotImplementedError):
                gmoe_utils.loss_inv_OT_reliable(
                    h_stack, pi, y, 1, domain_ids, 2,
                    sinkhorn_iters=2, reliability_mode=mode,
                )


class TestMoEBottleneck(unittest.TestCase):
    def test_explicit_moe_head_bottleneck_shapes_and_backward(self):
        head = gmoe_utils.ExplicitMoEHead(
            in_dim=2048, moe_dim=384, num_experts=6, num_classes=182,
        )
        x = torch.randn(3, 2048)
        logits, pi, h_stack = head(x)

        self.assertEqual(tuple(logits.shape), (3, 182))
        self.assertEqual(tuple(pi.shape), (3, 6))
        self.assertEqual(tuple(h_stack.shape), (3, 6, 384))
        self.assertEqual(head.expert_dim, 384)
        self.assertIsInstance(head.input_proj, torch.nn.Linear)

        loss = logits.mean() + pi.mean() + h_stack.mean()
        loss.backward()
        self.assertIsNotNone(head.input_proj.weight.grad)
        self.assertIsNotNone(head.experts[0][0].weight.grad)
        self.assertIsNotNone(head.router.weight.grad)
        self.assertIsNotNone(head.classifier.weight.grad)

    def test_explicit_moe_head_identity_when_dims_match(self):
        for dim in (384, 192):
            head = gmoe_utils.ExplicitMoEHead(
                in_dim=dim, moe_dim=dim, num_experts=2, num_classes=5,
            )
            x = torch.randn(4, dim)
            logits, pi, h_stack = head(x)
            self.assertIsInstance(head.input_proj, torch.nn.Identity)
            self.assertEqual(head.expert_dim, dim)
            self.assertEqual(tuple(logits.shape), (4, 5))
            self.assertEqual(tuple(pi.shape), (4, 2))
            self.assertEqual(tuple(h_stack.shape), (4, 2, dim))

    def test_resolve_moe_dim_auto_preserves_non_resnet(self):
        self.assertEqual(
            algorithms.GMoEVariantBase._resolve_moe_dim('auto', 'resnet50', 2048),
            384,
        )
        self.assertEqual(
            algorithms.GMoEVariantBase._resolve_moe_dim('auto', 'deit_small_patch16_224', 384),
            384,
        )
        self.assertEqual(
            algorithms.GMoEVariantBase._resolve_moe_dim('auto', 'deit_tiny_patch16_224', 192),
            192,
        )
        for raw in (None, 0, 'none', 'None'):
            self.assertEqual(
                algorithms.GMoEVariantBase._resolve_moe_dim(raw, 'resnet50', 2048),
                2048,
            )
        self.assertEqual(
            algorithms.GMoEVariantBase._resolve_moe_dim(512, 'resnet50', 2048),
            512,
        )

    def test_algorithm_resnet_uses_bottleneck_and_can_disable_it(self):
        class FakeResNet(torch.nn.Module):
            n_outputs = 2048
            def __init__(self, input_shape, hparams):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
            def forward(self, x):
                return torch.zeros(x.size(0), self.n_outputs, device=x.device)

        base_hparams = {
            'model': 'resnet50', 'moe_dim': 'auto', 'num_experts': 2,
            'lr': 1e-3, 'weight_decay': 0.0,
        }
        with mock.patch.object(torch.nn.Module, 'cuda', lambda self: self), \
             mock.patch.object(algorithms.networks, 'ResNet', FakeResNet):
            algo = algorithms.GMOE_InvOT((3, 448, 448), 7, 5, dict(base_hparams))
            self.assertEqual(algo.featurizer.n_outputs, 2048)
            self.assertEqual(algo.moe_head.expert_dim, 384)

            no_bottleneck = dict(base_hparams)
            no_bottleneck['moe_dim'] = 0
            algo_no = algorithms.GMOE_InvOT((3, 448, 448), 7, 5, no_bottleneck)
            self.assertEqual(algo_no.moe_head.expert_dim, 2048)

            params_bottleneck = sum(p.numel() for p in algo.parameters())
            params_no_bottleneck = sum(p.numel() for p in algo_no.parameters())
            self.assertLess(params_bottleneck, params_no_bottleneck)
            self.assertLess(params_bottleneck, 80_000_000)

    def test_algorithm_deit_tiny_auto_preserves_backbone_dim(self):
        class FakeDeiT(torch.nn.Module):
            n_outputs = 192
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
            def forward(self, x):
                return torch.zeros(x.size(0), self.n_outputs, device=x.device)

        hparams = {
            'model': 'deit_tiny_patch16_224', 'num_experts': 2,
            'lr': 1e-3, 'weight_decay': 0.0,
        }
        with mock.patch.object(torch.nn.Module, 'cuda', lambda self: self), \
             mock.patch.object(algorithms, 'DeiTFeaturizer', FakeDeiT):
            algo = algorithms.rMESSI_InvOT((3, 224, 224), 7, 5, hparams)
        self.assertEqual(algo.featurizer.n_outputs, 192)
        self.assertEqual(algo.moe_head.expert_dim, 192)


class TestDeiTResolution(unittest.TestCase):
    def test_resolution_precedence_helper(self):
        self.assertEqual(datasets._resolve_image_size_from_hparams({}), 224)
        self.assertEqual(
            datasets._resolve_image_size_from_hparams({'image_size': 448}),
            448,
        )
        self.assertEqual(
            datasets._resolve_image_size_from_hparams({'image_size': 224, 'resolution': 448}),
            448,
        )
        self.assertEqual(
            datasets._resolve_image_size_from_hparams({'image_size': 448, 'resolution': 'auto'}),
            448,
        )

    def test_resolution_hparam_default_does_not_mask_image_size(self):
        hparams = hparams_registry.default_hparams('rMESSI_InvOT', 'WILDSIWildCam')
        self.assertIsNone(hparams['resolution'])
        hparams['image_size'] = 448
        self.assertEqual(datasets._resolve_image_size_from_hparams(hparams), 448)

    def test_resize_pos_embed_deit_preserves_prefix_tokens(self):
        torch.manual_seed(0)
        pos_embed = torch.randn(1, 198, 384)
        resized = deit_transformer.resize_pos_embed_deit(
            pos_embed, old_grid_size=14, new_grid_size=28,
            num_prefix_tokens=2,
        )
        self.assertEqual(tuple(resized.shape), (1, 786, 384))
        self.assertTrue(torch.equal(resized[:, :2, :], pos_embed[:, :2, :]))
        self.assertFalse(torch.equal(resized[:, 2:198, :], pos_embed[:, 2:, :]))

    def test_deit_featurizer_interpolates_pos_embed_at_448(self):
        torch.manual_seed(0)
        pos_embed = torch.randn(1, 198, 384)
        checkpoint = {'model': {'pos_embed': pos_embed.clone()}}
        with mock.patch.object(deit_transformer.os.path, 'exists', return_value=False), \
             mock.patch.object(deit_transformer.torch.hub, 'load_state_dict_from_url', return_value=checkpoint):
            featurizer = deit_transformer.DeiTFeaturizer(
                pretrained=True, model_name='deit_small_patch16_224',
                img_size=448, patch_size=16, in_chans=3,
            )
        self.assertEqual(tuple(featurizer.vit.pos_embed.shape), (1, 786, 384))
        self.assertTrue(torch.equal(featurizer.vit.pos_embed[:, :2, :], pos_embed[:, :2, :]))
        self.assertEqual(featurizer.vit.num_tokens, 2)

    def test_deit_featurizer_224_does_not_interpolate_pos_embed(self):
        torch.manual_seed(0)
        pos_embed = torch.randn(1, 198, 384)
        checkpoint = {'model': {'pos_embed': pos_embed.clone()}}
        with mock.patch.object(deit_transformer.os.path, 'exists', return_value=False), \
             mock.patch.object(deit_transformer.torch.hub, 'load_state_dict_from_url', return_value=checkpoint):
            featurizer = deit_transformer.DeiTFeaturizer(
                pretrained=True, model_name='deit_small_patch16_224',
                img_size=224, patch_size=16, in_chans=3,
            )
        self.assertEqual(tuple(featurizer.vit.pos_embed.shape), (1, 198, 384))
        self.assertTrue(torch.equal(featurizer.vit.pos_embed, pos_embed))

    def test_deit_featurizer_forward_448_shape(self):
        torch.manual_seed(0)
        featurizer = deit_transformer.DeiTFeaturizer(
            pretrained=False, model_name='deit_small_patch16_224',
            img_size=448, patch_size=16, in_chans=3,
        )
        x = torch.randn(2, 3, 448, 448)
        with torch.no_grad():
            features = featurizer(x)
        self.assertEqual(tuple(features.shape), (2, 384))


if __name__ == '__main__':
    unittest.main()
