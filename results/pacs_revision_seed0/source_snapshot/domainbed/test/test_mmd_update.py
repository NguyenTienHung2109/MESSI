import math
import unittest

import torch

from domainbed import algorithms
from domainbed import datasets
from domainbed import hparams_registry


class TestMMDUpdate(unittest.TestCase):
    def _make_minibatches(self, dataset):
        minibatches = []
        for env_i, env in enumerate(dataset):
            batch_size = 3 + env_i
            x = torch.stack([env[i][0] for i in range(batch_size)]).cuda()
            y = torch.stack([
                torch.as_tensor(env[i][1]) for i in range(batch_size)
            ]).cuda()
            minibatches.append((x, y))
        return minibatches

    def _check_update(self, algorithm_name):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required by the current algorithm tests")

        hparams = hparams_registry.default_hparams(algorithm_name, "Debug224")
        hparams["model"] = "cnn"
        dataset = datasets.get_dataset_class("Debug224")('', [], hparams)
        minibatches = self._make_minibatches(dataset)

        algorithm_class = algorithms.get_algorithm_class(algorithm_name)
        algorithm = algorithm_class(
            dataset.input_shape, dataset.num_classes, len(dataset), hparams
        ).cuda()
        params_before = [
            p.detach().clone() for p in algorithm.parameters()
            if p.requires_grad
        ]

        update_vals = algorithm.update(minibatches)

        self.assertIn("loss", update_vals)
        self.assertIn("penalty", update_vals)
        self.assertTrue(math.isfinite(update_vals["loss"]))
        self.assertTrue(math.isfinite(update_vals["penalty"]))
        self.assertEqual(
            list(algorithm.predict(minibatches[0][0]).shape),
            [len(minibatches[0][0]), dataset.num_classes],
        )
        params_after = [
            p.detach() for p in algorithm.parameters()
            if p.requires_grad
        ]
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        ))

    def test_coral_update_with_concatenated_domain_batches(self):
        self._check_update("CORAL")

    def test_mmd_update_with_concatenated_domain_batches(self):
        self._check_update("MMD")
