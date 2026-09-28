# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""Unit tests."""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import unittest
import uuid

import torch

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed import networks
from domainbed.scripts import train as train_script

try:
    from parameterized import parameterized
except ImportError:
    class _ParameterizedFallback:
        @staticmethod
        def expand(cases):
            def decorate(fn):
                def wrapped(self):
                    for case in cases:
                        if not isinstance(case, tuple):
                            case = (case,)
                        with self.subTest(case=case):
                            fn(self, *case)
                return wrapped
            return decorate

    parameterized = _ParameterizedFallback()

from domainbed.test import helpers

class TestDatasets(unittest.TestCase):

    def test_iwildcam_erm_dataset_registered(self):
        cls = datasets.get_dataset_class("WILDSIWildCamERM")
        self.assertIs(cls, datasets.WILDSIWildCamERM)
        self.assertEqual(datasets.num_environments("WILDSIWildCamERM"), 5)

    def test_iwildcam_erm_dataset_disables_group_sampler(self):
        self.assertIsNone(getattr(datasets.WILDSIWildCamERM, "GROUP_SAMPLER_K", None))
        self.assertIsNone(getattr(datasets.WILDSIWildCamERM, "GROUP_SAMPLER_K_BATCH", None))
        self.assertEqual(datasets.WILDSIWildCam.GROUP_SAMPLER_K, 4)
        self.assertEqual(datasets.WILDSIWildCam.GROUP_SAMPLER_K_BATCH, 8)

    def test_iwildcam_erm_default_eval_metric(self):
        self.assertEqual(train_script._default_eval_metric("WILDSIWildCamERM"), "f1")
        self.assertEqual(train_script._default_eval_metric("WILDSIWildCam"), "f1")
        self.assertEqual(train_script._default_eval_metric("PACS"), "acc")

    @parameterized.expand(itertools.product(datasets.DATASETS))
    @unittest.skipIf('DATA_DIR' not in os.environ, 'needs DATA_DIR environment '
        'variable')
    def test_dataset_erm(self, dataset_name):
        """
        Test that ERM can complete one step on a given dataset without raising
        an error.
        Also test that num_environments() works correctly.
        """
        batch_size = 8
        hparams = hparams_registry.default_hparams('ERM', dataset_name)
        dataset = datasets.get_dataset_class(dataset_name)(
            os.environ['DATA_DIR'], [], hparams)
        self.assertEqual(datasets.num_environments(dataset_name),
                         len(dataset))
        algorithm = algorithms.get_algorithm_class('ERM')(
            dataset.input_shape,
            dataset.num_classes,
            len(dataset),
            hparams).cuda()
        minibatches = helpers.make_minibatches(dataset, batch_size)
        algorithm.update(minibatches)
