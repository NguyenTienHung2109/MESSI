"""Strict inductive single-query form of the authors' ECALP graph.

ECALP normally propagates labels through an expanding graph of earlier test
queries. Here each query is evaluated with a fresh two-copy query graph,
matching the first call in the author implementation. No target-query
history or target labels can influence later predictions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .primitives import Classifier, _matrix


class ECALPInductive(Classifier):
    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        text = _matrix(config["text_features"])
        if text.shape != (self.num_classes, self.dim):
            raise ValueError("ECALP text shape mismatch")
        self.device = torch.device(config.get("ecalp_device", "cpu"))
        self.support = F.normalize(torch.as_tensor(x, device=self.device,
                                                   dtype=torch.float32), dim=1)
        self.text = F.normalize(torch.as_tensor(text, device=self.device,
                                               dtype=torch.float32), dim=1)
        self.labels = torch.as_tensor(labels, device=self.device, dtype=torch.long)
        self.k_text = min(int(config.get("ecalp_k_text", 3)), self.num_classes)
        self.k_support = min(int(config.get("ecalp_k_fewshot", 8)), len(labels))
        self.gamma = float(config.get("ecalp_gamma", 10.0))
        self.alpha = float(config.get("ecalp_alpha", 1.0))
        self.iterations = int(config.get("ecalp_iterations", 3))
        if self.k_text < 1 or self.k_support < 1 or self.gamma <= 0 or not 0 < self.alpha <= 1:
            raise ValueError("Invalid ECALP configuration")
        if self.iterations < 1:
            raise ValueError("ECALP iterations must be positive")
        # Author edge_reweighting uses 1 / per-dimension few-shot variance.
        inverse_var = self.support.var(dim=0, unbiased=True).clamp_min(1e-12).reciprocal()
        self.weighted_support = self.support
        self.inverse_var = inverse_var
        self.text_variance = F.normalize(self.text.var(dim=0, unbiased=True), dim=0)
        return self

    def predict_scores(self, query_features):
        query = torch.as_tensor(self._query(query_features), device=self.device,
                                dtype=torch.float32)
        with torch.inference_mode():
            query = F.normalize(query, dim=1)
            text_similarity = query @ self.text.T
            text_sim, text_idx = text_similarity.topk(self.k_text, dim=1)
            weighted_query = F.normalize(query * self.inverse_var, dim=1)
            support_similarity = weighted_query @ self.weighted_support.T
            support_sim, support_idx = support_similarity.topk(self.k_support, dim=1)
            text_weights = text_sim.pow(self.gamma)
            support_weights = support_sim.pow(self.gamma)
            # Author edge_reweighting scales Q by text variance but leaves X
            # unweighted, even when both are copies of the same image.
            image_similarity = (F.normalize(query * self.text_variance, dim=1) * query).sum(dim=1)
            image_weight = image_similarity.pow(self.gamma)
            # Symmetric graph: two query nodes have two image edges each.
            # Each selected text/support anchor gets an edge from both copies.
            query_degree = 4.0 * image_weight + text_weights.sum(dim=1) + support_weights.sum(dim=1)
            text_contrib = text_weights.clamp_min(0).sqrt() / (2 * query_degree[:, None]).sqrt()
            support_contrib = support_weights.clamp_min(0).sqrt() / (2 * query_degree[:, None]).sqrt()
            scores = torch.zeros(len(query), self.num_classes, device=self.device)
            scores.scatter_add_(1, text_idx, text_contrib)
            scores.scatter_add_(1, self.labels[support_idx], support_contrib)
            query_eigenvalue = 4.0 * image_weight / query_degree
            factor = torch.zeros_like(query_eigenvalue)
            for step in range(self.iterations):
                factor += self.alpha ** (step + 1) * query_eigenvalue.pow(step)
            scores *= factor[:, None]
            # The author returns softmax(propagated scores * 10000).
            scores = torch.softmax(scores * 10000.0, dim=1)
        return scores.cpu().numpy()
