"""Training-free APE formula with source-only augmented support keys.

The original code selects feature channels using inter-class agreement and
text variance, then refines cache values using text affinity. This OOD
configuration fixes hyperparameters and uses the benchmark's shared prompt
rather than CuPL's dataset-specific GPT prompts.
"""

from __future__ import annotations

import numpy as np

from .primitives import Classifier, _matrix, _unit


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    ex = np.exp(shifted)
    return ex / ex.sum(axis=1, keepdims=True)


class APETrainingFree(Classifier):
    """Reference APE feature selection, prior refinement, and cache logits."""

    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        views = np.asarray(config["support_views"], dtype=np.float64)
        if views.ndim != 3 or views.shape[0] != len(x) or views.shape[2] != self.dim:
            raise ValueError("support_views must have shape [N_support, N_views, D]")
        if views.shape[1] < 2 or not np.isfinite(views).all():
            raise ValueError("APE requires at least two finite augmented support views")
        counts = np.bincount(labels, minlength=self.num_classes)
        if not np.all(counts == counts[0]):
            raise ValueError("APE requires the same shot count for every class")
        self.text = _unit(_matrix(config["text_features"]).copy())
        if self.text.shape != (self.num_classes, self.dim):
            raise ValueError("CLIP text feature shape mismatch")
        keys = _unit(views.mean(axis=1))
        self.alpha = float(config.get("alpha", 1.0))
        self.beta = float(config.get("beta", 1.0))
        self.gamma = float(config.get("gamma", 0.1))
        if self.alpha < 0 or self.beta <= 0 or self.gamma < 0:
            raise ValueError("Invalid APE coefficients")
        self.feature_count = int(config.get("ape_feature_count", 400))
        if not 1 <= self.feature_count <= self.dim:
            raise ValueError("APE feature count exceeds encoder dimension")
        weights = np.asarray(config.get("ape_criterion_weights", [0.5, 0.5]), dtype=np.float64)
        if weights.shape != (2,):
            raise ValueError("APE criterion needs two weights")
        class_sums = self.text.copy()
        np.add.at(class_sums, labels, keys)
        all_sum = class_sums.sum(axis=0)
        per_class_count = int(counts[0]) + 1
        pair_count = self.num_classes * (self.num_classes - 1) * per_class_count**2
        interclass = (all_sum**2 - np.sum(class_sums**2, axis=0)) / pair_count
        criterion = -weights[0] * interclass + weights[1] * np.var(self.text, axis=0, ddof=1)
        self.selected = np.lexsort((np.arange(self.dim), -criterion))[:self.feature_count]
        self.selected_keys = _unit(keys[:, self.selected])
        selected_text = _unit(self.text[:, self.selected])
        affinity = self.selected_keys @ selected_text.T
        prior = _softmax(affinity)
        true_prior = prior[np.arange(len(labels)), labels]
        divergence = np.log2((1.0 + 1e-6) / (true_prior + 1e-6))
        self.cache_values = np.eye(self.num_classes)[labels] * np.exp(self.gamma * divergence[:, None])
        return self

    def predict_scores(self, query_features):
        q = _unit(self._query(query_features))
        text_logits = 100.0 * q @ self.text.T
        q_selected = _unit(q[:, self.selected])
        affinity = q_selected @ self.selected_keys.T
        cache_logits = np.exp(-self.beta * (1.0 - affinity)) @ self.cache_values
        return text_logits + self.alpha * cache_logits
