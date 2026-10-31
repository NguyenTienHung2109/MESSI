"""Source-view Gaussian discriminant estimator from mrflogs/ICLR24.

The reference takes multiple augmented frozen CLIP views per support image.
Each original image remains one shot. Fixed alpha is required in the strict
OOD track because the author's labeled-validation grid search is extra data.
"""

from __future__ import annotations

import numpy as np

from .primitives import Classifier, _matrix, _unit


class GDA(Classifier):
    """Exact KS covariance regularization and uniform-prior GDA scores."""

    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        views = np.asarray(config["support_views"], dtype=np.float64)
        if views.ndim != 3 or views.shape[0] != len(x) or views.shape[2] != self.dim:
            raise ValueError("support_views must have shape [N_support, N_views, D]")
        if views.shape[1] < 2 or not np.isfinite(views).all():
            raise ValueError("GDA requires at least two finite augmented views per support image")
        z = _unit(views.reshape(-1, self.dim))
        repeated_labels = np.repeat(labels, views.shape[1])
        self.mus = np.stack([z[repeated_labels == c].mean(axis=0)
                             for c in range(self.num_classes)])
        centered = z - self.mus[repeated_labels]
        covariance = np.cov(centered, rowvar=False, bias=False)
        self.covariance = np.atleast_2d(covariance)
        regularized = ((len(centered) - 1) * self.covariance +
                       np.trace(self.covariance) * np.eye(self.dim))
        if not np.isfinite(regularized).all() or np.trace(self.covariance) <= 0:
            raise ValueError("Degenerate GDA covariance; no undocumented identity fallback")
        self.regularized_covariance = regularized
        self.covariance_inverse = self.dim * np.linalg.pinv(regularized)
        self.weight = self.mus @ self.covariance_inverse
        self.bias = -np.log(self.num_classes) - 0.5 * np.sum(self.weight * self.mus, axis=1)
        return self

    def predict_scores(self, query_features):
        return _unit(self._query(query_features)) @ self.weight.T + self.bias


class GDAClip(GDA):
    """Author's CLIP zero-shot plus GDA logits with fixed, disclosed alpha."""

    def fit(self, support_features, support_labels, config):
        super().fit(support_features, support_labels, config)
        self.text = _unit(_matrix(config["text_features"]).copy())
        if self.text.shape != (self.num_classes, self.dim):
            raise ValueError("Text feature shape mismatch")
        self.alpha = float(config["alpha"])
        self.logit_scale = float(config.get("logit_scale", 100.0))
        return self

    def predict_scores(self, query_features):
        q = _unit(self._query(query_features))
        return self.logit_scale * q @ self.text.T + self.alpha * (q @ self.weight.T + self.bias)
