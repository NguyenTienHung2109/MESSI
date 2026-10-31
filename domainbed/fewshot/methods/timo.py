"""Source-only TIMO adaptation with fixed prompt and fusion policies.

The author's TIMO-S searches several choices using labeled validation data;
this strict variant implements the training-free TIMO path with fixed choices.
"""

from __future__ import annotations

import numpy as np

from .primitives import Classifier, _matrix, _unit


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(shifted)
    return ex / ex.sum(axis=axis, keepdims=True)


class TIMOFixed(Classifier):
    """Image-guided text selection and instance-augmented GDA (author formula)."""

    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        views = np.asarray(config["support_views"], dtype=np.float64)
        prompts = np.asarray(config["prompt_features"], dtype=np.float64)
        if views.ndim != 3 or views.shape[0] != len(x) or views.shape[2] != self.dim:
            raise ValueError("TIMO support views have invalid shape")
        if views.shape[1] < 2 or not np.isfinite(views).all():
            raise ValueError("TIMO needs at least two finite support views")
        if (prompts.ndim != 3 or prompts.shape[0] != self.num_classes or
                prompts.shape[2] != self.dim or prompts.shape[1] < 2 or
                not np.isfinite(prompts).all()):
            raise ValueError("TIMO requires at least two finite text prompts per class")
        prompts = _unit(prompts.reshape(-1, self.dim)).reshape(prompts.shape)
        self.prompt_count = prompts.shape[1]
        self.prompt_gamma = float(config.get("timo_prompt_gamma", 50.0))
        self.alpha = float(config.get("timo_alpha", 100.0))
        if self.prompt_gamma <= 0 or self.alpha < 0:
            raise ValueError("Invalid TIMO fixed coefficients")

        # Author cache keys: mean raw augmented features, then L2 normalize.
        cache_keys = _unit(views.mean(axis=1))
        image_weights = np.stack([cache_keys[labels == c].mean(axis=0)
                                  for c in range(self.num_classes)])
        image_weights = _unit(image_weights)
        matching = np.einsum("cd,cpd->cp", image_weights, prompts)
        normed = matching / np.maximum(np.linalg.norm(matching, axis=1, keepdims=True), 1e-12)
        text_weights = _softmax(normed * self.prompt_gamma, axis=1)
        self.image_guided_text = _unit(np.einsum("cp,cpd->cd", text_weights, prompts))

        # TIMO's instance-based transfer weights each prompt vector by its
        # unnormalized visual/text matching score before KS covariance fit.
        order = np.argsort(-matching, axis=1, kind="stable")
        sorted_prompts = np.take_along_axis(prompts, order[:, :, None], axis=1)
        sorted_matching = np.take_along_axis(matching, order, axis=1)
        text_instances = (sorted_prompts * sorted_matching[:, :, None]).reshape(-1, self.dim)
        text_labels = np.repeat(np.arange(self.num_classes), self.prompt_count)
        visual_instances = _unit(views.reshape(-1, self.dim))
        visual_labels = np.repeat(labels, views.shape[1])
        instances = np.concatenate([text_instances, visual_instances], axis=0)
        instance_labels = np.concatenate([text_labels, visual_labels])
        mus = np.stack([instances[instance_labels == c].mean(axis=0)
                        for c in range(self.num_classes)])
        centered = instances - mus[instance_labels]
        covariance = np.cov(centered, rowvar=False, bias=False)
        covariance = np.atleast_2d(covariance)
        if np.trace(covariance) <= 0 or not np.isfinite(covariance).all():
            raise ValueError("Degenerate TIMO covariance")
        regularized = ((len(instances) - 1) * covariance +
                       np.trace(covariance) * np.eye(self.dim))
        covariance_inverse = self.dim * np.linalg.pinv(regularized)
        self.weight = mus @ covariance_inverse
        self.bias = -np.log(self.num_classes) - 0.5 * np.sum(self.weight * mus, axis=1)
        return self

    def predict_scores(self, query_features):
        q = _unit(self._query(query_features))
        return self.alpha * q @ self.image_guided_text.T + q @ self.weight.T + self.bias
