"""Parameter-free CALIP attention with disclosed fixed fusion coefficients."""

from __future__ import annotations

import numpy as np

from .primitives import Classifier, _matrix, _unit


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(shifted)
    return ex / ex.sum(axis=axis, keepdims=True)


class CALIPFixed(Classifier):
    """Author attention equations, using cached CLIP ViT image tokens.

    The original ImageNet script searches beta2/beta3 on labeled validation
    images. Strict OOD runs use fixed disclosed coefficients instead.
    """

    def fit(self, support_features, support_labels, config):
        text = _unit(_matrix(config["text_features"])).astype(np.float32)
        self.text = text
        self.num_classes, self.dim = text.shape
        self.beta2 = float(config.get("calip_beta2", 2.0))
        self.beta3 = float(config.get("calip_beta3", 0.1))
        if self.beta2 < 0 or self.beta3 < 0:
            raise ValueError("CALIP fusion coefficients must be nonnegative")
        return self

    def predict_scores(self, query_features):
        if not isinstance(query_features, tuple) or len(query_features) != 2:
            raise ValueError("CALIP expects (global_features, patch_features)")
        global_features, patch_features = query_features
        q = _unit(self._query(global_features)).astype(np.float32)
        patch = np.asarray(patch_features, dtype=np.float32)
        if patch.ndim != 3 or patch.shape[0] != len(q) or patch.shape[2] != self.dim:
            raise ValueError("CALIP patch features must have shape [N, P, D]")
        if not np.isfinite(patch).all():
            raise ValueError("Non-finite CALIP patch features")
        patch = patch / np.maximum(np.linalg.norm(patch, axis=2, keepdims=True), 1e-12)
        attention = 2.0 * np.einsum("bpd,cd->bpc", patch, self.text)
        spatial_weights = _softmax(attention, axis=1)
        class_weights = _softmax(attention, axis=2)
        attended_text = np.einsum("bpd,bpc->bcd", patch, spatial_weights)
        attended_visual = np.einsum("bpc,cd->bpd", class_weights, self.text)
        attended_visual = attended_visual.mean(axis=1) + attended_visual.max(axis=1)
        base = 100.0 * q @ self.text.T
        logits1 = 100.0 * np.einsum("bd,bcd->bc", q, attended_text)
        logits2 = 100.0 * attended_visual @ self.text.T
        return base + self.beta2 * logits1 + self.beta3 * logits2
