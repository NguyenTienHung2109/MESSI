"""Inductive classifiers using only frozen source support features.

NumPy arrays are deliberately copied during fit. No method retains query
features, labels, statistics, or predictions between calls.
"""

from __future__ import annotations

import numpy as np


def _matrix(features) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite [N, D] feature matrix")
    return x


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class Classifier:
    def reset(self) -> None:
        self.__dict__.clear()

    def predict(self, query_features) -> np.ndarray:
        return self.predict_scores(query_features).argmax(axis=1)

    def _fit_labels(self, x, labels, config):
        features = _matrix(x).copy()
        y = np.asarray(labels, dtype=np.int64).copy()
        classes = int(config["num_classes"])
        if len(y) != len(features) or classes < 2 or not np.array_equal(
                np.unique(y), np.arange(classes)):
            raise ValueError("Every canonical class must have labeled source support")
        self.num_classes, self.dim = classes, features.shape[1]
        return features, y

    def _query(self, query_features):
        q = _matrix(query_features)
        if q.shape[1] != self.dim:
            raise ValueError("Feature dimension mismatch")
        return q


class NCM(Classifier):
    """Euclidean or cosine nearest class mean; optional prototype L2 norm."""

    def fit(self, support_features, support_labels, config):
        x, y = self._fit_labels(support_features, support_labels, config)
        self.metric = config.get("metric", "cosine")
        self.normalize_prototypes = bool(config.get("normalize_prototypes", self.metric == "cosine"))
        if self.metric not in {"cosine", "euclidean"}:
            raise ValueError(self.metric)
        if self.metric == "cosine":
            x = _unit(x)
        prototypes = np.stack([x[y == c].mean(axis=0) for c in range(self.num_classes)])
        self.prototypes = _unit(prototypes) if self.normalize_prototypes else prototypes
        return self

    def predict_scores(self, query_features):
        q = self._query(query_features)
        if self.metric == "cosine":
            return _unit(q) @ _unit(self.prototypes).T
        return -np.sum((q[:, None, :] - self.prototypes[None, :, :]) ** 2, axis=2)


class KNN(Classifier):
    """k-NN with deterministic support-index and class-index tie breaking."""

    def fit(self, support_features, support_labels, config):
        self.support, self.labels = self._fit_labels(support_features, support_labels, config)
        self.metric = config.get("metric", "cosine")
        self.k_requested = int(config.get("k", 1))
        self.weighted = bool(config.get("weighted", False))
        if self.metric not in {"cosine", "euclidean"} or self.k_requested < 1:
            raise ValueError("Invalid k-NN configuration")
        self.k_effective = min(self.k_requested, len(self.support))
        if self.metric == "cosine":
            self.support = _unit(self.support)
        return self

    def predict_scores(self, query_features):
        q = self._query(query_features)
        if self.metric == "cosine":
            distance = 1 - _unit(q) @ self.support.T
        else:
            distance = np.maximum(0, np.sum(q * q, axis=1, keepdims=True) +
                                  np.sum(self.support * self.support, axis=1)[None, :] -
                                  2 * q @ self.support.T)
        if self.k_effective == 1:
            return np.stack([-distance[:, self.labels == c].min(axis=1)
                             for c in range(self.num_classes)], axis=1)
        scores = np.zeros((len(q), self.num_classes), dtype=np.float64)
        indices = np.arange(len(self.support))
        for row, values in enumerate(distance):
            nearest = np.lexsort((indices, values))[:self.k_effective]
            weights = 1.0 / np.maximum(values[nearest], 1e-8) if self.weighted else np.ones(len(nearest))
            np.add.at(scores[row], self.labels[nearest], weights)
        return scores


class Ridge(Classifier):
    """Frozen-feature regularized least squares, not R2-D2 meta-training."""

    def fit(self, support_features, support_labels, config):
        x, y = self._fit_labels(support_features, support_labels, config)
        self.alpha = float(config.get("alpha", 1.0))
        if self.alpha <= 0:
            raise ValueError("Ridge alpha must be positive")
        self.mean = x.mean(axis=0)
        z = x - self.mean
        target = np.eye(self.num_classes, dtype=np.float64)[y]
        target_mean = target.mean(axis=0)
        centered_target = target - target_mean
        if len(z) < z.shape[1]:
            dual = np.linalg.solve(z @ z.T + self.alpha * np.eye(len(z)), centered_target)
            self.weight = z.T @ dual
        else:
            self.weight = np.linalg.solve(z.T @ z + self.alpha * np.eye(z.shape[1]),
                                          z.T @ centered_target)
        self.bias = target_mean - self.mean @ self.weight
        return self

    def predict_scores(self, query_features):
        return self._query(query_features) @ self.weight + self.bias


class SimpleShotSupport(Classifier):
    """CL2N-style strict variant: center estimated from K-shot support only."""

    def fit(self, support_features, support_labels, config):
        x, y = self._fit_labels(support_features, support_labels, config)
        self.center = x.mean(axis=0)
        z = _unit(x - self.center)
        self.prototypes = np.stack([z[y == c].mean(axis=0) for c in range(self.num_classes)])
        return self

    def predict_scores(self, query_features):
        q = _unit(self._query(query_features) - self.center)
        return -np.sum((q[:, None, :] - self.prototypes[None, :, :]) ** 2, axis=2)


class VisualCache(Classifier):
    """Visual-only Tip-style retrieval; no CLIP text logits are included."""

    def fit(self, support_features, support_labels, config):
        x, y = self._fit_labels(support_features, support_labels, config)
        self.keys = _unit(x)
        self.values = np.eye(self.num_classes)[y]
        self.beta = float(config.get("beta", 5.5))
        if self.beta <= 0:
            raise ValueError("Cache beta must be positive")
        return self

    def predict_scores(self, query_features):
        affinity = _unit(self._query(query_features)) @ self.keys.T
        return np.exp(-self.beta * (1 - affinity)) @ self.values


class CLIPZeroShot(Classifier):
    """Official cosine image/text scoring with a fixed prompt ensemble."""

    def fit(self, support_features, support_labels, config):
        self.text = _unit(_matrix(config["text_features"]).copy())
        self.num_classes, self.dim = self.text.shape
        self.logit_scale = float(config.get("logit_scale", 100.0))
        return self

    def predict_scores(self, query_features):
        return self.logit_scale * (_unit(self._query(query_features)) @ self.text.T)


class TipAdapter(VisualCache):
    """Strict single-view Tip-style combination with fixed CLIP text logits.

    The original Tip-Adapter averages multiple augmented views of each support
    image. This class uses the one cached deterministic view and must be named
    ``tip_adapter_single_view`` in result tables, never original Tip-Adapter.
    """

    def fit(self, support_features, support_labels, config):
        super().fit(support_features, support_labels, config)
        self.text = _unit(_matrix(config["text_features"]).copy())
        if self.text.shape != (self.num_classes, self.dim):
            raise ValueError("Text feature shape mismatch")
        self.alpha = float(config.get("alpha", 1.0))
        self.logit_scale = float(config.get("logit_scale", 100.0))
        return self

    def predict_scores(self, query_features):
        q = self._query(query_features)
        return self.logit_scale * (_unit(q) @ self.text.T) + self.alpha * super().predict_scores(q)


class TipAdapterFull(TipAdapter):
    """Author-formula cache using mean raw features from augmented support views.

    The view cache uses deterministic source-only augmentation. Alpha/beta
    remain fixed for strict OOD evaluation; the author's validation search is
    reserved for the separate paper-reproduction track.
    """

    def fit(self, support_features, support_labels, config):
        x = _matrix(support_features)
        views = np.asarray(config["support_views"], dtype=np.float64)
        if views.ndim != 3 or views.shape[0] != len(x) or views.shape[2] != x.shape[1]:
            raise ValueError("support_views must have shape [N_support, N_views, D]")
        if views.shape[1] < 2 or not np.isfinite(views).all():
            raise ValueError("Tip-Adapter requires at least two finite augmented views")
        return super().fit(views.mean(axis=1), support_labels, config)
