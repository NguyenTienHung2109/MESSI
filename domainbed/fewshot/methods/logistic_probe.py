"""OpenAI CLIP README's scikit-learn frozen-feature linear probe.

The README uses LogisticRegression(C=0.316, max_iter=1000) on raw image
embeddings. With no target validation in strict OOD evaluation, C is fixed.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .primitives import Classifier


class LogisticProbe(Classifier):
    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        c = float(config.get("logistic_c", 0.316))
        if c <= 0:
            raise ValueError("Logistic probe C must be positive")
        self.model = LogisticRegression(random_state=0, C=c, max_iter=1000)
        self.model.fit(x, labels)
        if not np.array_equal(self.model.classes_, np.arange(self.num_classes)):
            raise ValueError("Logistic probe class mapping changed")
        return self

    def predict_scores(self, query_features):
        scores = np.asarray(self.model.decision_function(self._query(query_features)))
        return np.stack([-scores, scores], axis=1) if scores.ndim == 1 else scores
