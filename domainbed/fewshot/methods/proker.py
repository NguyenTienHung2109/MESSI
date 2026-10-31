"""ProKeR's closed-form kernel adaptation on frozen CLIP features.

The equations follow ybendou/ProKeR, trainers/proker.py.  The strict OOD
configuration uses the author's published ImageNet-selected coefficients;
no target labels or unlabeled target statistics are used during fit.
"""

from __future__ import annotations

import numpy as np
import torch

from .primitives import Classifier, _matrix, _unit


# Official RN50 ImageNet-selected coefficients for each K, from
# configs/RN50/configs_proker/hyperparameters.json at the pinned reference.
IMAGENET_HYPERPARAMETERS = {
    1: (3.0, 0.05),
    2: (2.6, 0.05),
    4: (1.5, 0.05),
    8: (1.66, 0.07),
    16: (1.7, 0.1),
}


class ProKeR(Classifier):
    """CLIP prior plus RBF kernel ridge correction fitted on support views."""

    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        text = _unit(_matrix(config["text_features"]).copy())
        if text.shape != (self.num_classes, self.dim):
            raise ValueError("ProKeR text features have an unexpected shape")
        views = np.asarray(config["support_views"], dtype=np.float64)
        if views.ndim != 3 or views.shape[0] != len(x) or views.shape[2] != self.dim:
            raise ValueError("ProKeR requires aligned [support, views, dim] features")
        if views.shape[1] < 2 or not np.isfinite(views).all():
            raise ValueError("ProKeR requires at least two finite augmented views")
        self.keys = _unit(views.reshape(-1, self.dim))
        view_labels = np.repeat(labels, views.shape[1])
        self.text = text
        self.beta = float(config["proker_beta"])
        self.lmbda = float(config["proker_lambda"])
        if self.beta <= 0 or self.lmbda <= 0:
            raise ValueError("ProKeR coefficients must be positive")
        # The solve and intermediate arrays can require several dense matrices.
        estimated_bytes = len(self.keys) ** 2 * 8 * 3
        if estimated_bytes > 48 * 1024**3:
            raise MemoryError("Exact ProKeR estimated working set exceeds 48 GiB; use the RFF solver")
        kernel = np.exp(-self.beta * (1.0 - self.keys @ self.keys.T))
        kernel.flat[::len(kernel) + 1] += self.lmbda
        target = np.eye(self.num_classes, dtype=np.float64)[view_labels]
        residual = target - self.keys @ self.text.T
        # The author's solve is (K/lambda + I)^-1 residual, equivalently
        # lambda * (K + lambda I)^-1 residual.
        self.coefficients = self.lmbda * np.linalg.solve(kernel, residual)
        return self

    def predict_scores(self, query_features):
        query = _unit(self._query(query_features))
        affinity = np.exp(-self.beta * (1.0 - query @ self.keys.T))
        return query @ self.text.T + affinity @ self.coefficients


class ProKeROrthogonalRFF(Classifier):
    """Paper section 4.5's orthogonal Fourier feature approximation.

    This is a distinct method key: finite Fourier features approximate the
    RBF kernel and must pass their own paper-accuracy gate. Source views are
    streamed into the primal normal equations to bound memory usage.
    """

    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        self.text = torch.as_tensor(_unit(_matrix(config["text_features"])),
                                    dtype=torch.float32, device=config.get("rff_device", "cpu"))
        views = np.asarray(config["support_views"], dtype=np.float32)
        if views.ndim != 3 or views.shape[0] != len(x) or views.shape[2] != self.dim:
            raise ValueError("ProKeR RFF requires aligned support views")
        if views.shape[1] < 2 or not np.isfinite(views).all():
            raise ValueError("ProKeR RFF requires at least two finite views")
        self.device = self.text.device
        self.beta = float(config["proker_beta"])
        self.lmbda = float(config["proker_lambda"])
        self.components = int(config.get("rff_components", 4096))
        if self.beta <= 0 or self.lmbda <= 0 or self.components < self.dim:
            raise ValueError("Invalid ProKeR RFF parameters")
        generator = torch.Generator(device=self.device).manual_seed(int(config.get("rff_seed", 0)))
        blocks = []
        remaining = self.components
        while remaining:
            count = min(self.dim, remaining)
            random = torch.randn((self.dim, self.dim), generator=generator, device=self.device)
            q, _ = torch.linalg.qr(random)
            radius = torch.sqrt(torch.sum(torch.randn((count, self.dim), generator=generator,
                                                        device=self.device) ** 2, dim=1))
            blocks.append(q[:, :count] * radius[None, :] * np.sqrt(self.beta))
            remaining -= count
        self.frequencies = torch.cat(blocks, dim=1)
        self.offset = 2 * np.pi * torch.rand((self.components,), generator=generator,
                                             device=self.device)
        gram = torch.zeros((self.components, self.components), dtype=torch.float32,
                           device=self.device)
        cross = torch.zeros((self.components, self.num_classes), dtype=torch.float32,
                            device=self.device)
        flattened = views.reshape(-1, self.dim)
        all_labels = np.repeat(labels, views.shape[1])
        block_size = int(config.get("rff_fit_batch", 1024))
        for start in range(0, len(flattened), block_size):
            end = min(start + block_size, len(flattened))
            keys = torch.as_tensor(flattened[start:end], device=self.device)
            keys = torch.nn.functional.normalize(keys, dim=1)
            feature_map = self._map(keys)
            target = torch.nn.functional.one_hot(
                torch.as_tensor(all_labels[start:end], device=self.device),
                self.num_classes).float() - keys @ self.text.T
            gram.addmm_(feature_map.T, feature_map)
            cross.addmm_(feature_map.T, target)
        gram.diagonal().add_(self.lmbda)
        self.weights = self.lmbda * torch.linalg.solve(gram, cross)
        return self

    def _map(self, x):
        return np.sqrt(2.0 / self.components) * torch.cos(x @ self.frequencies + self.offset)

    def predict_scores(self, query_features):
        query = torch.as_tensor(_unit(self._query(query_features)), dtype=torch.float32,
                                device=self.device)
        with torch.inference_mode():
            scores = query @ self.text.T + self._map(query) @ self.weights
        return scores.cpu().numpy()
