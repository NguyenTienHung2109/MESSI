"""Tip-Adapter-F on frozen source features.

This follows the author's learnable cache-key linear layer, AdamW optimizer,
and cosine schedule. The strict OOD protocol fixes the ImageNet initial
coefficients and uses the final epoch because target labels cannot select
the best epoch or tune alpha and beta.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .primitives import Classifier, _matrix


class TipAdapterF(Classifier):
    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        views = np.asarray(config["support_views"], dtype=np.float32)
        text = _matrix(config["text_features"])
        if views.ndim != 3 or views.shape[0] != len(x) or views.shape[2] != self.dim:
            raise ValueError("Tip-Adapter-F requires aligned source views")
        if views.shape[1] < 2 or not np.isfinite(views).all():
            raise ValueError("Tip-Adapter-F requires at least two finite views")
        if text.shape != (self.num_classes, self.dim):
            raise ValueError("Tip-Adapter-F text shape mismatch")
        self.device = torch.device(config.get("training_device", "cpu"))
        self.text = F.normalize(torch.as_tensor(text, dtype=torch.float32,
                                               device=self.device), dim=1)
        self.beta = float(config.get("tip_f_beta", 1.0))
        self.alpha = float(config.get("tip_f_alpha", 1.17))
        epochs = int(config.get("tip_f_epochs", 20))
        lr = float(config.get("tip_f_lr", 0.001))
        batch_size = int(config.get("training_batch_size", 256))
        if self.beta <= 0 or self.alpha < 0 or epochs < 1 or lr <= 0 or batch_size < 1:
            raise ValueError("Invalid Tip-Adapter-F configuration")
        seed = int(config.get("training_seed", 0))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        cache_keys = F.normalize(torch.as_tensor(views.mean(axis=1), dtype=torch.float32,
                                                device=self.device), dim=1)
        self.values = F.one_hot(torch.as_tensor(labels, device=self.device),
                                num_classes=self.num_classes).float()
        self.adapter = nn.Linear(self.dim, len(labels), bias=False).to(self.device)
        with torch.no_grad():
            self.adapter.weight.copy_(cache_keys)
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr, eps=1e-4)
        steps_per_epoch = (len(labels) + batch_size - 1) // batch_size
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * steps_per_epoch)
        source_views = torch.as_tensor(views)
        y = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        for epoch in range(epochs):
            epoch_views = source_views[:, epoch % views.shape[1]]
            for indices in torch.randperm(len(labels), generator=generator).split(batch_size):
                inputs = F.normalize(epoch_views[indices].to(self.device), dim=1)
                affinity = self.adapter(inputs)
                cache_logits = torch.exp(-self.beta * (1 - affinity)) @ self.values
                clip_logits = 100.0 * inputs @ self.text.T
                loss = F.cross_entropy(clip_logits + self.alpha * cache_logits,
                                       y[indices.to(self.device)])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        self.adapter.eval()
        return self

    def predict_scores(self, query_features):
        inputs = torch.as_tensor(self._query(query_features), dtype=torch.float32,
                                 device=self.device)
        with torch.inference_mode():
            query = F.normalize(inputs, dim=1)
            cache_logits = torch.exp(-self.beta * (1 - self.adapter(query))) @ self.values
            scores = 100.0 * query @ self.text.T + self.alpha * cache_logits
        return scores.cpu().numpy()
