"""APE-T with frozen CLIP features and source-only optimization.

The feature criterion, residual parameters, weighted cache values, logits,
optimizer and schedule follow the authors' APE implementation. The OOD run
uses fixed ImageNet coefficients and the final training epoch so that target
labels never choose hyperparameters or checkpoints.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .primitives import Classifier, _matrix


def ape_t_selected_channels(text: torch.Tensor, count: int,
                            weights: tuple[float, float] = (0.2, 0.8)) -> torch.Tensor:
    """Vectorized equivalent of author cal_criterion(only_use_txt=True)."""
    classes, dim = text.shape
    if classes < 2 or count < 1 or count > dim:
        raise ValueError("APE-T needs at least two classes and valid channel count")
    class_sum = text.sum(dim=0)
    interclass = (class_sum.square() - text.square().sum(dim=0)) / (classes * (classes - 1))
    criterion = -weights[0] * interclass + weights[1] * text.var(dim=0, unbiased=True)
    return torch.topk(criterion, k=count).indices


class APETraining(nn.Module):
    """Author APE_Training forward, with class-major keys."""

    def __init__(self, text: torch.Tensor, keys: torch.Tensor, shots: int,
                 selected: torch.Tensor):
        super().__init__()
        classes, dim = text.shape
        if keys.shape != (classes * shots, dim):
            raise ValueError("APE-T keys must be ordered class then shot")
        self.shots = shots
        self.dim = dim
        self.classes = classes
        self.register_buffer("selected", selected)
        self.residual = nn.Parameter(torch.zeros(classes, len(selected), device=text.device,
                                                 dtype=text.dtype))
        self.value_weights = nn.Parameter(torch.ones(classes * shots, 1,
                                                     device=text.device, dtype=text.dtype))

    def forward(self, keys: torch.Tensor, text: torch.Tensor,
                values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_keys = self.residual.repeat_interleave(self.shots, dim=0)
        new_keys = keys.clone()
        new_keys[:, self.selected] = new_keys[:, self.selected] + residual_keys
        new_text = text.clone()
        new_text[:, self.selected] = new_text[:, self.selected] + self.residual
        return new_keys, new_text, values * self.value_weights


class APET(Classifier):
    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        views = np.asarray(config["support_views"], dtype=np.float32)
        text = _matrix(config["text_features"])
        if views.ndim != 3 or views.shape[0] != len(x) or views.shape[2] != self.dim:
            raise ValueError("APE-T requires aligned source views")
        if views.shape[1] < 2 or not np.isfinite(views).all():
            raise ValueError("APE-T requires at least two finite views")
        if text.shape != (self.num_classes, self.dim):
            raise ValueError("APE-T text shape mismatch")
        counts = np.bincount(labels, minlength=self.num_classes)
        if not np.all(counts == counts[0]):
            raise ValueError("APE-T requires equal shots per class")
        shots = int(counts[0])
        order = np.argsort(labels, kind="stable")
        labels = labels[order]
        views = views[order]
        self.device = torch.device(config.get("training_device", "cpu"))
        self.text = F.normalize(torch.as_tensor(text, device=self.device,
                                               dtype=torch.float32), dim=1)
        keys = F.normalize(torch.as_tensor(views.mean(axis=1), device=self.device), dim=1)
        self.values = F.one_hot(torch.as_tensor(labels, device=self.device),
                                num_classes=self.num_classes).float()
        count = min(int(config.get("ape_t_feature_count", 500)), self.dim)
        selected = ape_t_selected_channels(self.text, count)
        self.model = APETraining(self.text, keys, shots, selected).to(self.device)
        self.keys = keys
        self.alpha = float(config.get("ape_t_alpha", 2.0))
        self.beta = float(config.get("ape_t_beta", 1.0))
        epochs = int(config.get("ape_t_epochs", 20))
        lr = float(config.get("ape_t_lr", 0.0001))
        eps = float(config.get("ape_t_eps", 0.001))
        batch_size = int(config.get("training_batch_size", 256))
        if self.alpha < 0 or self.beta <= 0 or epochs < 1 or lr <= 0 or eps <= 0 or batch_size < 1:
            raise ValueError("Invalid APE-T configuration")
        generator = torch.Generator(device="cpu").manual_seed(
            int(config.get("training_seed", 0)))
        source_views = torch.as_tensor(views)
        y = torch.as_tensor(labels, device=self.device, dtype=torch.long)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, eps=eps, weight_decay=0.1)
        steps_per_epoch = (len(labels) + batch_size - 1) // batch_size
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * steps_per_epoch)
        for epoch in range(epochs):
            epoch_views = source_views[:, epoch % views.shape[1]]
            for indices in torch.randperm(len(labels), generator=generator).split(batch_size):
                inputs = F.normalize(epoch_views[indices].to(self.device), dim=1)
                new_keys, new_text, new_values = self.model(
                    self.keys, self.text, self.values)
                cache = torch.exp(-self.beta * (1.0 - inputs @ new_keys.T)) @ new_values
                scores = 100.0 * inputs @ new_text.T + self.alpha * cache
                loss = F.cross_entropy(scores, y[indices.to(self.device)])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        self.model.eval()
        return self

    def predict_scores(self, query_features):
        inputs = torch.as_tensor(self._query(query_features), device=self.device,
                                 dtype=torch.float32)
        with torch.inference_mode():
            inputs = F.normalize(inputs, dim=1)
            keys, text, values = self.model(self.keys, self.text, self.values)
            cache = torch.exp(-self.beta * (1.0 - inputs @ keys.T)) @ values
            scores = 100.0 * inputs @ text.T + self.alpha * cache
        return scores.cpu().numpy()
