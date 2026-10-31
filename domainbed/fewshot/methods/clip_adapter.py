"""Visual CLIP-Adapter with frozen CLIP features.

The two-layer ReLU adapter, quarter-width bottleneck and 0.2 residual
fraction follow the authors' ``clip_adapter.py``. Only adapter weights are
trained; text prototypes and both CLIP encoders are fixed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .primitives import Classifier, _matrix


class _AuthorVisualAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 4, bias=False), nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim, bias=False), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fc(x)


class CLIPAdapter(Classifier):
    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        views = np.asarray(config["support_views"], dtype=np.float32)
        text = _matrix(config["text_features"])
        if views.ndim != 3 or views.shape[0] != len(x) or views.shape[2] != self.dim:
            raise ValueError("CLIP-Adapter requires aligned source views")
        if views.shape[1] < 2 or not np.isfinite(views).all():
            raise ValueError("CLIP-Adapter requires at least two finite views")
        if text.shape != (self.num_classes, self.dim):
            raise ValueError("CLIP-Adapter text feature shape mismatch")
        self.device = torch.device(config.get("training_device", "cpu"))
        self.text = F.normalize(torch.as_tensor(text, dtype=torch.float32,
                                               device=self.device), dim=1)
        self.ratio = float(config.get("clip_adapter_ratio", 0.2))
        self.logit_scale = float(config.get("logit_scale", 100.0))
        epochs = int(config.get("clip_adapter_epochs", 300))
        batch_size = int(config.get("training_batch_size", 256))
        lr = float(config.get("clip_adapter_lr", 0.1))
        optimizer_name = config.get("clip_adapter_optimizer", "sgd")
        if not 0 <= self.ratio <= 1 or epochs < 1 or batch_size < 1 or lr <= 0:
            raise ValueError("Invalid CLIP-Adapter configuration")
        seed = int(config.get("training_seed", 0))
        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.adapter = _AuthorVisualAdapter(self.dim).to(self.device)
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.adapter.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr, weight_decay=0)
        else:
            raise ValueError("Unsupported CLIP-Adapter optimizer")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        source_views = torch.as_tensor(views)
        y = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        for epoch in range(epochs):
            epoch_views = source_views[:, epoch % views.shape[1]]
            for indices in torch.randperm(len(labels), generator=generator).split(batch_size):
                inputs = epoch_views[indices].to(self.device)
                blended = self.ratio * self.adapter(inputs) + (1 - self.ratio) * inputs
                logits = self.logit_scale * F.normalize(blended, dim=1) @ self.text.T
                loss = F.cross_entropy(logits, y[indices.to(self.device)])
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
            blended = self.ratio * self.adapter(inputs) + (1 - self.ratio) * inputs
            scores = self.logit_scale * F.normalize(blended, dim=1) @ self.text.T
        return scores.cpu().numpy()
