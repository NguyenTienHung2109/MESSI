"""TaskRes text residual adaptation with a frozen CLIP encoder.

The weight parameterization and cross-entropy objective follow the authors'
``trainers/taskres.py``. Image augmentations are supplied by the shared
source-only view cache; the frozen encoder is never updated.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from .primitives import Classifier, _matrix


class TaskRes(Classifier):
    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        views = np.asarray(config["support_views"], dtype=np.float32)
        text = _matrix(config["raw_text_features"])
        if views.ndim != 3 or views.shape[0] != len(x) or views.shape[2] != self.dim:
            raise ValueError("TaskRes requires aligned source views")
        if views.shape[1] < 2 or not np.isfinite(views).all():
            raise ValueError("TaskRes requires at least two finite views")
        if text.shape != (self.num_classes, self.dim):
            raise ValueError("TaskRes raw text shape mismatch")
        self.device = torch.device(config.get("training_device", "cpu"))
        self.scale = float(config.get("taskres_scale", 0.5))
        self.logit_scale = float(config.get("logit_scale", 100.0))
        epochs = int(config.get("taskres_epochs", 100))
        batch_size = int(config.get("training_batch_size", 256))
        lr = float(config.get("taskres_lr", 0.002))
        warmup_lr = float(config.get("taskres_warmup_lr", 1e-5))
        if self.scale <= 0 or epochs < 1 or batch_size < 1 or lr <= 0 or warmup_lr <= 0:
            raise ValueError("Invalid TaskRes training configuration")
        generator = torch.Generator(device="cpu").manual_seed(int(config.get("training_seed", 0)))
        base = torch.as_tensor(text, dtype=torch.float32, device=self.device)
        residual = torch.nn.Parameter(torch.zeros_like(base))
        optimizer = torch.optim.Adam([residual], lr=lr)
        # Released 1-shot configs use a one-epoch constant warmup at 1e-5,
        # then cosine decay over the remaining epochs.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: warmup_lr / lr if epoch < 1 else
            0.5 * (1 + math.cos(math.pi * (epoch - 1) / max(1, epochs - 1))))
        y = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        source_views = torch.as_tensor(views, dtype=torch.float32)
        for epoch in range(epochs):
            order = torch.randperm(len(labels), generator=generator)
            # A fresh augmentation is drawn in the author training loop each
            # epoch. Cycling ten precomputed source views keeps the same shot
            # IDs while avoiding any use of target-domain images.
            epoch_views = source_views[:, epoch % views.shape[1]]
            for indices in order.split(batch_size):
                inputs = F.normalize(epoch_views[indices].to(self.device), dim=1)
                prototypes = F.normalize(base + self.scale * residual, dim=1)
                logits = self.logit_scale * inputs @ prototypes.T
                loss = F.cross_entropy(logits, y[indices.to(self.device)])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        with torch.inference_mode():
            self.prototypes = F.normalize(base + self.scale * residual, dim=1).detach()
        return self

    def predict_scores(self, query_features):
        inputs = torch.as_tensor(self._query(query_features), dtype=torch.float32,
                                 device=self.device)
        with torch.inference_mode():
            scores = self.logit_scale * F.normalize(inputs, dim=1) @ self.prototypes.T
        return scores.cpu().numpy()
