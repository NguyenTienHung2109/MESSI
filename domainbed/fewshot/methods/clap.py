"""Class-adaptive constrained linear probing (CLAP) on frozen CLIP features.

The prototype initialization, class-wise zero-shot multiplier and regularized
cross-entropy follow ``jusiro/CLAP/trainers/adapters.py``. Both CLIP encoders
remain frozen. Source views are the only training examples.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .primitives import Classifier, _matrix


class CLAP(Classifier):
    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        views = np.asarray(config["support_views"], dtype=np.float32)
        base = _matrix(config["raw_text_features"])
        if views.ndim != 3 or views.shape[0] != len(x) or views.shape[2] != self.dim:
            raise ValueError("CLAP requires aligned source views")
        if views.shape[1] < 2 or not np.isfinite(views).all():
            raise ValueError("CLAP requires at least two finite views")
        if base.shape != (self.num_classes, self.dim):
            raise ValueError("CLAP raw text shape mismatch")
        self.device = torch.device(config.get("training_device", "cpu"))
        self.logit_scale = float(config.get("logit_scale", 100.0))
        epochs = int(config.get("clap_epochs", 300))
        batch_size = int(config.get("training_batch_size", 256))
        lr = float(config.get("clap_lr", 0.1))
        warmup_lr = float(config.get("clap_warmup_lr", 1e-5))
        reference_views = int(config.get("clap_reference_views", 20))
        if epochs < 1 or batch_size < 1 or lr <= 0 or warmup_lr <= 0 or reference_views < 1:
            raise ValueError("Invalid CLAP training configuration")
        seed = int(config.get("training_seed", 0))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # The author extracts 20 augmentations per support image. OOD has ten
        # cached augmentations, so cycle those views to form the same sized
        # source pool without changing the labeled support identities.
        view_indices = np.arange(reference_views) % views.shape[1]
        source = torch.as_tensor(views[:, view_indices].reshape(-1, self.dim),
                                 dtype=torch.float32)
        all_labels = torch.as_tensor(np.repeat(labels, reference_views), dtype=torch.long)
        base = torch.as_tensor(base, dtype=torch.float32, device=self.device)
        prototypes = nn.Parameter(base.clone())
        # Author's init_lagrangian_multipliers uses the zero-shot probability
        # of each ground-truth class, averaged across its source examples.
        confidence = torch.zeros(self.num_classes, device=self.device)
        counts = torch.zeros(self.num_classes, device=self.device)
        with torch.inference_mode():
            base_norm = F.normalize(base, dim=1)
            for start in range(0, len(source), batch_size):
                image = F.normalize(source[start:start + batch_size].to(self.device), dim=1)
                truth = all_labels[start:start + batch_size].to(self.device)
                probabilities = F.softmax(self.logit_scale * image @ base_norm.T, dim=1)
                confidence.index_add_(0, truth, probabilities.gather(1, truth[:, None]).flatten())
                counts.index_add_(0, truth, torch.ones_like(truth, dtype=torch.float32))
        alpha_constraint = confidence / counts.clamp_min(1)
        optimizer = torch.optim.SGD([prototypes], lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: warmup_lr / lr if epoch < 1 else
            0.5 * (1 + math.cos(math.pi * (epoch - 1) / max(1, epochs - 1))))
        # Released run_epoch takes len(train_loader_x) batches of size 256
        # from the shuffled 20-view feature pool, rather than traversing the
        # entire augmented pool each epoch.
        epoch_examples = min(len(source), math.ceil(len(labels) / batch_size) * batch_size)
        for _ in range(epochs):
            order = torch.randperm(len(source), generator=generator)[:epoch_examples]
            for indices in order.split(batch_size):
                image = F.normalize(source[indices].to(self.device), dim=1)
                truth = all_labels[indices].to(self.device)
                logits = self.logit_scale * image @ F.normalize(prototypes, dim=1).T
                loss_ce = F.cross_entropy(logits, truth)
                loss_constraint = torch.mean(
                    alpha_constraint * torch.sum((prototypes - base) ** 2, dim=1))
                optimizer.zero_grad()
                (loss_ce + loss_constraint).backward()
                optimizer.step()
            scheduler.step()
        self.prototypes = F.normalize(prototypes.detach(), dim=1)
        self.alpha_constraint = alpha_constraint.detach().cpu().numpy()
        return self

    def predict_scores(self, query_features):
        image = torch.as_tensor(self._query(query_features), dtype=torch.float32,
                                device=self.device)
        with torch.inference_mode():
            logits = self.logit_scale * F.normalize(image, dim=1) @ self.prototypes.T
        return logits.cpu().numpy()
