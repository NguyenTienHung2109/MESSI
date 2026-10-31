"""Validation-free CrossModal LP rerun from the official CLAP repository.

CLAP reimplements Lin et al.'s image/text linear probe with zero-shot
initialization, unit-normalized classifier weights, SGD and 300 epochs. This
is distinct from Lin et al.'s original AdamW/validation-sweep protocol.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .primitives import Classifier, _matrix


class CrossModalCLAP(Classifier):
    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        text = _matrix(config["raw_text_features"])
        prompts = np.asarray(config["prompt_features"], dtype=np.float32)
        views = np.asarray(config["support_views"], dtype=np.float32)
        if text.shape != (self.num_classes, self.dim):
            raise ValueError("CrossModal CLAP text shape mismatch")
        if prompts.ndim == 2:
            prompts = prompts[:, None, :]
        if prompts.ndim != 3 or prompts.shape[0] != self.num_classes or prompts.shape[2] != self.dim:
            raise ValueError("CrossModal CLAP prompt bank shape mismatch")
        if views.ndim != 3 or views.shape[0] != len(x) or views.shape[2] != self.dim:
            raise ValueError("CrossModal CLAP support views shape mismatch")
        self.device = torch.device(config.get("training_device", "cpu"))
        image = F.normalize(torch.as_tensor(views.reshape(-1, self.dim),
                                                  dtype=torch.float32, device=self.device), dim=1)
        image_labels = torch.as_tensor(np.repeat(labels, views.shape[1]),
                                       device=self.device, dtype=torch.long)
        prompt_bank = F.normalize(torch.as_tensor(prompts.reshape(-1, self.dim),
                                                      device=self.device), dim=1)
        prompt_labels = torch.arange(self.num_classes, device=self.device).repeat_interleave(prompts.shape[1])
        seed = int(config.get("training_seed", 0))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        sampled_text = torch.randint(len(prompt_bank), (len(image),), generator=generator).to(self.device)
        training = torch.cat([image, prompt_bank[sampled_text]], dim=0)
        training_labels = torch.cat([image_labels, prompt_labels[sampled_text]], dim=0)
        # CLAP initializes from the *raw* CLIP text vectors. Their norms
        # strongly affect SGD updates even though scoring normalizes them.
        self.weights = nn.Parameter(torch.as_tensor(text, dtype=torch.float32,
                                                     device=self.device).clone())
        epochs = int(config.get("cross_modal_clap_epochs", 300))
        lr = float(config.get("cross_modal_clap_lr", 0.1))
        batch_size = int(config.get("training_batch_size", 256))
        if epochs < 1 or lr <= 0 or batch_size < 1:
            raise ValueError("Invalid CrossModal CLAP optimization configuration")
        optimizer = torch.optim.SGD([self.weights], lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        # The author loops over len(train_loader_x), i.e. the number of
        # *original support images* per batch, even after adding many cached
        # augmented image/text features to the sampled training pool.
        steps_per_epoch = (len(labels) + batch_size - 1) // batch_size
        for _ in range(epochs):
            sampled = torch.randint(len(training), (len(training) // 2,),
                                    generator=generator)
            order = sampled[torch.randperm(len(sampled), generator=generator)]
            for batch_index, indices in enumerate(order.split(batch_size)):
                if batch_index >= steps_per_epoch:
                    break
                batch = training[indices.to(self.device)]
                targets = training_labels[indices.to(self.device)]
                scores = 100.0 * batch @ F.normalize(self.weights, dim=1).T
                loss = F.cross_entropy(scores, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict_scores(self, query_features):
        query = torch.as_tensor(self._query(query_features), device=self.device,
                                dtype=torch.float32)
        with torch.inference_mode():
            scores = 100.0 * F.normalize(query, dim=1) @ F.normalize(self.weights, dim=1).T
        return scores.cpu().numpy()
