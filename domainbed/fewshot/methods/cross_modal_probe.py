"""Cross-modal linear probe on frozen CLIP image and text features.

The author trains a bias-free linear head from labeled image features and
class-name text features in equal-size image/text minibatches. The paper's
main result initializes the head from text features. This OOD variant uses a
fixed author hyperparameter combination and the final iteration; validation
search and checkpoint selection are excluded for OOD.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .primitives import Classifier


class CrossModalLinearProbe(Classifier):
    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        prompts = np.asarray(config["prompt_features"], dtype=np.float32)
        if prompts.ndim == 2:
            prompts = prompts[:, None, :]
        if prompts.ndim != 3 or prompts.shape[0] != self.num_classes or prompts.shape[2] != self.dim:
            raise ValueError("Cross-modal text features must be [C, P, D]")
        views = np.asarray(config["support_views"], dtype=np.float32)
        if views.ndim != 3 or views.shape[0] != len(x) or views.shape[2] != self.dim:
            raise ValueError("Cross-modal probe requires aligned source views")
        if not np.isfinite(prompts).all() or not np.isfinite(views).all():
            raise ValueError("Features must be finite")
        self.device = torch.device(config.get("training_device", "cpu"))
        image = np.stack([x, views[:, 0]], axis=1).reshape(-1, self.dim).astype(np.float32)
        image_labels = np.repeat(labels, 2)
        text = prompts.reshape(-1, self.dim)
        text_labels = np.repeat(np.arange(self.num_classes), prompts.shape[1])
        image = F.normalize(torch.as_tensor(image, device=self.device), dim=1)
        raw_prompts = torch.as_tensor(prompts, device=self.device)
        text = F.normalize(torch.as_tensor(text, device=self.device), dim=1)
        image_labels = torch.as_tensor(image_labels, device=self.device, dtype=torch.long)
        text_labels = torch.as_tensor(text_labels, device=self.device, dtype=torch.long)
        seed = int(config.get("training_seed", 0))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = nn.Linear(self.dim, self.num_classes, bias=False).to(self.device)
        initialization = config.get("cross_modal_init", "text")
        if initialization == "text":
            # The paper's main table initializes the classifier from CLIP
            # text features. The released head.py averages raw prompt
            # features for each class, then normalizes each resulting row.
            with torch.no_grad():
                self.head.weight.copy_(F.normalize(raw_prompts.mean(dim=1), dim=1))
        elif initialization != "random":
            raise ValueError("Unknown CrossModal classifier initialization")
        iterations = int(config.get("cross_modal_iterations", 12800))
        batch = int(config.get("training_batch_size", 32))
        lr = float(config.get("cross_modal_lr", 0.001))
        weight_decay = float(config.get("cross_modal_weight_decay", 0.0))
        if iterations < 1 or batch < 2 or batch % 2 or lr <= 0 or weight_decay < 0:
            raise ValueError("Invalid cross-modal optimization configuration")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        image_batch, text_batch = batch // 2, batch // 2
        if len(image) < image_batch or len(text) < text_batch:
            raise ValueError("Cross-modal batch exceeds available source features")
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        def lr_factor(step: int) -> float:
            if step < 50:
                return max(1e-5 / lr, step / 50)
            return 0.5 * (1 + math.cos(math.pi * (step - 50) / iterations))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        image_order = torch.randperm(len(image), generator=generator)
        text_order = torch.randperm(len(text), generator=generator)
        image_cursor = text_cursor = 0

        def next_batch(pool, order, cursor, size):
            if cursor + size > len(pool):
                order = torch.randperm(len(pool), generator=generator)
                cursor = 0
            selected = order[cursor:cursor + size].to(self.device)
            return selected, order, cursor + size

        selection = config.get("cross_modal_selection", "final")
        if selection not in {"final", "validation"}:
            raise ValueError("Unknown CrossModal checkpoint selection")
        validation = None
        if selection == "validation":
            validation_x = np.asarray(config["validation_features"], dtype=np.float32)
            validation_y = np.asarray(config["validation_labels"], dtype=np.int64)
            if (validation_x.ndim != 2 or validation_x.shape[1] != self.dim or
                    len(validation_x) != len(validation_y) or not len(validation_y) or
                    validation_y.min() < 0 or validation_y.max() >= self.num_classes):
                raise ValueError("Invalid CrossModal validation set")
            validation = F.normalize(torch.as_tensor(validation_x, device=self.device), dim=1)
            validation_labels = torch.as_tensor(validation_y, device=self.device,
                                                dtype=torch.long)
            best_accuracy = -1.0
            best_weight = None
        for step in range(iterations):
            ii, image_order, image_cursor = next_batch(image, image_order,
                                                       image_cursor, image_batch)
            ti, text_order, text_cursor = next_batch(text, text_order,
                                                     text_cursor, text_batch)
            batch_x = torch.cat([image[ii], text[ti]], dim=0)
            batch_y = torch.cat([image_labels[ii], text_labels[ti]], dim=0)
            scores = 100.0 * self.head(F.normalize(batch_x, dim=1))
            loss = F.cross_entropy(scores, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if validation is not None and step % 100 == 0:
                with torch.inference_mode():
                    accuracy = float((self.head(validation).argmax(dim=1) ==
                                      validation_labels).float().mean())
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_step = step
                        best_weight = self.head.weight.detach().clone()
        if validation is not None:
            with torch.no_grad():
                self.head.weight.copy_(best_weight)
            self.validation_accuracy = best_accuracy
            self.selected_step = best_step
        self.head.eval()
        return self

    def predict_scores(self, query_features):
        query = torch.as_tensor(self._query(query_features), device=self.device,
                                dtype=torch.float32)
        with torch.inference_mode():
            scores = 100.0 * self.head(F.normalize(query, dim=1))
        return scores.cpu().numpy()
