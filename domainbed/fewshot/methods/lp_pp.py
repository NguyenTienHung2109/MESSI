"""LP++ linear probe on frozen CLIP image and text features.

Implements the authors' centroid initialization, analytic step sizes,
classwise text weighting and SGD updates. The OOD classifier returns the
final epoch because the author code selects a checkpoint on validation labels.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .primitives import Classifier, _matrix


class LPPlusPlus(Classifier):
    def fit(self, support_features, support_labels, config):
        x, labels = self._fit_labels(support_features, support_labels, config)
        text = _matrix(config["text_features"])
        if text.shape != (self.num_classes, self.dim):
            raise ValueError("LP++ text shape mismatch")
        counts = np.bincount(labels, minlength=self.num_classes)
        if not np.all(counts == counts[0]):
            raise ValueError("LP++ requires equal shots per class")
        self.device = torch.device(config.get("training_device", "cpu"))
        features = F.normalize(torch.as_tensor(x, device=self.device,
                                               dtype=torch.float32), dim=1)
        self.text = F.normalize(torch.as_tensor(text, device=self.device,
                                               dtype=torch.float32), dim=1)
        y = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        centroids = torch.zeros(self.num_classes, self.dim, device=self.device)
        centroids.index_add_(0, y, features)
        # The released compute_centroids implementation sums support
        # embeddings; it does not divide by the number of shots.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config.get("training_seed", 0)))
            self.classifier = nn.Linear(self.dim, self.num_classes, bias=True).to(self.device)
        with torch.no_grad():
            self.classifier.weight.copy_(centroids)
            # The reference leaves the default random bias initialization.
            text_logits = features @ self.text.T
            alpha_init = 250.0 * text_logits[torch.arange(len(y), device=self.device), y].mean()
            self.alpha = torch.ones(1, self.num_classes, device=self.device) * alpha_init
            max_text_sum = text_logits.square().sum(dim=0).max()
            lr_alpha = len(y) / (4.0 * max_text_sum)
            gram = features.T @ features
            lr_weight = 4.0 * len(y) / torch.linalg.eigvalsh(gram).max()
        self.alpha.requires_grad_()
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=float(lr_weight), momentum=0.9)
        epochs = int(config.get("lp_pp_epochs", 300))
        if epochs < 1:
            raise ValueError("LP++ epochs must be positive")
        selection = config.get("lp_pp_selection", "final")
        if selection not in {"final", "source_views", "validation"}:
            raise ValueError("Unknown LP++ checkpoint selection")
        validation = None
        if selection == "source_views":
            views = np.asarray(config["support_views"], dtype=np.float32)
            if views.ndim != 3 or views.shape[0] != len(labels) or views.shape[2] != self.dim:
                raise ValueError("LP++ requires aligned source views for selection")
            view_index = int(config.get("lp_pp_validation_view", 0))
            if not 0 <= view_index < views.shape[1]:
                raise ValueError("Invalid LP++ validation view")
            validation = F.normalize(torch.as_tensor(views[:, view_index],
                                                     device=self.device), dim=1)
            validation_y = y
        elif selection == "validation":
            validation_x = _matrix(config["validation_features"])
            validation_labels = np.asarray(config["validation_labels"], dtype=np.int64)
            if validation_x.shape[1] != self.dim or len(validation_x) != len(validation_labels):
                raise ValueError("LP++ validation features/labels mismatch")
            if not len(validation_x) or validation_labels.min() < 0 or validation_labels.max() >= self.num_classes:
                raise ValueError("LP++ validation labels are invalid")
            validation = F.normalize(torch.as_tensor(validation_x, device=self.device,
                                                     dtype=torch.float32), dim=1)
            validation_y = torch.as_tensor(validation_labels, device=self.device,
                                           dtype=torch.long)
        if validation is not None:
            validation_text = validation @ self.text.T
            best_metric = float("inf") if selection == "source_views" else -1.0
            best_weight = best_bias = best_alpha = None
        for epoch in range(epochs):
            vision_logits = self.classifier(features)
            scores = vision_logits + self.alpha * text_logits
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Match the author's separate alpha gradient and update cadence.
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    self.alpha -= lr_alpha * self.alpha.grad
            if validation is not None:
                with torch.no_grad():
                    validation_scores = (self.classifier(validation)
                                         + self.alpha * validation_text)
                    if selection == "source_views":
                        metric = float(F.cross_entropy(validation_scores, validation_y))
                        improved = metric < best_metric
                    else:
                        metric = float((validation_scores.argmax(dim=1) == validation_y).float().mean())
                        # The released implementation replaces the selected
                        # checkpoint on ties in validation accuracy.
                        improved = metric >= best_metric
                    if improved:
                        best_metric = metric
                        best_weight = self.classifier.weight.detach().clone()
                        best_bias = self.classifier.bias.detach().clone()
                        best_alpha = self.alpha.detach().clone()
        if validation is not None:
            with torch.no_grad():
                self.classifier.weight.copy_(best_weight)
                self.classifier.bias.copy_(best_bias)
                self.alpha.copy_(best_alpha)
        self.classifier.eval()
        return self

    def predict_scores(self, query_features):
        query = torch.as_tensor(self._query(query_features), device=self.device,
                                dtype=torch.float32)
        with torch.inference_mode():
            query = F.normalize(query, dim=1)
            scores = self.classifier(query) + self.alpha * (query @ self.text.T)
        return scores.cpu().numpy()
