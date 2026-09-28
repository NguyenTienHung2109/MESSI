"""Observational diagnostics; never used in the training objective or selection."""
import numpy as np
import torch


class PredictionEDA:
    def __init__(self, experts, classes):
        self.experts, self.classes = experts, classes
        self.rows = []
        self.feature_sum = None
        self.feature_square_sum = None
        self.cosine_sum = None
        self.count = 0

    @torch.no_grad()
    def add(self, targets, mixed, logits, router, features):
        pi = router.softmax(1)
        probabilities = mixed.softmax(1)
        ce = -logits.log_softmax(2).gather(2, targets[:, None, None].expand(-1, self.experts, 1)).squeeze(2)
        self.rows.append(tuple(t.detach().cpu() for t in
                               (targets, probabilities, logits.argmax(2), pi, ce)))
        z = features.detach().double().cpu()
        sums, squares = z.sum(0), z.square().sum(0)
        cosines = torch.einsum('nmd,nkd->mk', z, z)
        if self.feature_sum is None:
            self.feature_sum, self.feature_square_sum, self.cosine_sum = sums, squares, cosines
        else:
            self.feature_sum += sums
            self.feature_square_sum += squares
            self.cosine_sum += cosines
        self.count += len(targets)

    def finalize(self):
        y, prob, pred, pi, ce = [torch.cat(v) for v in zip(*self.rows)]
        final = prob.argmax(1)
        confidence = prob.max(1).values
        correct = final.eq(y)
        confusion = torch.bincount(y * self.classes + final,
                                   minlength=self.classes**2).reshape(self.classes, self.classes)
        ece, bins = 0., []
        for i in range(10):
            mask = (confidence >= i / 10) & ((confidence < (i+1)/10) if i < 9 else confidence <= 1)
            n = int(mask.sum())
            acc = float(correct[mask].float().mean()) if n else None
            conf = float(confidence[mask].mean()) if n else None
            if n:
                ece += n / len(y) * abs(acc - conf)
            bins.append({'count': n, 'accuracy': acc, 'confidence': conf})
        per_class = []
        for c in range(self.classes):
            mask = y.eq(c)
            n = int(mask.sum())
            per_class.append({'class_id': c, 'count': n,
                'accuracy': float(correct[mask].float().mean()) if n else None,
                'expert_accuracy': pred[mask].eq(y[mask, None]).float().mean(0).tolist() if n else None,
                'expert_risk': ce[mask].mean(0).tolist() if n else None,
                'routing_mean': pi[mask].mean(0).tolist() if n else None})
        entropy = -(pi * pi.clamp_min(1e-30).log()).sum(1)
        oracle = pred.eq(y[:, None]).any(1).float().mean()
        disagreement = (pred[:, :, None] != pred[:, None, :]).float().mean(0)
        return {'n': len(y), 'class_counts': confusion.sum(1).tolist(),
                'confusion_matrix': confusion.tolist(), 'per_class': per_class,
                'routing_mean': pi.mean(0).tolist(),
                'routing_std': pi.std(0, unbiased=False).tolist(),
                'routing_top1_fraction': torch.bincount(pi.argmax(1), minlength=self.experts).div(len(y)).tolist(),
                'routing_entropy': float(entropy.mean()),
                'routing_effective_experts': float(entropy.exp().mean()),
                'routing_max_quantiles': torch.quantile(pi.max(1).values, torch.tensor([0., .1, .5, .9, 1.])).tolist(),
                'expert_oracle_accuracy': float(oracle),
                'expert_disagreement': disagreement.tolist(),
                'feature_variance_trace': (self.feature_square_sum / self.count - (self.feature_sum / self.count).square()).sum(1).tolist(),
                'feature_mean_norm': (self.feature_sum / self.count).norm(dim=1).tolist(),
                'expert_feature_cosine': (self.cosine_sum / self.count).tolist(),
                'ece_10bins': ece, 'calibration_bins': bins,
                'brier_score': float((prob - torch.nn.functional.one_hot(y, self.classes)).square().sum(1).mean()),
                'confidence_mean': float(confidence.mean())}

    def save(self, path):
        arrays = [torch.cat(v).numpy() for v in zip(*self.rows)]
        np.savez_compressed(path, **dict(zip(['labels', 'probabilities', 'expert_predictions', 'routing', 'expert_ce'], arrays)))
