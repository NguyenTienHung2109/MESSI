"""Descriptive diagnostics only; never used to adapt audit features or thresholds."""
import json
import math
import torch
from torch.nn import functional as F


def log_event(path, event):
    if path is not None:
        with open(path, 'a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')


@torch.no_grad()
def prediction_metrics(logits, y):
    probabilities = logits.softmax(1)
    confidence, prediction = probabilities.max(1)
    correct = prediction.eq(y)
    ce = F.cross_entropy(logits, y, reduction='none')
    classes = logits.shape[1]
    bins, ece = [], 0.
    for i in range(15):
        mask = (confidence >= i/15) & (confidence < (i+1)/15 if i < 14 else confidence <= 1)
        n = int(mask.sum())
        if n:
            acc, conf = float(correct[mask].float().mean()), float(confidence[mask].mean())
            ece += n / len(y) * abs(acc-conf)
            bins.append(dict(bin=i, n=n, accuracy=acc, confidence=conf))
    return dict(n=len(y), accuracy=float(correct.float().mean()), cross_entropy=float(ce.mean()),
                mean_confidence=float(confidence.mean()), ece_15=ece,
                brier=float((probabilities-F.one_hot(y, classes)).square().sum(1).mean()),
                confusion=torch.bincount(y*classes+prediction, minlength=classes**2).reshape(classes, classes).tolist(),
                calibration_bins=bins,
                per_class_accuracy={str(c): float(correct[y == c].float().mean()) for c in y.unique().tolist()},
                per_class_ce={str(c): float(ce[y == c].mean()) for c in y.unique().tolist()})


@torch.no_grad()
def bank_behavior(features, logits, weights, y):
    n, experts, _ = logits.shape
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(1)
    features_stats = []
    for z in features.unbind(1):
        centered = z-z.mean(0)
        eigen = torch.linalg.eigvalsh(centered.T @ centered / max(1, n-1)).clamp_min(0)
        p = eigen / eigen.sum().clamp_min(1e-30)
        features_stats.append(dict(variance_trace=float(eigen.sum()),
                                   effective_rank=float((-(p*p.clamp_min(1e-30).log()).sum()).exp()) if eigen.sum() > 0 else 0.,
                                   norm_mean=float(z.norm(dim=1).mean()),
                                   mean_vector_norm=float(z.mean(0).norm())))
    losses = torch.stack([F.cross_entropy(logits[:, i], y, reduction='none') for i in range(experts)], 1)
    mixed = (weights[:, :, None]*logits).sum(1)
    mixed_ce = F.cross_entropy(mixed, y, reduction='none')
    probs = logits.softmax(2)
    js = []
    for i in range(experts):
        row = []
        for j in range(experts):
            mean = (probs[:, i]+probs[:, j])/2
            row.append(float((.5*(probs[:, i]*(probs[:, i].log()-mean.log())).sum(1)
                              + .5*(probs[:, j]*(probs[:, j].log()-mean.log())).sum(1)).mean()))
        js.append(row)
    ablation = []
    for i in range(experts):
        keep = [j for j in range(experts) if j != i]
        # Conditional softmax over remaining experts; log-weights avoid underflow.
        reduced = weights[:, keep].clamp_min(1e-30)
        reduced = reduced/reduced.sum(1, keepdim=True)
        loss = F.cross_entropy((reduced[:, :, None]*logits[:, keep]).sum(1), y)
        ablation.append(float(loss-mixed_ce.mean()))
    return dict(feature_statistics=features_stats, probability_js=js,
                router_entropy=float(entropy.mean()),
                router_entropy_normalized=float(entropy.mean()/math.log(experts)) if experts > 1 else 0.,
                router_effective_experts=float(entropy.exp().mean()),
                router_top1_fraction=(torch.bincount(weights.argmax(1), minlength=experts)/n).tolist(),
                router_mass_by_class={str(c): weights[y == c].mean(0).tolist() for c in y.unique().tolist()},
                oracle_single_expert_ce=float(losses.min(1).values.mean()),
                jensen_slack=float(((weights*losses).sum(1)-mixed_ce).mean()),
                learned_minus_uniform_ce=float(mixed_ce.mean()-F.cross_entropy(logits.mean(1), y)),
                removal_delta_learned_ce=ablation)
