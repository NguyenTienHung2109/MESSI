"""Subset-IRM extension for the explicit-head MESSI implementation.

The legacy :class:`domainbed.algorithms.MESSI` path is not modified.  This
module provides a separately selected algorithm plus small, independently
testable mathematical building blocks.
"""

import copy
from collections import OrderedDict
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed import algorithms
from domainbed.deit_transformer import (
    loss_balance,
    loss_diversity,
    loss_inv_MMD,
    loss_sparse,
)


EPS = 1e-8


def per_sample_topk(probabilities, k, eps=EPS):
    """Keep and renormalize exactly ``k`` entries in every row."""
    if probabilities.ndim != 2:
        raise ValueError("routing probabilities must have shape [batch, experts]")
    experts = probabilities.shape[1]
    if k is None or int(k) >= experts:
        return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(eps)
    k = int(k)
    if k < 1:
        raise ValueError("router_topk must be positive or null")
    _, indices = torch.topk(probabilities, k=k, dim=1, sorted=True)
    mask = torch.zeros_like(probabilities).scatter_(1, indices, 1.0)
    selected = probabilities * mask
    return selected / selected.sum(dim=1, keepdim=True).clamp_min(eps)


class ExpertClassifierHeads(nn.Module):
    """One linear classifier per expert, identically initialized."""

    def __init__(self, num_experts, feature_dim, num_classes,
                 shared_classifier=None):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(feature_dim, num_classes) for _ in range(num_experts)
        ])
        with torch.no_grad():
            if (shared_classifier is not None and
                    shared_classifier.weight.shape == self.heads[0].weight.shape):
                weight = shared_classifier.weight.detach()
                bias = (shared_classifier.bias.detach()
                        if shared_classifier.bias is not None else None)
            else:
                weight = self.heads[0].weight.detach().clone()
                bias = (self.heads[0].bias.detach().clone()
                        if self.heads[0].bias is not None else None)
            for head in self.heads:
                head.weight.copy_(weight)
                if bias is not None and head.bias is not None:
                    head.bias.copy_(bias)

    def forward(self, expert_features):
        if expert_features.ndim != 3:
            raise ValueError("expert features must have shape [batch, experts, dim]")
        if expert_features.shape[1] != len(self.heads):
            raise ValueError("expert feature count does not match classifier heads")
        return torch.stack([
            head(expert_features[:, index])
            for index, head in enumerate(self.heads)
        ], dim=1)


def expert_domain_risk(expert_logits, targets, domain_ids, responsibilities,
                       num_domains, min_effective_samples=2.0,
                       detach_responsibility=True, eps=EPS):
    """Return routing-weighted risk, ESS, validity, and support matrices."""
    if expert_logits.ndim != 3:
        raise ValueError("expert_logits must have shape [batch, experts, classes]")
    gamma = responsibilities.detach() if detach_responsibility else responsibilities
    # Keep numerically sensitive reductions in FP32 under autocast.
    per_sample_ce = torch.stack([
        F.cross_entropy(expert_logits[:, m].float(), targets, reduction="none")
        for m in range(expert_logits.shape[1])
    ], dim=1)
    gamma32 = gamma.float()
    risks = per_sample_ce.new_zeros((num_domains, expert_logits.shape[1]))
    ess = risks.clone()
    support = risks.clone()
    valid = torch.zeros_like(risks, dtype=torch.bool)
    for domain in range(num_domains):
        domain_mask = domain_ids.eq(domain).unsqueeze(1)
        weights = gamma32 * domain_mask
        weight_sum = weights.sum(dim=0)
        squared_sum = weights.square().sum(dim=0)
        support[domain] = weight_sum
        ess[domain] = weight_sum.square() / squared_sum.clamp_min(eps)
        has_support = weight_sum > eps
        risks[domain] = torch.where(
            has_support,
            (weights * per_sample_ce).sum(dim=0) / weight_sum.clamp_min(eps),
            torch.zeros_like(weight_sum),
        )
        valid[domain] = has_support & ess[domain].ge(float(min_effective_samples))
    return risks, ess, valid, support


def expert_predictive_loss(expert_logits, targets, responsibilities,
                           detach_responsibility=True):
    """Routing-weighted per-sample expert cross entropy."""
    gamma = responsibilities.detach() if detach_responsibility else responsibilities
    ce = torch.stack([
        F.cross_entropy(expert_logits[:, m].float(), targets, reduction="none")
        for m in range(expert_logits.shape[1])
    ], dim=1)
    return (gamma.float() * ce).sum(dim=1).mean()


def global_irm_penalty(logits, targets, domain_ids, num_domains):
    """IRMv1 penalty for a global expert evaluated on every source domain."""
    if logits.ndim != 2:
        raise ValueError("global logits must have shape [batch, classes]")
    if targets.ndim != 1 or targets.numel() != logits.shape[0]:
        raise ValueError("targets must match global logits")
    if domain_ids.ndim != 1 or domain_ids.numel() != logits.shape[0]:
        raise ValueError("domain_ids must match global logits")
    penalties = []
    for domain in range(int(num_domains)):
        mask = domain_ids.eq(domain)
        if not bool(mask.any()):
            continue
        scale = logits.new_ones((), requires_grad=True)
        risk = F.cross_entropy(logits[mask].float() * scale, targets[mask])
        gradient = torch.autograd.grad(risk, [scale], create_graph=True)[0]
        penalties.append(gradient.square())
    if not penalties:
        return logits.sum() * 0.0
    return torch.stack(penalties).mean()


def expert_representation_diagnostics(expert_features, expert_logits,
                                      global_features, global_logits,
                                      responsibilities, eps=EPS):
    """Describe routing collapse separately from representation redundancy."""
    if expert_features.ndim != 3 or expert_logits.ndim != 3:
        raise ValueError("expert tensors must have shape [batch, experts, dim]")
    if global_features.ndim != 2 or global_logits.ndim != 2:
        raise ValueError("global tensors must have shape [batch, dim]")
    features = torch.cat([global_features.unsqueeze(1), expert_features], dim=1).float()
    logits = torch.cat([global_logits.unsqueeze(1), expert_logits], dim=1).float()
    normalized_features = F.normalize(features, dim=-1, eps=eps)
    normalized_logits = F.normalize(logits, dim=-1, eps=eps)
    feature_cosine = torch.einsum(
        "bmd,bnd->mn", normalized_features, normalized_features
    ) / max(features.shape[0], 1)
    logit_cosine = torch.einsum(
        "bmc,bnc->mn", normalized_logits, normalized_logits
    ) / max(logits.shape[0], 1)

    centered = features - features.mean(dim=0, keepdim=True)
    grams = torch.einsum("bmd,cmd->mbc", centered, centered)
    grams = grams - grams.mean(dim=1, keepdim=True)
    grams = grams - grams.mean(dim=2, keepdim=True)
    grams = grams + grams.mean(dim=(1, 2), keepdim=True)
    flat_grams = grams.flatten(1)
    cka = flat_grams @ flat_grams.T
    cka = cka / (
        flat_grams.norm(dim=1).unsqueeze(1)
        * flat_grams.norm(dim=1).unsqueeze(0)
    ).clamp_min(eps)

    predictions = logits.argmax(dim=-1)
    agreement = predictions.unsqueeze(2).eq(predictions.unsqueeze(1)).float().mean(0)
    selected = responsibilities.gt(0)
    selected_counts = selected.sum(dim=0)
    selected_global_cosine = []
    for expert in range(expert_features.shape[1]):
        mask = selected[:, expert]
        if bool(mask.any()):
            value = F.cosine_similarity(
                global_features[mask].float(),
                expert_features[mask, expert].float(), dim=-1, eps=eps,
            ).mean()
        else:
            value = features.new_zeros(())
        selected_global_cosine.append(value)

    return {
        "labels": ["global"] + [
            f"subset_{index}" for index in range(expert_features.shape[1])
        ],
        "feature_cosine_matrix": feature_cosine.detach(),
        "feature_linear_cka_matrix": cka.detach(),
        "logit_cosine_matrix": logit_cosine.detach(),
        "prediction_agreement_matrix": agreement.detach(),
        "feature_variance": centered.square().mean(dim=(0, 2)).detach(),
        "feature_centroid": features.mean(dim=0).detach(),
        "subset_selected_count": selected_counts.detach(),
        "global_to_selected_subset_cosine": torch.stack(
            selected_global_cosine
        ).detach(),
    }


def top_r_assignment(risk, valid, top_r=2, temperature=1.0, eps=EPS):
    """Convert lower source-only risks into detached row-wise Top-r scores."""
    if risk.ndim != 2 or valid.shape != risk.shape:
        raise ValueError("risk and valid must have matching [domains, experts] shape")
    if temperature <= 0:
        raise ValueError("assignment_temperature must be positive")
    result = torch.zeros_like(risk)
    for domain in range(risk.shape[0]):
        eligible = valid[domain] & torch.isfinite(risk[domain])
        count = int(eligible.sum().item())
        if count == 0:
            continue
        retain = min(int(top_r), count)
        if retain < 1:
            raise ValueError("assignment_topr must be positive")
        scores = -risk[domain] / float(temperature)
        scores = scores.masked_fill(~eligible, -torch.inf)
        _, indices = torch.topk(scores, k=retain)
        chosen_scores = scores[indices]
        chosen = F.softmax(chosen_scores, dim=0)
        result[domain].scatter_(0, indices, chosen)
    return result.detach()


def router_distillation_loss(dense_probabilities, domain_ids, assignment,
                             eps=EPS):
    """KL(stopgrad(Q_domain) || dense instance-router probabilities)."""
    teacher = assignment.detach()[domain_ids]
    log_teacher = torch.where(
        teacher > 0, teacher.clamp_min(eps).log(), torch.zeros_like(teacher)
    )
    log_student = dense_probabilities.clamp_min(eps).log()
    return (teacher * (log_teacher - log_student)).sum(dim=1).mean()


def q_capacity_loss(dense_probabilities, domain_ids, num_domains,
                    rho_max, eps=EPS):
    """Capacity penalty on a differentiable source-domain Q proxy.

    The risk-EMA assignment is hard Top-r and detached. To make the Q-level
    capacity constraint trainable, construct one row per source domain from
    its mean dense routing probabilities.
    """
    if dense_probabilities.ndim != 2:
        raise ValueError("routing probabilities must have shape [batch, experts]")
    if domain_ids.ndim != 1 or domain_ids.numel() != dense_probabilities.shape[0]:
        raise ValueError("domain_ids must match the routing batch")
    if int(num_domains) < 1:
        raise ValueError("num_domains must be positive")
    if not 0.0 < float(rho_max) <= 1.0:
        raise ValueError("rho_max must be in (0, 1]")

    rows = []
    for domain in range(int(num_domains)):
        mask = domain_ids.eq(domain)
        if not bool(mask.any()):
            raise ValueError("every source domain must be present in the batch")
        row = dense_probabilities[mask].float().mean(dim=0)
        rows.append(row / row.sum().clamp_min(eps))
    q_proxy = torch.stack(rows, dim=0)
    mean_mass = q_proxy.mean(dim=0)
    violation = F.relu(mean_mass - float(rho_max))
    loss = violation.square().mean()
    info = {
        "q_proxy": q_proxy.detach(),
        "mean_mass": mean_mass.detach(),
        "violation": violation.detach(),
        "max_mean_mass": float(mean_mass.max().detach().item()),
        "max_violation": float(violation.max().detach().item()),
        "violating_expert_count": int(violation.gt(0).sum().detach().item()),
    }
    return loss, info


def _entropic_ot_objective(first, second, epsilon=0.1, sinkhorn_iters=30):
    """Differentiable entropic OT objective with uniform empirical weights."""
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("Sinkhorn inputs must have shape [samples, features]")
    if first.shape[1] != second.shape[1]:
        raise ValueError("Sinkhorn inputs must share feature dimension")
    if first.shape[0] == 0 or second.shape[0] == 0:
        raise ValueError("Sinkhorn inputs must be non-empty")
    if epsilon <= 0:
        raise ValueError("Sinkhorn epsilon must be positive")
    if int(sinkhorn_iters) < 1:
        raise ValueError("Sinkhorn iterations must be positive")

    first = first.float()
    second = second.float()
    cost = (
        first.square().sum(dim=1, keepdim=True)
        + second.square().sum(dim=1).unsqueeze(0)
        - 2.0 * first @ second.T
    ).clamp_min(0.0)
    n, m = cost.shape
    log_mu = cost.new_full((n,), -torch.log(cost.new_tensor(float(n))))
    log_nu = cost.new_full((m,), -torch.log(cost.new_tensor(float(m))))
    log_kernel = -cost / float(epsilon)
    log_u = cost.new_zeros(n)
    log_v = cost.new_zeros(m)
    for _ in range(int(sinkhorn_iters)):
        log_u = log_mu - torch.logsumexp(
            log_kernel + log_v.unsqueeze(0), dim=1
        )
        log_v = log_nu - torch.logsumexp(
            log_kernel + log_u.unsqueeze(1), dim=0
        )

    log_plan = log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)
    plan = log_plan.exp()
    reference_log_density = log_mu.unsqueeze(1) + log_nu.unsqueeze(0)
    transport = (plan * cost).sum()
    relative_entropy = (
        plan * (log_plan - reference_log_density)
    ).sum()
    objective = transport + float(epsilon) * relative_entropy
    marginal_error = torch.maximum(
        (plan.sum(dim=1) - log_mu.exp()).abs().max(),
        (plan.sum(dim=0) - log_nu.exp()).abs().max(),
    )
    return objective, marginal_error


def sinkhorn_divergence(first, second, epsilon=0.1, sinkhorn_iters=30):
    """Debiased entropic OT discrepancy and detached solver diagnostics."""
    cross, error_cross = _entropic_ot_objective(
        first, second, epsilon, sinkhorn_iters
    )
    self_first, error_first = _entropic_ot_objective(
        first, first, epsilon, sinkhorn_iters
    )
    self_second, error_second = _entropic_ot_objective(
        second, second, epsilon, sinkhorn_iters
    )
    raw = cross - 0.5 * self_first - 0.5 * self_second
    value = raw.clamp_min(0.0)
    return value, {
        "raw": raw.detach(),
        "marginal_error": torch.maximum(
            error_cross, torch.maximum(error_first, error_second)
        ).detach(),
    }


def q_gated_sinkhorn_divergence(
        expert_features, targets, domain_ids, assignment, num_classes,
        epsilon=0.1, sinkhorn_iters=30, min_samples=2,
        normalize_features=True):
    """Class-conditional Sinkhorn divergence on domain pairs shared by Q."""
    if expert_features.ndim != 3:
        raise ValueError("expert_features must have shape [batch, experts, dim]")
    if targets.ndim != 1 or domain_ids.ndim != 1:
        raise ValueError("targets and domain_ids must have shape [batch]")
    if targets.numel() != expert_features.shape[0]:
        raise ValueError("targets must match expert feature batch dimension")
    if domain_ids.numel() != expert_features.shape[0]:
        raise ValueError("domain_ids must match expert feature batch dimension")
    num_domains, num_experts = assignment.shape
    if num_experts != expert_features.shape[1]:
        raise ValueError("Q expert count must match expert_features")
    if int(min_samples) < 1:
        raise ValueError("SSI-OT min_samples must be positive")

    features = expert_features.float()
    if normalize_features:
        features = F.normalize(features, p=2, dim=-1, eps=EPS)
    q = assignment.detach().float()
    q_support_cpu = q.gt(0).detach().cpu()
    numerator = features.sum() * 0.0
    denominator = features.new_zeros(())
    divergences = []
    marginal_errors = []
    negative_raw_pairs = 0
    cells = {}
    for class_index in range(int(num_classes)):
        class_mask = targets.eq(class_index)
        for domain in range(num_domains):
            mask = class_mask & domain_ids.eq(domain)
            if int(mask.sum().item()) >= int(min_samples):
                cells[class_index, domain] = mask

    for expert in range(num_experts):
        for class_index in range(int(num_classes)):
            for first_domain in range(num_domains):
                first_mask = cells.get((class_index, first_domain))
                if first_mask is None:
                    continue
                for second_domain in range(first_domain + 1, num_domains):
                    if not (
                        bool(q_support_cpu[first_domain, expert])
                        and bool(q_support_cpu[second_domain, expert])
                    ):
                        continue
                    second_mask = cells.get((class_index, second_domain))
                    if second_mask is None:
                        continue
                    weight = q[first_domain, expert] * q[second_domain, expert]
                    divergence, diagnostics = sinkhorn_divergence(
                        features[first_mask, expert],
                        features[second_mask, expert],
                        epsilon=epsilon,
                        sinkhorn_iters=sinkhorn_iters,
                    )
                    numerator = numerator + weight * divergence
                    denominator = denominator + weight
                    divergences.append(divergence.detach())
                    marginal_errors.append(diagnostics["marginal_error"])
                    negative_raw_pairs += int(bool(diagnostics["raw"] < 0))

    active = bool(divergences) and bool(denominator.detach() > 0)
    if not active:
        zero = features.sum() * 0.0
        return zero, {
            "active": False,
            "active_pair_count": 0,
            "weight_sum": 0.0,
            "divergence_mean": 0.0,
            "divergence_min": 0.0,
            "divergence_max": 0.0,
            "marginal_error_max": 0.0,
            "negative_raw_pair_count": 0,
        }

    values = torch.stack(divergences)
    errors = torch.stack(marginal_errors)
    return numerator / denominator.clamp_min(EPS), {
        "active": True,
        "active_pair_count": len(divergences),
        "weight_sum": float(denominator.detach().item()),
        "divergence_mean": float(values.mean().item()),
        "divergence_min": float(values.min().item()),
        "divergence_max": float(values.max().item()),
        "marginal_error_max": float(errors.max().item()),
        "negative_raw_pair_count": negative_raw_pairs,
    }


def subset_irm_penalty(expert_logits, targets, domain_ids, responsibilities,
                       assignment, min_effective_samples=2.0,
                       detach_responsibility=True, eps=EPS):
    """IRMv1 penalty within source-domain subsets selected by detached Q."""
    gamma = (responsibilities.detach() if detach_responsibility
             else responsibilities).float()
    q = assignment.detach().float()
    num_domains, num_experts = q.shape
    _, ess, valid, _ = expert_domain_risk(
        expert_logits, targets, domain_ids, gamma, num_domains,
        min_effective_samples=min_effective_samples,
        detach_responsibility=False,
        eps=eps,
    )
    assigned_valid = valid & q.gt(0)
    active_domains = assigned_valid.sum(dim=0)
    eligible_experts = active_domains.ge(2)
    numerator = expert_logits.sum() * 0.0
    denominator = q.new_zeros(())
    active_cells = torch.zeros_like(assigned_valid)
    for expert in range(num_experts):
        if not bool(eligible_experts[expert]):
            continue
        for domain in range(num_domains):
            if not bool(assigned_valid[domain, expert]):
                continue
            mask = domain_ids.eq(domain)
            weights = gamma[mask, expert]
            weight_sum = weights.sum()
            if not bool(weight_sum > eps):
                continue
            scale = expert_logits.new_ones((), requires_grad=True)
            cell_ce = F.cross_entropy(
                scale.float() * expert_logits[mask, expert].float(),
                targets[mask], reduction="none",
            )
            cell_risk = (weights * cell_ce).sum() / weight_sum.clamp_min(eps)
            gradient = torch.autograd.grad(
                cell_risk, scale, create_graph=True, retain_graph=True
            )[0]
            weight = q[domain, expert]
            numerator = numerator + weight * gradient.square()
            denominator = denominator + weight
            active_cells[domain, expert] = True
    penalty = numerator / denominator.clamp_min(eps)
    active = bool(eligible_experts.any() and denominator.detach().item() > 0)
    return penalty, {
        "ess": ess.detach(),
        "valid_cells": valid.detach(),
        "active_cells": active_cells.detach(),
        "active_domains_per_expert": active_domains.detach(),
        "active": active,
    }


class SourceRiskEMA(nn.Module):
    """Detached source-only risk EMA used to construct Q."""

    def __init__(self, num_domains, num_experts, beta=0.9):
        super().__init__()
        self.beta = float(beta)
        self.register_buffer("risk", torch.zeros(num_domains, num_experts))
        self.register_buffer("valid", torch.zeros(
            num_domains, num_experts, dtype=torch.bool
        ))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, probe_risk, probe_valid):
        finite = probe_valid & torch.isfinite(probe_risk)
        first = finite & ~self.valid
        continuing = finite & self.valid
        self.risk[first] = probe_risk[first]
        self.risk[continuing] = (
            self.beta * self.risk[continuing]
            + (1.0 - self.beta) * probe_risk[continuing]
        )
        self.valid |= finite
        self.updates += 1

    @property
    def ready(self):
        return bool(self.valid.any(dim=1).all())


def _group_gradient_norm(loss, parameters):
    params = [p for p in parameters if p.requires_grad]
    if not params or not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True
    )
    squared = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    return float(squared.sqrt().item())


def q_capacity_lambda_at_step(step, base_lambda, anneal_steps=0,
                              ramp_steps=0, post_step=-1,
                              post_lambda=None, decay_start_step=-1,
                              decay_end_step=-1):
    """Compute the Q-capacity coefficient for one optimizer step."""
    step = int(step)
    base_lambda = float(base_lambda)
    anneal_steps = int(anneal_steps)
    ramp_steps = int(ramp_steps)
    post_step = int(post_step)
    post_lambda = (base_lambda if post_lambda is None
                   else float(post_lambda))
    decay_start_step = int(decay_start_step)
    decay_end_step = int(decay_end_step)

    decay_enabled = decay_start_step >= 0 or decay_end_step >= 0
    if decay_enabled and not (0 <= decay_start_step < decay_end_step):
        raise ValueError(
            "Q-capacity decay requires 0 <= decay_start_step < decay_end_step"
        )

    if step < anneal_steps:
        coefficient = 0.0
    elif ramp_steps > 0:
        coefficient = base_lambda * min(
            1.0, max(0.0, (step - anneal_steps) / ramp_steps)
        )
    else:
        coefficient = base_lambda

    if post_step >= 0 and step >= post_step:
        coefficient = post_lambda

    if decay_enabled:
        if step >= decay_end_step:
            coefficient = 0.0
        elif step > decay_start_step:
            coefficient *= (
                (decay_end_step - step) /
                (decay_end_step - decay_start_step)
            )
    return coefficient


class MESSISubsetIRM(algorithms.MESSI):
    """Config-gated Subset-IRM algorithm; disabled mode delegates to MESSI."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.subset_enabled = bool(hparams.get("subset_irm_enabled", False))
        self.prediction_mode = hparams.get(
            "subset_irm_prediction_mode", "shared_feature_mix"
        )
        if self.prediction_mode not in ("shared_feature_mix", "expert_logit_mix"):
            raise ValueError("invalid subset_irm_prediction_mode")
        self.router_topk = hparams.get("subset_irm_router_topk")
        self.topk_warmup_steps = int(hparams.get(
            "subset_irm_topk_warmup_steps", 0
        ))
        self.use_expert_heads = bool(hparams.get(
            "subset_irm_use_expert_heads", self.subset_enabled
        ))
        self.lambda_expert = float(hparams.get("subset_irm_lambda_expert", 0.0))
        self.lambda_route = float(hparams.get("subset_irm_lambda_route", 0.0))
        self.lambda_sirm = float(hparams.get("subset_irm_lambda_sirm", 0.0))
        self.lambda_q_capacity = float(hparams.get(
            "subset_irm_lambda_q_capacity", 0.0
        ))
        self.q_capacity_rho_max = float(hparams.get(
            "subset_irm_q_capacity_rho_max", 1.0
        ))
        if not 0.0 < self.q_capacity_rho_max <= 1.0:
            raise ValueError("subset_irm_q_capacity_rho_max must be in (0, 1]")
        self.q_capacity_anneal_steps = int(hparams.get(
            "subset_irm_q_capacity_anneal_steps", 0
        ))
        self.q_capacity_ramp_steps = int(hparams.get(
            "subset_irm_q_capacity_ramp_steps", 0
        ))
        self.q_capacity_post_step = int(hparams.get(
            "subset_irm_q_capacity_post_step", -1
        ))
        self.q_capacity_post_lambda = float(hparams.get(
            "subset_irm_q_capacity_post_lambda", self.lambda_q_capacity
        ))
        self.q_capacity_decay_start_step = int(hparams.get(
            "subset_irm_q_capacity_decay_start_step", -1
        ))
        self.q_capacity_decay_end_step = int(hparams.get(
            "subset_irm_q_capacity_decay_end_step", -1
        ))
        decay_enabled = (self.q_capacity_decay_start_step >= 0 or
                         self.q_capacity_decay_end_step >= 0)
        if decay_enabled and not (
                0 <= self.q_capacity_decay_start_step <
                self.q_capacity_decay_end_step):
            raise ValueError(
                "Q-capacity decay requires 0 <= decay_start_step < "
                "decay_end_step"
            )
        self.ssi_mode = hparams.get("subset_irm_ssi_mode", "mmd")
        if self.ssi_mode not in ("none", "mmd", "q_sinkhorn"):
            raise ValueError(
                "subset_irm_ssi_mode must be none, mmd, or q_sinkhorn"
            )
        self.ssi_anneal_steps = int(hparams.get(
            "subset_irm_ssi_anneal_steps", 0
        ))
        self.ssi_ramp_steps = int(hparams.get(
            "subset_irm_ssi_ramp_steps", 0
        ))
        self.ssi_ot_epsilon = float(hparams.get(
            "subset_irm_ssi_ot_epsilon", 0.1
        ))
        self.ssi_ot_iters = int(hparams.get(
            "subset_irm_ssi_ot_iters", 30
        ))
        self.ssi_ot_min_samples = int(hparams.get(
            "subset_irm_ssi_ot_min_samples", 2
        ))
        self.ssi_ot_normalize_features = bool(hparams.get(
            "subset_irm_ssi_ot_normalize_features", True
        ))
        self.route_anneal_steps = int(hparams.get(
            "subset_irm_route_anneal_steps", 0
        ))
        self.route_ramp_steps = int(hparams.get(
            "subset_irm_route_ramp_steps", 0
        ))
        self.sirm_ramp_steps = int(hparams.get(
            "subset_irm_sirm_ramp_steps", 0
        ))
        self.post_warmup_lambda_sp = float(hparams.get(
            "subset_irm_post_warmup_lambda_sp", self.lambda_sp
        ))
        self.post_warmup_lambda_bal = float(hparams.get(
            "subset_irm_post_warmup_lambda_bal", self.lambda_bal
        ))
        self.responsibility_detach = bool(hparams.get(
            "subset_irm_responsibility_detach", True
        ))
        self.assignment_mode = hparams.get(
            "subset_irm_assignment_mode", "risk_ema"
        )
        if self.assignment_mode not in ("risk_ema", "routing_mass"):
            raise ValueError("assignment mode must be risk_ema or routing_mass")
        self.assignment_topr = int(hparams.get("subset_irm_assignment_topr", 2))
        self.assignment_temperature = float(hparams.get(
            "subset_irm_assignment_temperature", 1.0
        ))
        self.probe_interval = int(hparams.get("subset_irm_probe_interval", 100))
        self.min_effective_samples = float(hparams.get(
            "subset_irm_min_effective_samples", 2
        ))
        self.sirm_anneal_steps = int(hparams.get(
            "subset_irm_sirm_anneal_steps", 0
        ))
        self.gradient_log_interval = int(hparams.get(
            "subset_irm_gradient_log_interval", 0
        ))
        self.global_expert_enabled = bool(hparams.get(
            "subset_irm_global_expert_enabled", False
        ))
        self.global_training_mode = hparams.get(
            "subset_irm_global_training_mode", "joint"
        )
        if self.global_training_mode not in ("joint", "separate"):
            raise ValueError(
                "subset_irm_global_training_mode must be joint or separate"
            )
        if self.global_training_mode == "separate" and not self.global_expert_enabled:
            raise ValueError("separate global training requires global expert")
        self.global_logit_weight = float(hparams.get(
            "subset_irm_global_logit_weight", 1.0
        ))
        self.lambda_global_expert = float(hparams.get(
            "subset_irm_lambda_global_expert", 1.0
        ))
        self.lambda_global_irm = float(hparams.get(
            "subset_irm_lambda_global_irm", 0.0
        ))
        self.global_irm_anneal_steps = int(hparams.get(
            "subset_irm_global_irm_anneal_steps", 0
        ))
        self.global_irm_ramp_steps = int(hparams.get(
            "subset_irm_global_irm_ramp_steps", 0
        ))
        self.similarity_log_interval = int(hparams.get(
            "subset_irm_similarity_log_interval", 100
        ))
        self.last_diagnostics = {}

        if not self.subset_enabled:
            return
        if not self.use_expert_heads:
            raise ValueError("Subset-IRM enabled mode requires expert heads")
        self.expert_heads = ExpertClassifierHeads(
            self.num_experts, self.moe_head.expert_dim, self.num_classes,
            shared_classifier=self.moe_head.classifier,
        ).to(next(self.moe_head.parameters()).device)
        if self.global_expert_enabled:
            # Same capacity as one subset expert, but it always receives every
            # source sample and never participates in Q or top-k routing.
            self.global_expert = copy.deepcopy(self.moe_head.experts[0])
            self.global_head = copy.deepcopy(self.expert_heads.heads[0])
            if self.global_training_mode == "separate":
                # No parameter is shared between the global and subset paths.
                self.global_featurizer = copy.deepcopy(self.featurizer)
                self.global_input_proj = copy.deepcopy(self.moe_head.input_proj)
        self.risk_ema = SourceRiskEMA(
            num_domains, self.num_experts,
            beta=float(hparams.get("subset_irm_assignment_ema", 0.9)),
        ).to(next(self.moe_head.parameters()).device)
        self.register_buffer("subset_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("q_active_step", torch.full(
            (), -1, dtype=torch.long
        ))
        self.register_buffer("sirm_active_steps", torch.zeros((), dtype=torch.long))

        subset_trainable = [p for p in chain(
            self.featurizer.parameters(), self.moe_head.parameters(),
            self.expert_heads.parameters(),
        ) if p.requires_grad]
        global_trainable = [p for p in chain(
            self.global_featurizer.parameters()
            if self.global_training_mode == "separate" else [],
            self.global_input_proj.parameters()
            if self.global_training_mode == "separate" else [],
            self.global_expert.parameters() if self.global_expert_enabled else [],
            self.global_head.parameters() if self.global_expert_enabled else [],
        ) if p.requires_grad]
        if self.global_training_mode == "separate":
            self.optimizer = torch.optim.Adam(
                subset_trainable,
                lr=hparams.get("lr", 0.0),
                weight_decay=hparams.get("weight_decay", 0.0),
            )
            self.global_optimizer = torch.optim.Adam(
                global_trainable,
                lr=hparams.get("lr", 0.0),
                weight_decay=hparams.get("weight_decay", 0.0),
            )
        else:
            self.optimizer = torch.optim.Adam(
                subset_trainable + global_trainable,
                lr=hparams.get("lr", 0.0),
                weight_decay=hparams.get("weight_decay", 0.0),
            )

    def _global_forward(self, x, shared_z_moe=None):
        if not self.global_expert_enabled:
            return None, None
        if self.global_training_mode == "separate":
            global_z = self.global_featurizer(x)
            global_z_moe = self.global_input_proj(global_z)
        else:
            global_z_moe = shared_z_moe
        global_features = self.global_expert(global_z_moe)
        return global_features, self.global_head(global_features)

    def _routing_responsibilities(self, dense_pi):
        if (self.router_topk is None or
                int(self.subset_step.item()) < self.topk_warmup_steps):
            return dense_pi
        return per_sample_topk(dense_pi, self.router_topk)

    def _subset_forward_components(self, x):
        z = self.featurizer(x)
        z_moe = self.moe_head.input_proj(z)
        h_stack = torch.stack([
            expert(z_moe) for expert in self.moe_head.experts
        ], dim=1)
        dense_pi = F.softmax(self.moe_head.router(z_moe), dim=-1)
        gamma = self._routing_responsibilities(dense_pi)
        shared_features = (dense_pi.unsqueeze(-1) * h_stack).sum(dim=1)
        shared_logits = self.moe_head.classifier(shared_features)
        expert_logits = self.expert_heads(h_stack)
        if self.prediction_mode == "expert_logit_mix":
            subset_logits = (gamma.unsqueeze(-1) * expert_logits).sum(dim=1)
        else:
            subset_logits = shared_logits
        if self.global_expert_enabled:
            global_features, global_logits = self._global_forward(
                x, shared_z_moe=z_moe
            )
            logits = subset_logits + self.global_logit_weight * global_logits
        else:
            global_features = None
            global_logits = None
            logits = subset_logits
        return (logits, dense_pi, gamma, h_stack, expert_logits, shared_logits,
                subset_logits, global_features, global_logits)

    def _subset_forward(self, x):
        # Preserve the public tuple used by existing diagnostics and tests.
        return self._subset_forward_components(x)[:6]

    def _forward(self, x):
        if not self.subset_enabled:
            return super()._forward(x)
        logits, dense_pi, _, h_stack, _, _ = self._subset_forward(x)
        return logits, dense_pi, h_stack

    def _probe_risk(self, expert_logits, targets, domain_ids):
        with torch.no_grad():
            uniform = torch.ones_like(expert_logits[:, :, 0])
            risk, _, valid, _ = expert_domain_risk(
                expert_logits.detach(), targets, domain_ids, uniform,
                self.num_domains, min_effective_samples=1,
                detach_responsibility=True,
            )
        return risk, valid

    def _assignment(self, dense_pi, domain_ids):
        if self.assignment_mode == "risk_ema":
            if not self.risk_ema.ready:
                return torch.zeros_like(self.risk_ema.risk)
            return top_r_assignment(
                self.risk_ema.risk, self.risk_ema.valid,
                self.assignment_topr, self.assignment_temperature,
            )
        mass = dense_pi.new_zeros(self.num_domains, self.num_experts)
        valid = torch.zeros_like(mass, dtype=torch.bool)
        for domain in range(self.num_domains):
            mask = domain_ids.eq(domain)
            if bool(mask.any()):
                mass[domain] = dense_pi[mask].mean(dim=0)
                valid[domain] = True
        # top_r_assignment expects lower values to be better.
        return top_r_assignment(
            -mass, valid, self.assignment_topr,
            self.assignment_temperature,
        )

    def _gradient_diagnostics(self, losses, prefix="grad"):
        groups = OrderedDict([
            ("backbone", list(self.featurizer.parameters())),
            ("router", list(self.moe_head.router.parameters())),
            ("experts", list(self.moe_head.experts.parameters())),
            ("expert_heads", list(self.expert_heads.parameters())),
        ])
        if self.global_expert_enabled:
            groups["global_expert"] = list(self.global_expert.parameters())
            groups["global_head"] = list(self.global_head.parameters())
        if self.global_training_mode == "separate":
            groups["global_backbone"] = list(self.global_featurizer.parameters())
            groups["global_input_proj"] = list(
                self.global_input_proj.parameters()
            )
        result = {}
        for loss_name, loss in losses.items():
            for group_name, parameters in groups.items():
                result[f"{prefix}_{loss_name}_{group_name}"] = _group_gradient_norm(
                    loss, parameters
                )
        return result

    def update(self, minibatches, unlabeled=None):
        if not self.subset_enabled:
            return super().update(minibatches, unlabeled)
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])
        domain_ids = self._get_domain_ids(minibatches)
        (logits, dense_pi, gamma, h_stack, expert_logits, _, subset_logits,
         global_features, global_logits) = self._subset_forward_components(all_x)

        # In separate mode the combined prediction is evaluation-only. The
        # subset classification loss cannot update global parameters.
        classification_logits = (
            subset_logits if self.global_training_mode == "separate" else logits
        )
        l_cls = F.cross_entropy(classification_logits, all_y)
        l_combined_eval = F.cross_entropy(logits, all_y)
        l_expert = expert_predictive_loss(
            expert_logits, all_y, gamma, self.responsibility_detach
        )
        if self.global_expert_enabled:
            l_global_expert = F.cross_entropy(global_logits, all_y)
            l_global_irm = global_irm_penalty(
                global_logits, all_y, domain_ids, self.num_domains
            )
        else:
            l_global_expert = logits.sum() * 0.0
            l_global_irm = logits.sum() * 0.0
        l_sp = loss_sparse(dense_pi)
        l_bal = loss_balance(dense_pi)
        l_div = loss_diversity(h_stack) if self.lambda_div != 0 else logits.sum() * 0.0

        step = int(self.subset_step.item())
        sparse_phase = step >= self.topk_warmup_steps
        lambda_sp = (self.post_warmup_lambda_sp
                     if sparse_phase else self.lambda_sp)
        lambda_bal = (self.post_warmup_lambda_bal
                      if sparse_phase else self.lambda_bal)
        if step < self.route_anneal_steps:
            lambda_route = 0.0
        elif self.route_ramp_steps > 0:
            lambda_route = self.lambda_route * min(
                1.0,
                max(0.0, (step - self.route_anneal_steps) /
                    self.route_ramp_steps),
            )
        else:
            lambda_route = self.lambda_route
        lambda_q_capacity = q_capacity_lambda_at_step(
            step,
            self.lambda_q_capacity,
            anneal_steps=self.q_capacity_anneal_steps,
            ramp_steps=self.q_capacity_ramp_steps,
            post_step=self.q_capacity_post_step,
            post_lambda=self.q_capacity_post_lambda,
            decay_start_step=self.q_capacity_decay_start_step,
            decay_end_step=self.q_capacity_decay_end_step,
        )
        if step < self.sirm_anneal_steps:
            lambda_sirm = 0.0
        elif self.sirm_ramp_steps > 0:
            lambda_sirm = self.lambda_sirm * min(
                1.0,
                max(0.0, (step - self.sirm_anneal_steps) /
                    self.sirm_ramp_steps),
            )
        else:
            lambda_sirm = self.lambda_sirm
        if self.ssi_mode == "none" or step < self.ssi_anneal_steps:
            lambda_ssi = 0.0
        elif self.ssi_ramp_steps > 0:
            lambda_ssi = self.lambda_inv * min(
                1.0,
                max(0.0, (step - self.ssi_anneal_steps) /
                    self.ssi_ramp_steps),
            )
        else:
            lambda_ssi = self.lambda_inv
        if step < self.global_irm_anneal_steps:
            lambda_global_irm = 0.0
        elif self.global_irm_ramp_steps > 0:
            lambda_global_irm = self.lambda_global_irm * min(
                1.0,
                max(0.0, (step - self.global_irm_anneal_steps) /
                    self.global_irm_ramp_steps),
            )
        else:
            lambda_global_irm = self.lambda_global_irm
        if self.assignment_mode == "risk_ema" and step % self.probe_interval == 0:
            probe_risk, probe_valid = self._probe_risk(
                expert_logits, all_y, domain_ids
            )
            self.risk_ema.update(probe_risk, probe_valid)
        q = self._assignment(dense_pi, domain_ids)
        q_ready = bool(q.sum(dim=1).gt(0).all())
        l_q_capacity, q_capacity_info = q_capacity_loss(
            dense_pi, domain_ids, self.num_domains,
            self.q_capacity_rho_max,
        )
        if q_ready and int(self.q_active_step.item()) < 0:
            self.q_active_step.fill_(step)

        l_route = (router_distillation_loss(dense_pi, domain_ids, q)
                   if q_ready and lambda_route != 0.0
                   else logits.sum() * 0.0)
        sirm_allowed = (
            lambda_sirm != 0.0
            and q_ready
        )
        if sirm_allowed:
            l_sirm, sirm_info = subset_irm_penalty(
                expert_logits, all_y, domain_ids, gamma, q,
                min_effective_samples=self.min_effective_samples,
                detach_responsibility=self.responsibility_detach,
            )
        else:
            l_sirm = logits.sum() * 0.0
            sirm_info = {
                "ess": torch.zeros_like(q),
                "valid_cells": torch.zeros_like(q, dtype=torch.bool),
                "active_cells": torch.zeros_like(q, dtype=torch.bool),
                "active_domains_per_expert": torch.zeros(
                    self.num_experts, device=q.device, dtype=torch.long
                ),
                "active": False,
            }
        if sirm_info["active"]:
            self.sirm_active_steps += 1

        ssi_info = {
            "active": False,
            "active_pair_count": 0,
            "weight_sum": 0.0,
            "divergence_mean": 0.0,
            "divergence_min": 0.0,
            "divergence_max": 0.0,
            "marginal_error_max": 0.0,
            "negative_raw_pair_count": 0,
        }
        if lambda_ssi == 0.0:
            l_ssi = logits.sum() * 0.0
        elif self.ssi_mode == "mmd":
            l_ssi = loss_inv_MMD(
                h_stack, dense_pi, all_y, self.num_classes,
                domain_ids, self.num_domains,
                alpha=self.alpha, sigmas=self.sigmas,
            )
            ssi_info["active"] = True
        elif self.ssi_mode == "q_sinkhorn" and q_ready:
            l_ssi, ssi_info = q_gated_sinkhorn_divergence(
                h_stack, all_y, domain_ids, q, self.num_classes,
                epsilon=self.ssi_ot_epsilon,
                sinkhorn_iters=self.ssi_ot_iters,
                min_samples=self.ssi_ot_min_samples,
                normalize_features=self.ssi_ot_normalize_features,
            )
        else:
            l_ssi = logits.sum() * 0.0

        subset_total = (
            l_cls
            + self.lambda_expert * l_expert
            + lambda_route * l_route
            + lambda_q_capacity * l_q_capacity
            + lambda_sirm * l_sirm
            + lambda_ssi * l_ssi
            + lambda_sp * l_sp
            + lambda_bal * l_bal
            + self.lambda_div * l_div
        )
        global_total = (
            self.lambda_global_expert * l_global_expert
            + lambda_global_irm * l_global_irm
        )
        total = subset_total + global_total
        weighted_losses = OrderedDict([
            ("cls", l_cls),
            ("expert", self.lambda_expert * l_expert),
            ("global_expert", self.lambda_global_expert * l_global_expert),
            ("global_irm", lambda_global_irm * l_global_irm),
            ("route", lambda_route * l_route),
            ("q_capacity", lambda_q_capacity * l_q_capacity),
            ("sirm", lambda_sirm * l_sirm),
            ("ssi", lambda_ssi * l_ssi),
        ])
        gradient_metrics = {}
        if self.gradient_log_interval > 0 and step % self.gradient_log_interval == 0:
            gradient_metrics = self._gradient_diagnostics(weighted_losses)
            gradient_metrics.update(self._gradient_diagnostics(OrderedDict([
                ("cls", l_cls),
                ("expert", l_expert),
                ("global_expert", l_global_expert),
                ("global_irm", l_global_irm),
                ("route", l_route),
                ("q_capacity", l_q_capacity),
                ("sirm", l_sirm),
                ("ssi", l_ssi),
            ]), prefix="grad_raw"))

        risk, ess, valid, support = expert_domain_risk(
            expert_logits, all_y, domain_ids, gamma, self.num_domains,
            min_effective_samples=self.min_effective_samples,
            detach_responsibility=self.responsibility_detach,
        )
        entropy = -(dense_pi * dense_pi.clamp_min(EPS).log()).sum(dim=1).mean()
        load = gamma.mean(dim=0)
        selected = gamma.gt(0)
        expert_acc = []
        for expert in range(self.num_experts):
            mask = selected[:, expert]
            expert_acc.append(float(
                expert_logits[mask, expert].argmax(dim=1).eq(all_y[mask]).float().mean().item()
            ) if bool(mask.any()) else 0.0)
        representation_info = None
        similarity_metrics = {}
        if (self.global_expert_enabled and self.similarity_log_interval > 0
                and step % self.similarity_log_interval == 0):
            representation_info = expert_representation_diagnostics(
                h_stack, expert_logits, global_features, global_logits, gamma
            )
            subset_cka = representation_info["feature_linear_cka_matrix"][1:, 1:]
            off_diagonal = ~torch.eye(
                self.num_experts, device=subset_cka.device, dtype=torch.bool
            )
            active = representation_info["subset_selected_count"].gt(0)
            active_pairs = active.unsqueeze(0) & active.unsqueeze(1) & off_diagonal
            similarity_metrics = {
                "similarity_global_subset_cka_mean": float(
                    representation_info["feature_linear_cka_matrix"][0, 1:].mean().item()
                ),
                "similarity_subset_cka_mean": float(
                    subset_cka[off_diagonal].mean().item()
                ),
                "similarity_active_subset_cka_mean": float(
                    subset_cka[active_pairs].mean().item()
                ) if bool(active_pairs.any()) else 0.0,
                "similarity_global_subset_prediction_agreement": float(
                    representation_info["prediction_agreement_matrix"][0, 1:].mean().item()
                ),
                "similarity_min_feature_variance": float(
                    representation_info["feature_variance"].min().item()
                ),
            }

        self.optimizer.zero_grad(set_to_none=True)
        if self.global_training_mode == "separate":
            self.global_optimizer.zero_grad(set_to_none=True)
        total.backward()
        if self.global_training_mode == "separate":
            global_parameters = chain(
                self.global_featurizer.parameters(),
                self.global_input_proj.parameters(),
                self.global_expert.parameters(), self.global_head.parameters(),
            )
        elif self.global_expert_enabled:
            global_parameters = chain(
                self.global_expert.parameters(), self.global_head.parameters()
            )
        else:
            global_parameters = []
        optimized_parameters = list(chain(
            self.featurizer.parameters(), self.moe_head.parameters(),
            self.expert_heads.parameters(), global_parameters,
        ))
        total_grad_sq = total.new_zeros(())
        for parameter in optimized_parameters:
            if parameter.grad is not None:
                total_grad_sq += parameter.grad.detach().float().square().sum()
        total_grad_norm = float(total_grad_sq.sqrt().item())
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in optimized_parameters
        )
        if not bool(torch.isfinite(total)) or not finite_gradients:
            raise FloatingPointError(f"non-finite Subset-IRM update at step {step}")
        self.optimizer.step()
        if self.global_training_mode == "separate":
            self.global_optimizer.step()
        self.subset_step += 1

        active_cell_count = int(sirm_info["active_cells"].sum().item())
        self.last_diagnostics = {
            "risk_matrix": risk.detach().cpu().tolist(),
            "risk_valid_matrix": valid.detach().cpu().tolist(),
            "risk_support_matrix": support.detach().cpu().tolist(),
            "risk_ema_matrix": self.risk_ema.risk.detach().cpu().tolist(),
            "risk_ema_valid_matrix": self.risk_ema.valid.detach().cpu().tolist(),
            "q_matrix": q.detach().cpu().tolist(),
            "q_capacity_proxy_matrix": q_capacity_info["q_proxy"].cpu().tolist(),
            "q_capacity_mean_mass": q_capacity_info["mean_mass"].cpu().tolist(),
            "q_capacity_violation": q_capacity_info["violation"].cpu().tolist(),
            "q_capacity_max_mean_mass": q_capacity_info["max_mean_mass"],
            "q_capacity_max_violation": q_capacity_info["max_violation"],
            "q_capacity_violating_expert_count": q_capacity_info["violating_expert_count"],
            "ess_matrix": ess.detach().cpu().tolist(),
            "active_sirm_cells": sirm_info["active_cells"].cpu().tolist(),
            "active_domains_per_expert": sirm_info[
                "active_domains_per_expert"
            ].cpu().tolist(),
            "expert_load": load.detach().cpu().tolist(),
            "per_expert_accuracy": expert_acc,
            "ssi_mode": self.ssi_mode,
            "ssi_active": float(ssi_info["active"]),
            "ssi_active_pair_count": int(ssi_info["active_pair_count"]),
            "ssi_q_weight_sum": float(ssi_info["weight_sum"]),
            "ssi_divergence_mean": float(ssi_info["divergence_mean"]),
            "ssi_divergence_min": float(ssi_info["divergence_min"]),
            "ssi_divergence_max": float(ssi_info["divergence_max"]),
            "ssi_marginal_error_max": float(ssi_info["marginal_error_max"]),
            "ssi_negative_raw_pair_count": int(
                ssi_info["negative_raw_pair_count"]
            ),
            **gradient_metrics,
        }
        if representation_info is not None:
            self.last_diagnostics.update({
                f"expert_similarity_{key}": (
                    value.cpu().tolist() if torch.is_tensor(value) else value
                )
                for key, value in representation_info.items()
            })
        return {
            "loss": float(total.item()),
            "loss_cls": float(l_cls.item()),
            "loss_combined_eval": float(l_combined_eval.item()),
            "loss_expert": float(l_expert.item()),
            "loss_global_expert": float(l_global_expert.item()),
            "loss_global_irm": float(l_global_irm.item()),
            "loss_route": float(l_route.item()),
            "loss_q_capacity": float(l_q_capacity.item()),
            "loss_sirm": float(l_sirm.item()),
            "loss_ssi": float(l_ssi.item()),
            "loss_sp": float(l_sp.item()),
            "loss_bal": float(l_bal.item()),
            "loss_div": float(l_div.item()),
            "weighted_loss_cls": float(l_cls.item()),
            "weighted_loss_expert": float((self.lambda_expert * l_expert).item()),
            "weighted_loss_global_expert": float(
                (self.lambda_global_expert * l_global_expert).item()
            ),
            "weighted_loss_global_irm": float(
                (lambda_global_irm * l_global_irm).item()
            ),
            "weighted_loss_route": float((lambda_route * l_route).item()),
            "weighted_loss_q_capacity": float(
                (lambda_q_capacity * l_q_capacity).item()
            ),
            "weighted_loss_sirm": float((lambda_sirm * l_sirm).item()),
            "weighted_loss_ssi": float((lambda_ssi * l_ssi).item()),
            "weighted_loss_sp": float((lambda_sp * l_sp).item()),
            "weighted_loss_bal": float((lambda_bal * l_bal).item()),
            "effective_lambda_route": float(lambda_route),
            "effective_lambda_global_irm": float(lambda_global_irm),
            "effective_lambda_q_capacity": float(lambda_q_capacity),
            "effective_lambda_sirm": float(lambda_sirm),
            "effective_lambda_sp": float(lambda_sp),
            "effective_lambda_bal": float(lambda_bal),
            "effective_lambda_ssi": float(lambda_ssi),
            "train_acc": float(logits.argmax(dim=1).eq(all_y).float().mean().item()),
            "train_subset_only_acc": float(
                subset_logits.argmax(dim=1).eq(all_y).float().mean().item()
            ),
            "train_global_only_acc": float(
                global_logits.argmax(dim=1).eq(all_y).float().mean().item()
            ) if self.global_expert_enabled else 0.0,
            "routing_entropy": float(entropy.item()),
            "expert_load_mean": float(load.mean().item()),
            "expert_load_std": float(load.std(unbiased=False).item()),
            "dead_expert_count": int(load.eq(0).sum().item()),
            "active_experts_per_sample": float(selected.sum(dim=1).float().mean().item()),
            "active_sirm_cell_rate": float(active_cell_count / max(q.numel(), 1)),
            "sirm_active": float(sirm_info["active"]),
            "sirm_active_step_rate": float(
                self.sirm_active_steps.item() / max(step + 1, 1)
            ),
            "ess_mean": float(ess.mean().item()),
            "ess_min": float(ess.min().item()),
            "ess_max": float(ess.max().item()),
            "q_active_step": int(self.q_active_step.item()),
            "q_capacity_max_mean_mass": q_capacity_info["max_mean_mass"],
            "q_capacity_max_violation": q_capacity_info["max_violation"],
            "q_capacity_violating_expert_count": q_capacity_info["violating_expert_count"],
            "total_grad_norm": total_grad_norm,
            **similarity_metrics,
            **gradient_metrics,
        }
