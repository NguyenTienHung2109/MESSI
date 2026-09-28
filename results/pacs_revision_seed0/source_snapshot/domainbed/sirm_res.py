"""S-IRM-res: an anchored global classifier plus sparse residual experts.

This module is deliberately independent from ``MESSI_SubsetIRM``.  In
particular, the global expert is never a router option and never appears in
the Q assignment used by the residual experts.
"""

import hashlib
import warnings
from collections import OrderedDict
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed import algorithms
from domainbed.subset_irm import (
    EPS,
    ExpertClassifierHeads,
    SourceRiskEMA,
    _group_gradient_norm,
    expert_domain_risk,
    router_distillation_loss,
    top_r_assignment,
)


def sparse_router_with_floor(logits, top_k, q_min=0.0, temperature=1.0):
    """Top-k residual routing with an exact positive floor on active paths."""
    if logits.ndim != 2:
        raise ValueError("router logits must have shape [batch, residual_experts]")
    experts = logits.shape[1]
    top_k = int(top_k)
    if not 1 <= top_k <= experts:
        raise ValueError("sirm_res_router_topk must be in [1, residual experts]")
    if temperature <= 0:
        raise ValueError("sirm_res_router_temperature must be positive")
    if not 0.0 <= float(q_min) < 1.0 / top_k:
        raise ValueError("sirm_res_q_min must be in [0, 1/top_k)")
    selected_logits, selected_indices = torch.topk(logits, k=top_k, dim=1)
    selected_probabilities = F.softmax(selected_logits / float(temperature), dim=1)
    selected_weights = float(q_min) + (1.0 - top_k * float(q_min)) * selected_probabilities
    weights = torch.zeros_like(logits).scatter_(1, selected_indices, selected_weights)
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, selected_indices, True)
    return weights, mask


def proper_subset_assignment(risk, valid, top_r=2, temperature=1.0):
    """Risk Top-r Q with a best-effort proper-subset restriction.

    The legacy ``top_r_assignment`` can assign every source domain to one
    expert.  Here, if a column covers all domains, its weakest domain is moved
    to an eligible expert that is not already universal.  If no alternative
    exists (e.g. one residual expert), the row is retained and the condition
    is reported by ``proper_subset_mask`` diagnostics instead of silently
    pretending it holds.
    """
    q = top_r_assignment(risk, valid, top_r, temperature).clone()
    domains, experts = q.shape
    if domains <= 1:
        return q.detach()
    for expert in range(experts):
        support = q[:, expert].gt(0)
        if not bool(support.all()):
            continue
        # Remove the least competent assigned domain, then choose the best
        # candidate whose addition cannot make another column universal.
        assigned = torch.where(support)[0]
        remove_domain = assigned[risk[assigned, expert].argmax()]
        candidates = valid[remove_domain] & torch.isfinite(risk[remove_domain])
        candidates[expert] = False
        for candidate in torch.argsort(risk[remove_domain]):
            candidate = int(candidate.item())
            if not bool(candidates[candidate]):
                continue
            if bool(q[:, candidate].gt(0).sum() < domains - 1):
                q[remove_domain, expert] = 0.0
                q[remove_domain, candidate] = 1.0
                q[remove_domain] /= q[remove_domain].sum().clamp_min(EPS)
                break
    return q.detach()


def marginal_residual_sirm_penalty(global_logits, residual_logits, targets,
                                   domain_ids, responsibilities, assignment,
                                   residual_scale=1.0,
                                   min_effective_samples=2.0):
    """IRMv1 over each residual expert's *marginal* contribution.

    ``global_logits`` and all non-current residual paths are detached in the
    marginal risk.  Router weights are detached too, so this loss can only
    change the residual expert/head that is currently evaluated.
    """
    if residual_logits.ndim != 3:
        raise ValueError("residual_logits must have shape [batch, experts, classes]")
    gamma = responsibilities.detach().float()
    q = assignment.detach().float()
    domains, experts = q.shape
    if experts != residual_logits.shape[1]:
        raise ValueError("Q and residual logits expert dimensions must match")
    # Validity is a routing/support property.  The CE values returned here are
    # intentionally ignored: the marginal logits below define the S-IRM risk.
    _, ess, valid, _ = expert_domain_risk(
        residual_logits.detach(), targets, domain_ids, gamma, domains,
        min_effective_samples=min_effective_samples,
        detach_responsibility=False,
    )
    assigned_valid = valid & q.gt(0)
    active_domains = assigned_valid.sum(dim=0)
    proper = active_domains.le(max(domains - 1, 0))
    eligible = active_domains.ge(2) & proper
    mixed = (gamma.unsqueeze(-1) * residual_logits).sum(dim=1)
    numerator = residual_logits.sum() * 0.0
    denominator = q.new_zeros(())
    active_cells = torch.zeros_like(assigned_valid)
    for expert in range(experts):
        if not bool(eligible[expert]):
            continue
        contribution = gamma[:, expert].unsqueeze(1) * residual_logits[:, expert]
        base_without = (global_logits + float(residual_scale) * (mixed - contribution)).detach()
        for domain in range(domains):
            if not bool(assigned_valid[domain, expert]):
                continue
            mask = domain_ids.eq(domain)
            weights = gamma[mask, expert]
            if not bool(weights.sum() > EPS):
                continue
            scale = residual_logits.new_ones((), requires_grad=True)
            marginal_logits = base_without[mask] + (
                float(residual_scale) * scale * weights.unsqueeze(1)
                * residual_logits[mask, expert]
            )
            cell_ce = F.cross_entropy(marginal_logits.float(), targets[mask], reduction="none")
            risk = (weights * cell_ce).sum() / weights.sum().clamp_min(EPS)
            gradient = torch.autograd.grad(risk, scale, create_graph=True, retain_graph=True)[0]
            numerator = numerator + q[domain, expert] * gradient.square()
            denominator = denominator + q[domain, expert]
            active_cells[domain, expert] = True
    penalty = numerator / denominator.clamp_min(EPS)
    return penalty, {
        "ess": ess.detach(),
        "valid_cells": valid.detach(),
        "active_cells": active_cells.detach(),
        "active_domains_per_expert": active_domains.detach(),
        "proper_subset_per_expert": proper.detach(),
        "skipped_universal_experts": int(
            (active_domains.ge(domains) & (domains > 1)).sum().item()
        ),
        "active": bool(denominator.detach().item() > 0),
    }


def _domain_balanced_ce(logits, targets, domain_ids, num_domains):
    losses = []
    for domain in range(num_domains):
        mask = domain_ids.eq(domain)
        if bool(mask.any()):
            losses.append(F.cross_entropy(logits[mask].float(), targets[mask]))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def _module_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SIRMRes(algorithms.MESSI):
    """Two-stage global-anchor / residual-expert implementation of S-IRM-res."""

    is_sirm_res = True

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        hparams = dict(hparams)
        total = int(hparams.get("num_total_experts", hparams.get("num_experts", 6)))
        if total < 2:
            raise ValueError("S-IRM-res requires num_total_experts >= 2")
        hparams["num_experts"] = total
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.algorithm_name = "S-IRM-res"
        self.num_total_experts = total
        self.num_residual_experts = total - 1
        self.num_experts = self.num_residual_experts  # Q/router expert count
        self.global_steps = int(hparams.get("sirm_res_global_steps", 0))
        self.residual_steps = int(hparams.get("sirm_res_residual_steps", 0))
        if self.global_steps < 1 or self.residual_steps < 1:
            raise ValueError("S-IRM-res needs positive global and residual stage steps")
        self.residual_scale = float(hparams.get("sirm_res_residual_scale", 1.0))
        self.router_topk = int(hparams.get("sirm_res_router_topk", 2))
        self.q_min = float(hparams.get("sirm_res_q_min", 0.2))
        self.router_temperature = float(hparams.get("sirm_res_router_temperature", 1.0))
        if self.router_topk > self.num_residual_experts:
            raise ValueError("sirm_res_router_topk cannot exceed residual expert count")
        if not 0.0 <= self.q_min < 1.0 / self.router_topk:
            raise ValueError("sirm_res_q_min must be in [0, 1/router_topk)")
        self.lambda_router = float(hparams.get("sirm_res_lambda_router", 0.0))
        self.lambda_sirm = float(hparams.get("sirm_res_lambda_sirm", 0.0))
        self.assignment_topr = int(hparams.get("sirm_res_assignment_topr", 2))
        self.assignment_temperature = float(hparams.get("sirm_res_assignment_temperature", 1.0))
        self.min_effective_samples = float(hparams.get("sirm_res_min_effective_samples", 2.0))
        if int(num_domains) <= 2:
            warnings.warn(
                "S-IRM-res has fewer than three source domains: no residual "
                "expert can have a proper subset with at least two domains; "
                "the S-IRM-res penalty will remain inactive.", RuntimeWarning,
            )

        # Re-home the base head parts, then discard the shared-softmax MoE
        # container.  This makes parameter ownership explicit in checkpoints.
        old_head = self.moe_head
        self.input_proj = old_head.input_proj
        self.global_expert = old_head.experts[0]
        self.global_head = old_head.classifier
        self.residual_experts = nn.ModuleList(list(old_head.experts[1:]))
        self.residual_heads = ExpertClassifierHeads(
            self.num_residual_experts, old_head.expert_dim, num_classes
        ).to(next(old_head.parameters()).device)
        with torch.no_grad():
            for head in self.residual_heads.heads:
                head.weight.zero_()
                if head.bias is not None:
                    head.bias.zero_()
        self.residual_router = nn.Linear(old_head.expert_dim, self.num_residual_experts).to(next(old_head.parameters()).device)
        del self.moe_head

        self.risk_ema = SourceRiskEMA(
            num_domains, self.num_residual_experts,
            beta=float(hparams.get("sirm_res_assignment_ema", 0.9)),
        ).to(next(self.global_head.parameters()).device)
        self.register_buffer("phase_code", torch.zeros((), dtype=torch.long))
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("residual_step", torch.zeros((), dtype=torch.long))
        self.global_anchor_hash = ""
        self.global_optimizer = self._make_optimizer(self._global_parameters())
        self.residual_optimizer = None
        self.optimizer = self.global_optimizer
        self.last_diagnostics = {}

    @property
    def phase(self):
        return "global" if int(self.phase_code.item()) == 0 else "residual"

    def _make_optimizer(self, parameters):
        params = [p for p in parameters if p.requires_grad]
        return torch.optim.Adam(params, lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])

    def _global_parameters(self):
        return chain(self.featurizer.parameters(), self.input_proj.parameters(),
                     self.global_expert.parameters(), self.global_head.parameters())

    def _residual_parameters(self):
        return chain(self.residual_router.parameters(), self.residual_experts.parameters(),
                     self.residual_heads.parameters())

    def _anchor_module(self):
        return nn.ModuleDict({
            "backbone": self.featurizer, "input_proj": self.input_proj,
            "global_expert": self.global_expert, "global_head": self.global_head,
        })

    def anchor_hash(self):
        return _module_hash(self._anchor_module())

    def enter_residual_stage(self):
        if self.phase == "residual" and self.residual_optimizer is not None:
            return
        if not self.global_anchor_hash:
            self.global_anchor_hash = self.anchor_hash()
        for module in (self.featurizer, self.input_proj, self.global_expert, self.global_head):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for module in (self.residual_router, self.residual_experts, self.residual_heads):
            module.train()
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.residual_optimizer = self._make_optimizer(self._residual_parameters())
        self.optimizer = self.residual_optimizer
        self.phase_code.fill_(1)

    def train(self, mode=True):
        super().train(mode)
        if self.phase == "residual":
            for module in (self.featurizer, self.input_proj, self.global_expert, self.global_head):
                module.eval()
            if mode:
                for module in (self.residual_router, self.residual_experts, self.residual_heads):
                    module.train()
        return self

    def _global_forward(self, x, frozen=False):
        context = torch.no_grad() if frozen else torch.enable_grad()
        with context:
            features = self.featurizer(x)
            projected = self.input_proj(features)
            global_features = self.global_expert(projected)
            global_logits = self.global_head(global_features)
        return projected, global_features, global_logits

    def forward_components(self, x, remove_residual=None, global_only=False):
        frozen = self.phase == "residual"
        projected, global_features, global_logits = self._global_forward(x, frozen=frozen)
        if global_only or self.phase == "global":
            return {"final_logits": global_logits, "global_logits": global_logits,
                    "residual_logits": None, "routing": None, "global_features": global_features,
                    "residual_features": None}
        projected = projected.detach()
        residual_features = torch.stack([expert(projected) for expert in self.residual_experts], dim=1)
        residual_logits = self.residual_heads(residual_features)
        router_logits = self.residual_router(projected)
        routing, selected = sparse_router_with_floor(router_logits, self.router_topk, self.q_min, self.router_temperature)
        if remove_residual is not None:
            routing = routing.clone()
            routing[:, int(remove_residual)] = 0.0  # intentionally no renormalization
        residual_mix = (routing.unsqueeze(-1) * residual_logits).sum(dim=1)
        return {"final_logits": global_logits + self.residual_scale * residual_mix,
                "global_logits": global_logits, "residual_logits": residual_logits,
                "routing": routing, "selected": selected, "router_logits": router_logits,
                "global_features": global_features, "residual_features": residual_features}

    def predict(self, x):
        return self.forward_components(x)["final_logits"]

    def predict_mode(self, x, mode="full", remove_residual=None):
        if mode in ("global", "global_only", "disable_all_residual"):
            return self.forward_components(x, global_only=True)["final_logits"]
        if mode == "remove_residual":
            return self.forward_components(x, remove_residual=remove_residual)["final_logits"]
        if mode != "full":
            raise ValueError("mode must be full, global_only, or remove_residual")
        return self.predict(x)

    def _assignment(self):
        if not self.risk_ema.ready:
            return torch.zeros_like(self.risk_ema.risk)
        return proper_subset_assignment(self.risk_ema.risk, self.risk_ema.valid,
                                        self.assignment_topr, self.assignment_temperature)

    def _update_global(self, all_x, all_y, domain_ids):
        _, _, logits = self._global_forward(all_x, frozen=False)
        loss = _domain_balanced_ce(logits, all_y, domain_ids, self.num_domains)
        self.global_optimizer.zero_grad()
        loss.backward()
        self.global_optimizer.step()
        self.global_step += 1
        return {"loss": float(loss.item()), "loss_global_ce": float(loss.item()),
                "phase": "global", "global_step": int(self.global_step.item()),
                "residual_step": int(self.residual_step.item())}

    def _update_residual(self, all_x, all_y, domain_ids):
        values = self.forward_components(all_x)
        final_logits = values["final_logits"]
        global_logits = values["global_logits"]
        residual_logits = values["residual_logits"]
        routing = values["routing"]
        dense_probabilities = F.softmax(values["router_logits"] / self.router_temperature, dim=1)
        competence_logits = global_logits.detach().unsqueeze(1) + self.residual_scale * residual_logits.detach()
        with torch.no_grad():
            risk, _, valid, _ = expert_domain_risk(
                competence_logits, all_y, domain_ids, torch.ones_like(routing),
                self.num_domains, min_effective_samples=1.0,
            )
            self.risk_ema.update(risk, valid)
        q = self._assignment()
        q_ready = bool(q.sum(dim=1).gt(0).all())
        l_cls = F.cross_entropy(final_logits.float(), all_y)
        l_router = router_distillation_loss(dense_probabilities, domain_ids, q) if q_ready else final_logits.sum() * 0.0
        l_sirm, sirm_info = marginal_residual_sirm_penalty(
            global_logits, residual_logits, all_y, domain_ids, routing, q,
            self.residual_scale, self.min_effective_samples,
        ) if q_ready else (final_logits.sum() * 0.0, {"active": False, "active_cells": torch.zeros_like(q, dtype=torch.bool), "active_domains_per_expert": torch.zeros(self.num_residual_experts, device=q.device), "proper_subset_per_expert": torch.zeros(self.num_residual_experts, dtype=torch.bool, device=q.device), "skipped_universal_experts": 0})
        total = l_cls + self.lambda_router * l_router + self.lambda_sirm * l_sirm
        grad = OrderedDict([
            ("backbone", list(self.featurizer.parameters())),
            ("global", list(chain(self.global_expert.parameters(), self.global_head.parameters()))),
            ("router", list(self.residual_router.parameters())),
            ("residual", list(chain(self.residual_experts.parameters(), self.residual_heads.parameters()))),
        ])
        grad_norms = {f"grad_total_{name}": _group_gradient_norm(total, params) for name, params in grad.items()}
        self.residual_optimizer.zero_grad()
        total.backward()
        self.residual_optimizer.step()
        self.residual_step += 1
        after_hash = self.anchor_hash()
        info = {"loss": float(total.item()), "loss_cls": float(l_cls.item()),
                "loss_router": float(l_router.item()), "loss_sirm_res": float(l_sirm.item()),
                "phase": "residual", "global_step": int(self.global_step.item()),
                "residual_step": int(self.residual_step.item()), "q_ready": q_ready,
                "q": q.detach(), "routing_min_active": float(routing[routing.gt(0)].min().item()),
                "routing_max_active": float(routing.max().item()),
                "routing_sum_error": float((routing.sum(1) - 1).abs().max().item()),
                "anchor_hash_before": self.global_anchor_hash, "anchor_hash_after": after_hash,
                "anchor_unchanged": after_hash == self.global_anchor_hash,
                "sirm_active": sirm_info["active"],
                "sirm_active_cells": int(sirm_info["active_cells"].sum().item()),
                "sirm_skipped_universal_experts": sirm_info["skipped_universal_experts"]}
        info.update(grad_norms)
        return info

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])
        domain_ids = self._get_domain_ids(minibatches)
        info = self._update_global(all_x, all_y, domain_ids) if self.phase == "global" else self._update_residual(all_x, all_y, domain_ids)
        self.last_diagnostics = info
        return info
