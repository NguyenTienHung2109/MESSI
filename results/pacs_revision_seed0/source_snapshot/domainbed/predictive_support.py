"""Predictive binary subset supports; no target labels enter structure learning."""
import itertools
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


def feasible_supports(class_presence, experts, budget, max_candidates=200000):
    """Exact enumeration, counting only pairs with a population shared class.

    Refuse oversized problems instead of silently claiming an exact solution.
    """
    presence = torch.as_tensor(class_presence, dtype=torch.bool, device='cpu')
    if presence.ndim != 2 or len(presence) > 6:
        raise ValueError('exact search supports a [domains, classes] table with <=6 domains')
    domains = len(presence)
    pairs = [(i, j) for i in range(domains) for j in range(i + 1, domains)
             if bool((presence[i] & presence[j]).any())]
    if budget <= 0 or experts < 1:
        raise ValueError('positive pair budget and expert count required')
    options = []
    for bits in itertools.product((False, True), repeat=domains):
        cost = sum(bits[i] and bits[j] for i, j in pairs)
        if sum(bits) == 0 or sum(bits) >= 2:
            options.append((bits, cost))
    results = []
    visits = 0
    def visit(columns, spent):
        nonlocal visits
        visits += 1
        if visits > max_candidates * 100:
            raise ValueError('exact search too large; reduce domains/experts/budget')
        if len(columns) == experts:
            if spent == budget and all(any(col[k] for col in columns) for k in range(domains)):
                results.append(columns)
                if len(results) > max_candidates:
                    raise ValueError('too many feasible supports for exact search')
            return
        for bits, cost in options:
            if spent + cost <= budget:
                visit(columns + [bits], spent + cost)
    visit([], 0)
    if not results:
        raise ValueError('no feasible support for class presence and pair budget')
    return torch.tensor(results, dtype=torch.bool).transpose(1, 2), pairs


def mmd_unbiased(x, y, bandwidth):
    """Gaussian U-statistic; may be negative. Never clamp this estimator."""
    if min(len(x), len(y)) < 2 or bandwidth <= 0:
        raise ValueError('MMD needs >=2 independent samples per cell and positive bandwidth')
    def kernel(a, b):
        return torch.exp(-torch.cdist(a.float(), b.float()).square() / (2 * bandwidth ** 2))
    xx, yy, xy = kernel(x, x), kernel(y, y), kernel(x, y)
    return ((xx.sum() - xx.diagonal().sum()) / (len(x) * (len(x) - 1))
            + (yy.sum() - yy.diagonal().sum()) / (len(y) * (len(y) - 1)) - 2 * xy.mean())


def predictive_terms(expert_logits, router_logits, targets, support):
    """Per-example losses; support has shape [batch, experts]."""
    ce = F.cross_entropy(expert_logits.flatten(0, 1),
                         targets[:, None].expand(expert_logits.shape[:2]).flatten(),
                         reduction='none').view(expert_logits.shape[:2])
    local = ce.masked_fill(~support, -torch.inf).max(1).values
    log_pi = router_logits.log_softmax(1)
    adm = -log_pi.masked_fill(~support, -torch.inf).logsumexp(1)
    return local, adm, ce


def information_constraint(domain_risks, supports, class_probabilities, margin):
    """Support-conditional risk/entropy under uniform source-domain Q.

    supports can be [K,M] or [candidates,K,M]. No router values enter here.
    The hinge is summed over active experts, after averaging risks over each
    expert's domains. Entropy is of the mixed label law, not mean cell entropy.
    A zero empirical hinge is not a population mutual-information certificate.
    """
    weights = supports.to(domain_risks.dtype)
    sizes = weights.sum(-2)
    active = sizes > 0
    risks = (weights * domain_risks).sum(-2) / sizes.clamp_min(1)
    probabilities = torch.einsum('...km,kc->...mc', weights, class_probabilities)
    probabilities = probabilities / sizes.clamp_min(1).unsqueeze(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
    thresholds = entropy - margin
    violations = F.relu(risks - thresholds) * active
    return violations.sum(-1), {
        'risk': risks, 'entropy': entropy, 'threshold': thresholds,
        'information_lower_bound_estimate': entropy - risks,
        'violation': violations, 'active': active,
    }


class PredictiveSupport(nn.Module):
    """Dense MoE with shared norm-bounded classifier and alternating exact S step.

    The dedicated runner supplies population class presence and independent
    stratified alignment batches; ordinary sparse minibatches are insufficient.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, featurizer=None):
        super().__init__()
        from domainbed.deit_transformer import ExplicitMoEHead, DeiTFeaturizer
        self.hparams = dict(hparams)
        self.num_domains, self.num_classes = num_domains, num_classes
        self.num_experts = int(hparams.get('num_experts', 6))
        self.scale = float(hparams.get('support_classifier_norm', 5.0))
        self.bandwidth = float(hparams.get('support_bandwidth', 1.0))
        self.lambda_sub = float(hparams.get('support_lambda_sub', 1.0))
        self.warmup = int(hparams.get('support_warmup_steps', 100))
        if self.scale <= 0 or self.bandwidth <= 0 or self.lambda_sub < 0 or self.warmup < 0:
            raise ValueError('invalid support objective hyperparameters')
        self.lmax = math.log(num_classes) + 2 * self.scale
        self.predictive_objective = hparams.get('support_predictive_objective', 'max_ce')
        if self.predictive_objective not in ('max_ce', 'information_hinge'):
            raise ValueError('unknown support predictive objective')
        self.information_margin = float(hparams.get('support_information_margin', 0.1))
        self.lambda_pred = float(hparams.get('support_lambda_pred', 1.0))
        self.lambda_adm = float(hparams.get('support_lambda_adm', self.lmax))
        self.structure_router_weight = float(hparams.get('support_structure_router_weight', self.lmax))
        self.adm_ramp_steps = int(hparams.get('support_adm_ramp_steps', 0))
        if (not 0 < self.information_margin < math.log(num_classes) or
                min(self.lambda_pred, self.lambda_adm, self.structure_router_weight, self.adm_ramp_steps) < 0):
            raise ValueError('invalid information margin or loss weights')
        if self.predictive_objective == 'information_hinge' and self.structure_router_weight != 0:
            raise ValueError('information_hinge requires router-independent structure search')
        self.featurizer = featurizer if featurizer is not None else DeiTFeaturizer(
            pretrained=hparams.get('pretrained', True),
            model_name=hparams.get('model', 'deit_small_patch16_224'),
            img_size=input_shape[1], in_chans=input_shape[0],
            patch_size=hparams.get('patch_size', 16))
        self.moe_head = ExplicitMoEHead(self.featurizer.n_outputs, self.num_experts,
                                       num_classes, hparams.get('expert_mlp_ratio', 4),
                                       hparams.get('moe_dim', None))
        self.moe_head.classifier = nn.Linear(self.moe_head.expert_dim, num_classes, bias=False)
        self.register_buffer('support', torch.zeros(num_domains, self.num_experts, dtype=torch.bool))
        self.register_buffer('step', torch.zeros((), dtype=torch.long))
        self.register_buffer('structure_ready', torch.tensor(False))
        self.register_buffer('class_presence', torch.zeros(num_domains, num_classes, dtype=torch.bool))
        # Only the new objective adds a persistent buffer, preserving old checkpoints.
        self.register_buffer('source_class_probabilities',
                             torch.zeros(num_domains, num_classes),
                             persistent=self.predictive_objective == 'information_hinge')
        self.budget = int(hparams.get('support_pair_budget', math.comb(num_domains, 2)))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=hparams.get('lr', 3e-5),
                                          weight_decay=hparams.get('weight_decay', 1e-6))
        self.project_classifier()

    @torch.no_grad()
    def project_classifier(self):
        w = self.moe_head.classifier.weight
        w.div_((w.norm(dim=1, keepdim=True) / self.scale).clamp_min(1))

    def components(self, x):
        h = self.moe_head
        v = h.input_proj(self.featurizer(x))
        z = F.normalize(torch.stack([e(v) for e in h.experts], 1), dim=-1, eps=1e-8)
        router = h.router(v)
        logits = h.classifier(z)
        # Do not normalize the mixture: Jensen requires this linear combination.
        mixed = (router.softmax(1).unsqueeze(-1) * logits).sum(1)
        return mixed, logits, router, z

    def predict(self, x):
        return self.components(x)[0]

    def configure_support(self, presence, class_counts=None):
        presence = torch.as_tensor(presence, dtype=torch.bool, device=self.support.device)
        if presence.shape != self.class_presence.shape:
            raise ValueError('class presence shape must match domains and classes')
        self.class_presence.copy_(presence)
        candidates, _ = feasible_supports(self.class_presence, self.num_experts, self.budget)
        self.support.copy_(candidates[0])
        if class_counts is not None:
            counts = torch.as_tensor(class_counts, device=self.support.device, dtype=torch.float32)
            if (counts.shape != presence.shape or not torch.isfinite(counts).all()
                    or (counts < 0).any() or (counts.sum(1) <= 0).any()
                    or not torch.equal(counts > 0, presence)):
                raise ValueError('class counts must describe the same nonempty source-training cells')
            self.source_class_probabilities.copy_(counts / counts.sum(1, keepdim=True))
        elif self.predictive_objective == 'information_hinge':
            raise ValueError('information_hinge requires full source-training class counts')

    def predictive_constraint(self, domain_risks, supports=None):
        if not bool(self.source_class_probabilities.sum(1).gt(0).all()):
            raise RuntimeError('source-training class probabilities are not initialized')
        return information_constraint(domain_risks, self.support if supports is None else supports,
                                      self.source_class_probabilities, self.information_margin)

    @torch.no_grad()
    def update_structure(self, prediction_batches, cells, mode='learned'):
        """Score fixed neural parameters using fresh source-only observations.

        prediction_batches sample the source reference law uniformly per domain;
        cells contain >=2 independently drawn images for every cell participating in a valid pair.
        """
        if len(prediction_batches) != self.num_domains or any(len(y) == 0 for _, y in prediction_batches):
            raise ValueError('structure scoring requires observations from every source domain')
        shared_presence = self.class_presence & (self.class_presence.sum(0) >= 2)
        expected = set(map(tuple, shared_presence.nonzero().cpu().tolist()))
        if set(cells) != expected or any(len(x) < 2 for x in cells.values()):
            raise ValueError('structure scoring needs >=2 samples in every present source cell')
        start = time.perf_counter()
        was_training = self.training
        self.eval()
        try:
            candidates, pairs = feasible_supports(self.class_presence, self.num_experts, self.budget)
            candidates = candidates.to(self.support.device)
            encoded = {key: self.components(x)[3] for key, x in cells.items()}
            discrepancies = []
            class_discrepancies = []
            for i, j in pairs:
                classes = torch.where(self.class_presence[i] & self.class_presence[j])[0].tolist()
                per_class_mmd = torch.stack([
                    torch.stack([mmd_unbiased(encoded[i, c][:, m], encoded[j, c][:, m], self.bandwidth)
                                 for m in range(self.num_experts)]) for c in classes])
                discrepancies.append(per_class_mmd.mean(0))
                class_discrepancies.append({'pair': [i,j], 'classes': classes,
                                            'mmd_by_class_expert': per_class_mmd.cpu().tolist()})
            d = torch.stack(discrepancies)
            pair_mask = torch.stack([candidates[:, i] & candidates[:, j] for i, j in pairs], 1)
            sub = (pair_mask * d).sum((1, 2)) / self.budget
            scores = torch.zeros(len(candidates), device=d.device)
            local_scores = torch.zeros_like(scores)
            adm_scores = torch.zeros_like(scores)
            domain_risks = []
            # Evaluate all candidates in bounded chunks, retaining exact per-sample max.
            for k, (x, y) in enumerate(prediction_batches):
                _, logits, router, _ = self.components(x)
                _, _, ce = predictive_terms(logits, router, y, torch.ones_like(router, dtype=torch.bool))
                log_pi = router.log_softmax(1)
                domain_risks.append(ce.mean(0))
                for offset in range(0, len(candidates), 128):
                    mask = candidates[offset:offset + 128, k, None, :]
                    loc = ce[None].masked_fill(~mask, -torch.inf).max(2).values.mean(1)
                    adm = -log_pi[None].masked_fill(~mask, -torch.inf).logsumexp(2).mean(1)
                    local_scores[offset:offset + 128] += loc / self.num_domains
                    adm_scores[offset:offset + 128] += adm / self.num_domains
            constraint_info = None
            if self.predictive_objective == 'information_hinge':
                pred_scores, constraint_info = self.predictive_constraint(torch.stack(domain_risks), candidates)
            else:
                pred_scores = local_scores
            # Do not even add a zero times router term: nonfinite router
            # diagnostics must never contaminate a router-independent score.
            scores = self.lambda_pred * pred_scores
            if self.structure_router_weight > 0:
                scores = scores + self.structure_router_weight * adm_scores
            scores_without_discrepancy = scores.clone()
            if mode not in ('learned', 'no_discrepancy', 'fixed', 'random'):
                raise ValueError('unknown support intervention')
            if mode != 'no_discrepancy':
                scores += self.lambda_sub * sub
            index = int(scores.argmin())
            if mode == 'random':
                index = int(torch.randint(len(candidates), ()).item())
            if mode == 'fixed':
                index = int((candidates == self.support).all(2).all(1).nonzero()[0])
            previous = self.support.clone()
            old_index = int((candidates == previous).all(2).all(1).nonzero()[0])
            order = scores.argsort()
            top = order[:min(10, len(order))]
            runner_up = int(order[1]) if len(order) > 1 else index
            best_index = int(order[0])
            discrepancy_weight = self.lambda_sub if mode != 'no_discrepancy' else 0.0
            structure_eda = {
                'predictive_objective': self.predictive_objective,
                'structure_router_weight': self.structure_router_weight,
                'gap_predictive': float(self.lambda_pred * (pred_scores[runner_up] - pred_scores[best_index])),
                'gap_mmd': float(discrepancy_weight * (sub[runner_up] - sub[best_index])),
                'gap_router': (float(self.structure_router_weight * (adm_scores[runner_up] - adm_scores[best_index]))
                               if self.structure_router_weight > 0 else 0.0),
                'selected_score_predictive': float(pred_scores[index]),
                'no_discrepancy_optimum_support': candidates[scores_without_discrepancy.argmin()].cpu().tolist(),
                'selection_differs_without_discrepancy': bool((candidates[index] != candidates[scores_without_discrepancy.argmin()]).any()),
                'empirical_best_tie_count_at_1e8': int((scores - scores[best_index]).abs().le(1e-8).sum()),
                'constraint': ({k: v[index].cpu().tolist() for k, v in constraint_info.items()}
                               if constraint_info is not None else None),
                'support_changed_bits': int((previous != candidates[index]).sum()),
                'support_sizes': candidates[index].sum(0).tolist(),
                'domain_coverage': candidates[index].sum(1).tolist(),
                'active_experts': int(candidates[index].any(0).sum()),
                'previous_support_score': float(scores[old_index]),
                'structure_improvement_same_probe': float(scores[old_index] - scores[index]),
                'empirical_runner_up_gap': float(scores[order[1]] - scores[order[0]]) if len(order)>1 else None,
                'pair_indices': pairs, 'pair_expert_mmd': d.cpu().tolist(),
                'class_conditional_mmd': class_discrepancies,
                'selected_pair_mask': pair_mask[index].cpu().tolist(),
                'negative_mmd_fraction': float(d.lt(0).float().mean()),
                'top_candidates': [{'support': candidates[t].cpu().tolist(),
                    'score': float(scores[t]), 'local': float(local_scores[t]),
                    'predictive': float(pred_scores[t]),
                    'adm': float(adm_scores[t]), 'mmd': float(sub[t])} for t in top],
                'selected_score_local': float(local_scores[index]),
                'selected_score_adm': float(adm_scores[index]),
                'selected_score_mmd': float(sub[index])}
            self.support.copy_(candidates[index])
            self.structure_ready.fill_(True)
            return {'structure_score': scores[index].item(), 'structure_candidates': len(candidates),
                    'structure_seconds': time.perf_counter() - start,
                    'support': self.support.cpu().tolist(), 'structure_mode': mode,
                    'structure_eda': structure_eda}
        finally:
            self.train(was_training)

    def update(self, minibatches, unlabeled=None, alignment_batches=None):
        if len(minibatches) != self.num_domains or any(len(y) == 0 for _, y in minibatches):
            raise ValueError('neural updates require nonempty batches from every source domain')
        warm = int(self.step) < self.warmup
        if not warm and not bool(self.structure_ready):
            raise RuntimeError('run update_structure before post-warmup neural updates')
        locals_, adms, mixes, masses, warm_losses = [], [], [], [], []
        worst_counts = []
        domain_risks = []
        for k, (x, y) in enumerate(minibatches):
            mix, logits, router, _ = self.components(x)
            s = self.support[k].expand(len(x), -1)
            loc, adm, ce = predictive_terms(logits, router, y, s)
            worst = ce.detach().masked_fill(~s, -torch.inf).argmax(1)
            worst_counts.append(torch.bincount(worst, minlength=self.num_experts).div(len(y)).tolist())
            locals_.append(loc.mean())
            warm_losses.append(ce.mean())
            domain_risks.append(ce.mean(0))
            adms.append(adm.mean())
            mixes.append(F.cross_entropy(mix, y))
            masses.append((1 - (router.softmax(1) * s).sum(1)).mean())
        loc, adm, mix = [torch.stack(v).mean() for v in (locals_, adms, mixes)]
        sub = loc * 0
        if not warm and self.lambda_sub > 0:
            if not alignment_batches:
                raise ValueError('post-warmup updates need sampled expert/pair/class cell batches')
            terms = []
            for m, i, j, c, x, y in alignment_batches:
                if not (self.support[i, m] and self.support[j, m] and
                        self.class_presence[i, c] and self.class_presence[j, c]):
                    raise ValueError('alignment outside selected valid support')
                terms.append(mmd_unbiased(self.components(x)[3][:, m],
                                          self.components(y)[3][:, m], self.bandwidth))
            # Uniform selected pairs, then uniform shared class: unbiased for /B objective.
            sub = torch.stack(terms).mean()
        adm_coefficient = 0.0 if warm else self.lambda_adm
        if not warm and self.adm_ramp_steps > 0:
            adm_coefficient *= min(1., (int(self.step) - self.warmup + 1) / self.adm_ramp_steps)
        constraint_info = None
        if self.predictive_objective == 'information_hinge':
            # Warm-up is mixture CE plus a global predictive threshold for all
            # experts. It trains the router as a predictor, without inventing S.
            constraint_support = torch.ones_like(self.support) if warm else self.support
            pred, constraint_info = self.predictive_constraint(torch.stack(domain_risks), constraint_support)
            loss = mix + self.lambda_pred * pred + adm_coefficient * adm + self.lambda_sub * sub
        else:
            pred = loc
            loss = torch.stack(warm_losses).mean() if warm else self.lambda_pred * loc + adm_coefficient * adm + self.lambda_sub * sub
        self.optimizer.zero_grad()
        loss.backward()
        diagnostics = {}
        if constraint_info is not None:
            diagnostics['predictive_constraint'] = {k: v.detach().cpu().tolist() for k, v in constraint_info.items()}
        if int(self.step) % int(self.hparams.get('support_gradient_interval', 100)) == 0:
            groups = {'backbone': self.featurizer, 'router': self.moe_head.router,
                      'classifier': self.moe_head.classifier,
                      **{f'expert_{m}': e for m, e in enumerate(self.moe_head.experts)}}
            diagnostics['gradient_norms'] = {name: float(torch.stack([
                p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None
            ]).sum().sqrt()) if any(p.grad is not None for p in module.parameters()) else 0.
                for name, module in groups.items()}
        self.optimizer.step()
        self.project_classifier()
        self.step.add_(1)
        return {'loss': loss.item(), 'local_risk': loc.item(), 'admissibility': adm.item(),
                'mmd_u': sub.item(), 'mixture_ce': mix.item(),
                'prediction_bound': (loc + self.lmax * adm).item(),
                'bound_gap': (loc + self.lmax * adm - mix).item(),
                'inadmissible_mass': torch.stack(masses).mean().item(), 'warmup': warm,
                'predictive_objective': self.predictive_objective,
                'predictive_hinge': float(pred.detach()) if constraint_info is not None else None,
                'weighted_predictive': float((self.lambda_pred * pred).detach()),
                'admissibility_coefficient': adm_coefficient,
                'weighted_admissibility': float((adm_coefficient * adm).detach()),
                'weighted_mmd': float(self.lambda_sub * sub),
                'local_risk_by_domain': [float(v.detach()) for v in locals_],
                'admissibility_by_domain': [float(v.detach()) for v in adms],
                'worst_expert_fraction_by_domain': worst_counts,
                'classifier_row_norms': self.moe_head.classifier.weight.detach().norm(dim=1).tolist(),
                **diagnostics}
