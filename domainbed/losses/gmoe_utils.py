# gmoe_utils.py
# Shared building blocks for GMoE variants:
#   - DeiTFeaturizer   — pretrained DeiT-small backbone (CLS token output)
#   - ExplicitMoEHead  — M expert MLPs + soft router + classifier
#   - loss functions   — inv_A, inv_B, sparse, balance, diversity, cond_independence

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed import vision_transformer as vit_module


# ---------------------------------------------------------------------------
# Explicit MoE head
# ---------------------------------------------------------------------------

class ExplicitMoEHead(nn.Module):
    """
    Implements:
        h_m = E_m(z_moe)      M expert MLPs
        pi(x) = softmax(G(z_moe)) soft routing weights
        h(x)  = sum_m pi_m * h_m
        y_hat = C(h(x))       linear classifier

    Optional input projection maps backbone features into a compact MoE space
    before routing, expert computation, and classification.
    """
    def __init__(self, in_dim, num_experts, num_classes, mlp_ratio=4,
                 moe_dim=None):
        super().__init__()
        self.num_experts = num_experts
        self.in_dim      = in_dim
        if moe_dim is None or (isinstance(moe_dim, str) and moe_dim.lower() in ('auto', 'none')):
            moe_dim = in_dim
        elif int(moe_dim) == 0:
            moe_dim = in_dim
        self.expert_dim  = int(moe_dim)
        self.mlp_ratio   = mlp_ratio
        hidden_dim       = int(round(mlp_ratio * self.expert_dim))

        if self.expert_dim == in_dim:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(in_dim, self.expert_dim)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.expert_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.expert_dim),
            )
            for _ in range(num_experts)
        ])

        self.router = nn.Linear(self.expert_dim, num_experts, bias=True)
        self.classifier = nn.Linear(self.expert_dim, num_classes)

    def forward(self, z):
        z_moe = self.input_proj(z)
        h_list  = [E(z_moe) for E in self.experts]
        h_stack = torch.stack(h_list, dim=1)
        pi = F.softmax(self.router(z_moe), dim=-1)
        h  = (pi.unsqueeze(-1) * h_stack).sum(dim=1)
        logits = self.classifier(h)
        return logits, pi, h_stack


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _routed_class_mean(h_m, pi_m, y, num_classes):
    """
    Routing-weighted class-conditional mean for a single expert.

    Args:
        h_m:  (B, r)  expert m output
        pi_m: (B,)    routing weight for expert m
        y:    (B,)    class labels
    Returns:
        mu:   (C, r)  — zero rows where class has no samples in this batch
    """
    r  = h_m.size(1)
    mu = h_m.new_zeros(num_classes, r)
    for c in range(num_classes):
        mask  = (y == c).float()
        w     = pi_m * mask
        denom = w.sum()
        if denom > 1e-8:
            mu[c] = (w.unsqueeze(-1) * h_m).sum(0) / denom
    return mu


def _routed_class_cov(h_m, pi_m, mu_m, y, num_classes):
    """
    Routing-weighted class-conditional covariance for a single expert.

    Args:
        h_m:  (B, r)
        pi_m: (B,)
        mu_m: (C, r)  class means from _routed_class_mean
        y:    (B,)
    Returns:
        Sigma: (C, r, r)
    """
    r     = h_m.size(1)
    Sigma = h_m.new_zeros(num_classes, r, r)
    for c in range(num_classes):
        mask  = (y == c).float()
        w     = pi_m * mask
        denom = w.sum()
        if denom > 1e-8:
            diff     = h_m - mu_m[c]                          # (B, r)
            Sigma[c] = (w.unsqueeze(-1).unsqueeze(-1)
                        * diff.unsqueeze(-1) * diff.unsqueeze(-2)
                       ).sum(0) / denom
    return Sigma


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def loss_inv_A(h_stack, pi, y, num_classes, domain_ids, num_domains):
    """
    Option A — Expert-wise first-order (mean) alignment.

    L_inv^A = sum_m sum_c sum_{d != d'} || mu_{m,d,c} - mu_{m,d',c} ||^2

    Args:
        h_stack:    (B, M, r)
        pi:         (B, M)
        y:          (B,)   class labels
        domain_ids: (B,)   domain indices in [0, num_domains)
    """
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(1).squeeze()

    for m in range(M):
        h_m  = h_stack[:, m, :]
        pi_m = pi[:, m]

        mu_per_domain = []
        for d in range(num_domains):
            mask_d = (domain_ids == d)
            if mask_d.sum() == 0:
                mu_per_domain.append(None)
                continue
            mu_per_domain.append(
                _routed_class_mean(h_m[mask_d], pi_m[mask_d], y[mask_d], num_classes)
            )

        for d in range(num_domains):
            if mu_per_domain[d] is None:
                continue
            for dp in range(d + 1, num_domains):
                if mu_per_domain[dp] is None:
                    continue
                diff = mu_per_domain[d] - mu_per_domain[dp]   # (C, r)
                loss = loss + (diff ** 2).sum()

    return loss


def loss_inv_B(h_stack, pi, y, num_classes, domain_ids, num_domains, alpha=1.0):
    """
    Option B — Expert-wise second-order (mean + covariance) alignment.

    L_inv^B = sum_{m,c} sum_{d != d'} (
        || mu_{m,d,c} - mu_{m,d',c} ||^2
      + alpha * || Sigma_{m,d,c} - Sigma_{m,d',c} ||_F^2
    )

    Args:
        alpha: weight on the covariance term (default 1.0)
    """
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(1).squeeze()

    for m in range(M):
        h_m  = h_stack[:, m, :]
        pi_m = pi[:, m]

        mu_per_domain    = []
        sigma_per_domain = []
        for d in range(num_domains):
            mask_d = (domain_ids == d)
            if mask_d.sum() == 0:
                mu_per_domain.append(None)
                sigma_per_domain.append(None)
                continue
            mu_d    = _routed_class_mean(h_m[mask_d], pi_m[mask_d], y[mask_d], num_classes)
            sigma_d = _routed_class_cov(h_m[mask_d], pi_m[mask_d], mu_d, y[mask_d], num_classes)
            mu_per_domain.append(mu_d)
            sigma_per_domain.append(sigma_d)

        for d in range(num_domains):
            if mu_per_domain[d] is None:
                continue
            for dp in range(d + 1, num_domains):
                if mu_per_domain[dp] is None:
                    continue
                mu_diff    = mu_per_domain[d] - mu_per_domain[dp]
                sigma_diff = sigma_per_domain[d] - sigma_per_domain[dp]
                loss = loss + (mu_diff ** 2).sum()
                loss = loss + alpha * (sigma_diff ** 2).sum()

    return loss


def loss_sparse(pi):
    """
    Sparsity penalty — minimises routing entropy to encourage peaked routing.

    L_sp = E_x [ -sum_m pi_m log pi_m ]
    (minimising this minimises entropy → more concentrated routing)

    Args:
        pi: (B, M)
    """
    entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)   # (B,)
    return entropy.mean()


def loss_balance(pi):
    """
    Load balancing — penalises deviation of mean routing from uniform 1/M.

    L_bal = sum_m ( E[pi_m] - 1/M )^2

    Args:
        pi: (B, M)
    """
    M    = pi.size(1)
    mean = pi.mean(dim=0)                     # (M,)
    return ((mean - 1.0 / M) ** 2).sum()


def loss_diversity(h_stack):
    """
    Expert diversity — minimises cross-expert batch correlation.

    L_div = sum_{m != n} || (1/B) H_m^T H_n ||_F^2

    Args:
        h_stack: (B, M, r)
    """
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(1).squeeze()
    for m in range(M):
        for n in range(M):
            if m == n:
                continue
            C = (h_stack[:, m, :].T @ h_stack[:, n, :]) / B   # (r, r)
            loss = loss + (C ** 2).sum()
    return loss


def loss_cond_independence(h_stack, y, num_classes):
    """
    Conditional independence — minimises class-conditional cross-expert correlation.

    L_cind = sum_c sum_{m != n} || C_mn^(c) ||_F^2

    where C_mn^(c) is the class-conditional cross-correlation matrix between
    experts m and n.

    Args:
        h_stack: (B, M, r)
        y:       (B,)
    """
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(1).squeeze()

    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 2:
            continue
        H_c = h_stack[mask]           # (B_c, M, r)
        B_c = H_c.size(0)
        for m in range(M):
            for n in range(M):
                if m == n:
                    continue
                Hm = H_c[:, m, :] - H_c[:, m, :].mean(0)
                Hn = H_c[:, n, :] - H_c[:, n, :].mean(0)
                C_mn = (Hm.T @ Hn) / max(B_c - 1, 1)   # (r, r)
                loss = loss + (C_mn ** 2).sum()

    return loss


# ===========================================================================
# Subset-aware invariance variants
# ===========================================================================
#
# All four variants share the same skeleton:
#   for each expert m, class c, pair of domains (i, j):
#       rho_i  = mean(pi_m(x))  over samples of class c in domain i   (stop-grad)
#       rho_j  = mean(pi_m(x))  over samples of class c in domain j   (stop-grad)
#       a_ijc  = sigma(alpha * rho_i) * sigma(alpha * rho_j)
#       L     += a_ijc * DISTANCE(Z_m_i_c, Z_m_j_c)
#
# The variants differ only in DISTANCE:
#   loss_inv_MMD : routing-weighted conditional MMD²  (RBF kernel, multi-bandwidth)
#   loss_inv_OT  : routing-weighted entropic Wasserstein (Sinkhorn)
#   loss_inv_Adv : routing-weighted conditional adversarial alignment
#   loss_inv_ED  : routing-weighted energy distance


def _routing_weight(pi_m_i, pi_m_j, alpha):
    """
    Routing-dependent pairwise weight:
        a = sigma(alpha * rho_i) * sigma(alpha * rho_j)    (stop-gradient)
    Both rhos are detached so the weight does not back-propagate into the router.
    """
    rho_i = pi_m_i.detach().mean()
    rho_j = pi_m_j.detach().mean()
    return torch.sigmoid(alpha * rho_i) * torch.sigmoid(alpha * rho_j)


# ---------------------------------------------------------------------------
# Variant 1: Conditional MMD
# ---------------------------------------------------------------------------

def _pairwise_sq_dist(A, B):
    """
    Squared Euclidean pairwise distances, shape (|A|, |B|).
    Uses the (a - b)^2 = a^2 - 2ab + b^2 identity.
    """
    A2 = (A * A).sum(-1, keepdim=True)           # (|A|, 1)
    B2 = (B * B).sum(-1, keepdim=True).T        # (1,   |B|)
    return (A2 + B2 - 2.0 * A @ B.T).clamp_min(0.0)


def _mmd2_rbf(A, B, sigmas=(1., 2., 4., 8., 16.)):
    """
    Multi-bandwidth RBF MMD² between feature sets A, B  (unbiased U-statistic).

    MMD²(P, Q) = E_pp k(z,z') + E_qq k(z,z') - 2 E_pq k(z,z')
    """
    if A.size(0) < 2 or B.size(0) < 2:
        return A.new_zeros(())       # not enough samples for a meaningful stat

    d_AA = _pairwise_sq_dist(A, A)
    d_BB = _pairwise_sq_dist(B, B)
    d_AB = _pairwise_sq_dist(A, B)

    mmd2 = A.new_zeros(())
    for s in sigmas:
        k_AA = torch.exp(-d_AA / (2.0 * s * s))
        k_BB = torch.exp(-d_BB / (2.0 * s * s))
        k_AB = torch.exp(-d_AB / (2.0 * s * s))

        # Unbiased estimator: drop diagonal
        n = A.size(0); m = B.size(0)
        k_AA_nd = (k_AA.sum() - k_AA.diag().sum()) / (n * (n - 1))
        k_BB_nd = (k_BB.sum() - k_BB.diag().sum()) / (m * (m - 1))
        k_AB_m  = k_AB.mean()

        mmd2 = mmd2 + (k_AA_nd + k_BB_nd - 2.0 * k_AB_m)

    return mmd2 / len(sigmas)


def loss_inv_MMD(h_stack, pi, y, num_classes, domain_ids, num_domains,
                 alpha=4.0, sigmas=(1., 2., 4., 8., 16.)):
    """
    Subset-aware invariance via conditional MMD.

    L_inv^MMD = sum_m sum_c sum_{i<j} a_ijc * MMD²( Z_m_i_c , Z_m_j_c )

    Args:
        h_stack:    (B, M, r)
        pi:         (B, M)
        y:          (B,)        class labels
        domain_ids: (B,)        domain indices
        alpha:      temperature for the routing-weight sigmoid
        sigmas:     RBF bandwidths (multi-scale)
    """
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(())

    for m in range(M):
        h_m  = h_stack[:, m, :]     # (B, r)
        pi_m = pi[:, m]             # (B,)

        for c in range(num_classes):
            mask_c = (y == c)
            if mask_c.sum() < 2:
                continue

            for i in range(num_domains):
                mask_i = mask_c & (domain_ids == i)
                if mask_i.sum() < 2:
                    continue
                Z_i   = h_m[mask_i]
                pi_i  = pi_m[mask_i]

                for j in range(i + 1, num_domains):
                    mask_j = mask_c & (domain_ids == j)
                    if mask_j.sum() < 2:
                        continue
                    Z_j  = h_m[mask_j]
                    pi_j = pi_m[mask_j]

                    a = _routing_weight(pi_i, pi_j, alpha)
                    loss = loss + a * _mmd2_rbf(Z_i, Z_j, sigmas=sigmas)

    return loss


def loss_inv_MMD_ablation(h_stack, y, num_classes, domain_ids, num_domains,
                          alignment_mode, sigmas=(1., 2., 4., 8., 16.),
                          random_subset_q=0.30, random_seed=None):
    """
    Controlled-alignment ablations of L_inv^MMD.

    Same architecture and same inner discrepancy (multi-bandwidth RBF MMD²) as
    `loss_inv_MMD`, but replaces the routing-induced pair weights:

        'global'        a_ijc^(m) = 1   for every valid (m,c,i,j); average
                                          over valid pairs (weight scale matched
                                          to MESSI by mean-normalization).

        'random_subset' a_ijc^(m) = 1   for a random subset of valid (m,c,i,j),
                                          0 otherwise; subset size matches the
                                          top-`random_subset_q` count;
                                          average over selected pairs.

    A pair (m,c,i,j) is *valid* when both class-domain groups (m,i,c) and
    (m,j,c) have ≥ 2 samples in the batch — exactly the same guard used by the
    MESSI path in `loss_inv_MMD`.

    Notes:
        * `pi` is intentionally not consumed here: routing probabilities should
          play no role in pair selection for these baselines.
        * Random selection uses an isolated `torch.Generator` seeded by
          `random_seed` (typically `base_seed + step`) so the loss is
          reproducible and does not perturb the global RNG used by the rest
          of training.

    Args:
        h_stack:        (B, M, r)
        y:              (B,)        class labels
        domain_ids:     (B,)        domain indices
        alignment_mode: 'global' | 'random_subset'
        sigmas:         RBF bandwidths (multi-scale)
        random_subset_q: target sparsity for 'random_subset' (default 0.30)
        random_seed:    seed for the random-subset RNG; if None falls back to
                        torch's global RNG state.
    """
    if alignment_mode not in ('global', 'random_subset'):
        raise ValueError(
            f"alignment_mode must be 'global' or 'random_subset', "
            f"got {alignment_mode!r}"
        )

    B, M, r = h_stack.shape

    # PASS 1 — enumerate valid pairs using the same guard as the messi path.
    valid_pairs = []
    for m in range(M):
        h_m = h_stack[:, m, :]
        for c in range(num_classes):
            mask_c = (y == c)
            if mask_c.sum() < 2:
                continue
            for i in range(num_domains):
                mask_i = mask_c & (domain_ids == i)
                if mask_i.sum() < 2:
                    continue
                Z_i = h_m[mask_i]
                for j in range(i + 1, num_domains):
                    mask_j = mask_c & (domain_ids == j)
                    if mask_j.sum() < 2:
                        continue
                    Z_j = h_m[mask_j]
                    valid_pairs.append((Z_i, Z_j))

    n_valid = len(valid_pairs)
    if n_valid == 0:
        return h_stack.new_zeros(())

    if alignment_mode == 'global':
        loss_sum = h_stack.new_zeros(())
        for Z_i, Z_j in valid_pairs:
            loss_sum = loss_sum + _mmd2_rbf(Z_i, Z_j, sigmas=sigmas)
        return loss_sum / n_valid

    # random_subset — exact-count selection matched to MESSI's top-q sparsity.
    n_select = max(1, int(round(random_subset_q * n_valid)))
    g = torch.Generator(device='cpu')
    if random_seed is not None:
        g.manual_seed(int(random_seed) & 0x7FFFFFFFFFFFFFFF)
    chosen = torch.randperm(n_valid, generator=g).tolist()[:n_select]
    loss_sum = h_stack.new_zeros(())
    for idx in chosen:
        Z_i, Z_j = valid_pairs[idx]
        loss_sum = loss_sum + _mmd2_rbf(Z_i, Z_j, sigmas=sigmas)
    return loss_sum / n_select


def loss_inv_MMD_domain_routing(h_stack, pi, y, num_classes, domain_ids,
                                num_domains, alpha=4.0,
                                sigmas=(1., 2., 4., 8., 16.)):
    """
    Domain-only routing variant of L_inv^MMD.

    Routing mass is computed per *domain* only — the class index is dropped:

        rho_k^{(m)} = (1/|D_k|) * sum_{(x,y) in D_k} pi_m(x)

    The pair-responsibility weight therefore loses its class index too:

        a_{ij}^{(m)} = sigma(alpha * rho_i^{(m)}) * sigma(alpha * rho_j^{(m)})

    The outer summation is unchanged from MESSI: still over (m, c, i<j) with
    the same class-conditional MMD² between expert subsets
    Z_{i,c}^{(m)}, Z_{j,c}^{(m)}. Only the pair weight differs (class-
    independent here, class-conditional in MESSI). No mean-normalization —
    the soft sigmoid weight serves as the implicit mask, identical to MESSI.

    rho is detached so the router does not back-propagate through the weight.

    Args:
        h_stack:    (B, M, r)
        pi:         (B, M)
        y:          (B,)        class labels
        domain_ids: (B,)        domain indices
        alpha:      temperature for the routing-weight sigmoid (default 4.0)
        sigmas:     RBF bandwidths (multi-scale)
    """
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(())

    for m in range(M):
        h_m  = h_stack[:, m, :]
        pi_m = pi[:, m]

        # Pre-compute per-domain sigmoid(alpha * rho_k^(m)) once for this expert.
        sig_per_dom = h_stack.new_zeros(num_domains)
        valid_dom = [False] * num_domains
        for k in range(num_domains):
            mask_k = (domain_ids == k)
            if mask_k.sum() < 2:
                continue
            rho_k = pi_m[mask_k].detach().mean()
            sig_per_dom[k] = torch.sigmoid(alpha * rho_k)
            valid_dom[k] = True

        for c in range(num_classes):
            mask_c = (y == c)
            if mask_c.sum() < 2:
                continue

            for i in range(num_domains):
                if not valid_dom[i]:
                    continue
                mask_i = mask_c & (domain_ids == i)
                if mask_i.sum() < 2:
                    continue
                Z_i = h_m[mask_i]

                for j in range(i + 1, num_domains):
                    if not valid_dom[j]:
                        continue
                    mask_j = mask_c & (domain_ids == j)
                    if mask_j.sum() < 2:
                        continue
                    Z_j = h_m[mask_j]

                    a = sig_per_dom[i] * sig_per_dom[j]   # class-independent
                    loss = loss + a * _mmd2_rbf(Z_i, Z_j, sigmas=sigmas)

    return loss


# ---------------------------------------------------------------------------
# Variant 2: Conditional entropic Optimal Transport (Sinkhorn)
# ---------------------------------------------------------------------------

def _sinkhorn(A, B, epsilon=0.1, n_iter=50):
    """
    Entropic-regularised squared-Wasserstein distance between two empirical
    distributions with uniform weights.  Differentiable Sinkhorn iterations
    in log-space for numerical stability.
    """
    n, m = A.size(0), B.size(0)
    if n == 0 or m == 0:
        return A.new_zeros(())

    C = _pairwise_sq_dist(A, B)                                     # (n, m)
    log_mu = -torch.log(A.new_tensor(float(n)))                     # uniform
    log_nu = -torch.log(A.new_tensor(float(m)))

    log_u = A.new_zeros(n)
    log_v = A.new_zeros(m)
    log_K = -C / epsilon                                            # (n, m)

    for _ in range(n_iter):
        # u-update: log_u = log_mu - logsumexp_j (log_K + log_v)
        log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        # v-update: log_v = log_nu - logsumexp_i (log_K + log_u)
        log_v = log_nu - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)

    # Transport plan: gamma = exp(log_u_i + log_K_ij + log_v_j)
    log_gamma = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    gamma = log_gamma.exp()
    return (gamma * C).sum()


def loss_inv_OT(h_stack, pi, y, num_classes, domain_ids, num_domains,
                alpha=4.0, epsilon=0.1, sinkhorn_iters=50):
    """
    Subset-aware invariance via conditional entropic optimal transport.

    L_inv^OT = sum_m sum_c sum_{i<j} a_ijc * W_eps( Z_m_i_c , Z_m_j_c )

    Args:
        alpha:          temperature for routing-weight sigmoid
        epsilon:        entropy regularisation for Sinkhorn
        sinkhorn_iters: number of Sinkhorn updates
    """
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(())

    for m in range(M):
        h_m  = h_stack[:, m, :]
        pi_m = pi[:, m]

        for c in range(num_classes):
            mask_c = (y == c)
            if mask_c.sum() < 2:
                continue

            for i in range(num_domains):
                mask_i = mask_c & (domain_ids == i)
                if mask_i.sum() == 0:
                    continue
                Z_i  = h_m[mask_i]
                pi_i = pi_m[mask_i]

                for j in range(i + 1, num_domains):
                    mask_j = mask_c & (domain_ids == j)
                    if mask_j.sum() == 0:
                        continue
                    Z_j  = h_m[mask_j]
                    pi_j = pi_m[mask_j]

                    a = _routing_weight(pi_i, pi_j, alpha)
                    loss = loss + a * _sinkhorn(Z_i, Z_j,
                                                epsilon=epsilon,
                                                n_iter=sinkhorn_iters)

    return loss


def loss_inv_OT_reliable(
        h_stack, pi, y, num_classes, domain_ids, num_domains,
        alpha=4.0, epsilon=0.1, sinkhorn_iters=50,
        reliability_mode="none", r_n_min=2, r_tau=4.0,
        r_emin=4.0, ema_evidence=None, r_detach=True, eps=1e-8,
        reliable_log_detail="compact"):
    """
    Reliability-aware conditional entropic OT for rMESSI_InvOT.

    The legacy none mode intentionally keeps the original unnormalized
    routing-weighted sum so it can be compared directly to loss_inv_OT.
    Reliability-enabled modes normalize by the effective weight sum.
    """
    if reliability_mode in ("ema_coverage_conf", "full"):
        raise NotImplementedError(
            f"reliability_mode={reliability_mode!r} is reserved for a later "
            "confidence/stability ablation"
        )
    if reliability_mode not in ("none", "mask", "coverage", "ema_coverage"):
        raise ValueError(
            "reliability_mode must be one of: none, mask, coverage, "
            "ema_coverage, ema_coverage_conf, full"
        )
    if reliable_log_detail not in ("compact", "full"):
        raise ValueError("reliable_log_detail must be 'compact' or 'full'")
    if reliability_mode == "ema_coverage" and ema_evidence is None:
        raise ValueError("ema_evidence is required for reliability_mode=ema_coverage")

    B, M, r = h_stack.shape
    del B, r

    loss_sum = h_stack.new_zeros(())
    weight_sum = h_stack.new_zeros(())
    legacy_loss = h_stack.new_zeros(())

    r_sum = h_stack.new_zeros(())
    r_min = None
    r_max = None
    coverage_sum = h_stack.new_zeros(())
    coverage_candidate_sum = h_stack.new_zeros(())
    ot_sum = h_stack.new_zeros(())
    resp_sum = h_stack.new_zeros(())

    num_candidate_pairs = 0
    num_batch_keep_pairs = 0
    num_valid_pairs = 0
    num_valid_slots = 0
    num_skipped_low_count = 0
    nonzero_reliability_slots = 0
    ema_E_valid_pairs = 0
    ema_extra_pairs = 0

    for c in range(num_classes):
        mask_c = (y == c)
        if mask_c.sum() < 2:
            continue

        domain_masks = []
        domain_counts = []
        present_domains = []
        for i in range(num_domains):
            mask_i = mask_c & (domain_ids == i)
            n_i = int(mask_i.sum().item())
            domain_masks.append(mask_i)
            domain_counts.append(n_i)
            if n_i > 0:
                present_domains.append(i)

        for pos, i in enumerate(present_domains):
            n_i = domain_counts[i]
            for j in present_domains[pos + 1:]:
                n_j = domain_counts[j]
                num_candidate_pairs += 1
                batch_keep = n_i >= r_n_min and n_j >= r_n_min
                if batch_keep:
                    num_batch_keep_pairs += 1

                if reliability_mode in ("mask", "coverage"):
                    if n_i < r_n_min or n_j < r_n_min:
                        num_skipped_low_count += 1
                        continue
                    coverage = h_stack.new_tensor(1.0)
                    if reliability_mode == "coverage":
                        ni = h_stack.new_tensor(float(n_i))
                        nj = h_stack.new_tensor(float(n_j))
                        coverage = torch.sqrt(
                            (ni * nj) / ((ni + r_tau) * (nj + r_tau))
                        )
                elif reliability_mode == "ema_coverage":
                    e_i = ema_evidence[i, c].to(device=h_stack.device,
                                                dtype=h_stack.dtype)
                    e_j = ema_evidence[j, c].to(device=h_stack.device,
                                                dtype=h_stack.dtype)
                    ema_valid = bool((e_i >= r_emin).item() and (e_j >= r_emin).item())
                    if ema_valid:
                        ema_E_valid_pairs += 1
                        if not batch_keep:
                            ema_extra_pairs += 1
                    if not ema_valid:
                        continue
                    coverage = torch.sqrt(
                        (e_i * e_j) / ((e_i + r_tau) * (e_j + r_tau))
                    )
                else:
                    coverage = h_stack.new_tensor(1.0)

                coverage_candidate_sum = coverage_candidate_sum + coverage.detach()
                r_weight = coverage.detach() if r_detach else coverage
                num_valid_pairs += 1

                mask_i = domain_masks[i]
                mask_j = domain_masks[j]
                for m in range(M):
                    h_m = h_stack[:, m, :]
                    pi_m = pi[:, m]
                    Z_i = h_m[mask_i]
                    Z_j = h_m[mask_j]
                    pi_i = pi_m[mask_i]
                    pi_j = pi_m[mask_j]

                    a = _routing_weight(pi_i, pi_j, alpha)
                    ot_cost = _sinkhorn(Z_i, Z_j, epsilon=epsilon,
                                        n_iter=sinkhorn_iters)
                    weighted = r_weight * a

                    legacy_loss = legacy_loss + a * ot_cost
                    loss_sum = loss_sum + weighted * ot_cost
                    weight_sum = weight_sum + weighted

                    num_valid_slots += 1
                    r_detached = r_weight.detach()
                    if bool((r_detached > 0).item()):
                        nonzero_reliability_slots += 1
                    r_sum = r_sum + r_detached
                    coverage_sum = coverage_sum + coverage.detach()
                    ot_sum = ot_sum + ot_cost.detach()
                    resp_sum = resp_sum + a.detach()
                    r_min = r_detached if r_min is None else torch.minimum(r_min, r_detached)
                    r_max = r_detached if r_max is None else torch.maximum(r_max, r_detached)

    if reliability_mode == "none":
        loss = legacy_loss
    elif num_valid_slots == 0:
        loss = h_stack.sum() * 0.0
    else:
        loss = loss_sum / (weight_sum + eps)

    slot_denominator = max(float(num_candidate_pairs * M), 1.0)
    pair_denominator = max(float(num_candidate_pairs), 1.0)
    valid_slot_denominator = float(num_valid_slots) if num_valid_slots else 1.0
    weighted_ot = loss_sum / (weight_sum + eps) if num_valid_slots else h_stack.new_zeros(())

    # coverage_mean follows the historical convention: valid slots only.
    diagnostics = {
        "num_valid_slots": float(num_valid_slots),
        "slot_density": float(num_valid_slots) / slot_denominator,
        "r_mean": float((r_sum / slot_denominator).item()),
        "r_nonzero_frac": float(nonzero_reliability_slots) / slot_denominator,
        "w_sum": float(weight_sum.detach().item()),
        "coverage_mean": float((coverage_sum / valid_slot_denominator).item()) if num_valid_slots else 0.0,
        "mask_keep_frac": float(num_batch_keep_pairs) / pair_denominator,
        "ot_cost_weighted_mean": float(weighted_ot.detach().item()),
    }
    if reliability_mode == "ema_coverage":
        diagnostics.update({
            "ema_E_valid_frac": float(ema_E_valid_pairs) / pair_denominator,
            "ema_extra_pairs": float(ema_extra_pairs),
        })
    if reliable_log_detail == "full":
        diagnostics.update({
            "r_min": float(r_min.item()) if r_min is not None else 0.0,
            "r_max": float(r_max.item()) if r_max is not None else 0.0,
            "r_sum": float(r_sum.item()),
            "num_candidate_pairs": float(num_candidate_pairs),
            "num_valid_pairs": float(num_valid_pairs),
            "num_skipped_low_count": float(num_skipped_low_count),
            "mean_responsibility": float((resp_sum / valid_slot_denominator).item()) if num_valid_slots else 0.0,
            "mean_ot_cost": float((ot_sum / valid_slot_denominator).item()) if num_valid_slots else 0.0,
            "coverage_mean_candidate": float((coverage_candidate_sum / pair_denominator).item()),
            "coverage_mean_valid": float((coverage_sum / valid_slot_denominator).item()) if num_valid_slots else 0.0,
        })
        if reliability_mode == "ema_coverage":
            diagnostics.update({
                "ema_keep_frac": float(ema_E_valid_pairs) / pair_denominator,
                "ema_extra_frac": float(ema_extra_pairs) / pair_denominator,
            })
    return loss, diagnostics


# ---------------------------------------------------------------------------
# Variant 3: Conditional adversarial alignment
# ---------------------------------------------------------------------------

class _GradReverse(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


class ConditionalDomainDiscriminators(nn.Module):
    """
    One (expert, class) -> domain discriminator:  D^{(m)}_c(z) -> (num_domains,)

    Implemented as an M x C grid of small MLPs.  At forward time we dispatch
    each (sample, expert, class) to the corresponding discriminator.
    """
    def __init__(self, num_experts, num_classes, feat_dim, num_domains, hidden=128):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.discriminators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, num_domains),
            )
            for _ in range(num_experts * num_classes)
        ])

    def predict(self, expert_idx, class_idx, z):
        """Forward pass through D^{(m=expert_idx)}_{c=class_idx}."""
        head = self.discriminators[expert_idx * self.num_classes + class_idx]
        return head(z)


def loss_inv_Adv(h_stack, pi, y, num_classes, domain_ids, num_domains,
                 discriminators, alpha=4.0, grl_lambda=1.0):
    """
    Subset-aware invariance via conditional adversarial alignment.

    For each (expert m, class c), a discriminator D^{(m)}_c tries to predict
    the domain from features; a gradient-reversal layer flips the gradient
    back to the expert, encouraging the expert to produce domain-invariant
    features within the class.

    L_inv^Adv = sum_m sum_c sum_{i<j} a_ijc * L_adv^{(m,c)}

    where L_adv^{(m,c)} is the cross-entropy of the discriminator on the
    union of samples of class c from domains i and j.  The same total CE
    is added once per (m, c) scaled by the mean of a_ijc across pairs.

    Args:
        discriminators: ConditionalDomainDiscriminators module
        alpha:          routing-weight temperature
        grl_lambda:     gradient reversal scale
    """
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(())

    for m in range(M):
        h_m  = h_stack[:, m, :]
        pi_m = pi[:, m]

        for c in range(num_classes):
            mask_c = (y == c)
            if mask_c.sum() < 2:
                continue

            # Compute per-pair routing weights and accumulate their sum
            total_a = h_m.new_zeros(())
            pair_count = 0
            for i in range(num_domains):
                mask_i = mask_c & (domain_ids == i)
                if mask_i.sum() == 0:
                    continue
                for j in range(i + 1, num_domains):
                    mask_j = mask_c & (domain_ids == j)
                    if mask_j.sum() == 0:
                        continue
                    total_a = total_a + _routing_weight(
                        pi_m[mask_i], pi_m[mask_j], alpha)
                    pair_count += 1

            if pair_count == 0:
                continue

            # Adversarial CE on all samples of class c across all domains
            Z_c   = grad_reverse(h_m[mask_c], grl_lambda)
            dom_c = domain_ids[mask_c]
            logits = discriminators.predict(m, c, Z_c)
            ce = F.cross_entropy(logits, dom_c)

            # Weight the CE by mean(a_ijc) — single scalar summarising all pairs
            loss = loss + (total_a / pair_count) * ce

    return loss


# ---------------------------------------------------------------------------
# Variant 4: Energy distance
# ---------------------------------------------------------------------------

def _energy_distance(A, B):
    """
    Energy distance:
        E(P, Q) = 2 E ||z - z'|| - E ||z - z''|| - E ||z' - z'''||
    where z,z'' ~ P and z',z''' ~ Q.

    Uses L2 (non-squared) Euclidean distance.
    """
    if A.size(0) < 2 or B.size(0) < 2:
        return A.new_zeros(())

    d_AA = _pairwise_sq_dist(A, A).clamp_min(1e-12).sqrt()
    d_BB = _pairwise_sq_dist(B, B).clamp_min(1e-12).sqrt()
    d_AB = _pairwise_sq_dist(A, B).clamp_min(1e-12).sqrt()

    n, m = A.size(0), B.size(0)

    # Unbiased means (drop diagonal on within-set terms)
    mean_AA = (d_AA.sum() - d_AA.diag().sum()) / (n * (n - 1))
    mean_BB = (d_BB.sum() - d_BB.diag().sum()) / (m * (m - 1))
    mean_AB = d_AB.mean()

    return 2.0 * mean_AB - mean_AA - mean_BB


def loss_inv_ED(h_stack, pi, y, num_classes, domain_ids, num_domains,
                alpha=4.0):
    """
    Subset-aware invariance via energy distance.

    L_inv^ED = sum_m sum_c sum_{i<j} a_ijc * E( Z_m_i_c , Z_m_j_c )
    """
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(())

    for m in range(M):
        h_m  = h_stack[:, m, :]
        pi_m = pi[:, m]

        for c in range(num_classes):
            mask_c = (y == c)
            if mask_c.sum() < 2:
                continue

            for i in range(num_domains):
                mask_i = mask_c & (domain_ids == i)
                if mask_i.sum() < 2:
                    continue
                Z_i  = h_m[mask_i]
                pi_i = pi_m[mask_i]

                for j in range(i + 1, num_domains):
                    mask_j = mask_c & (domain_ids == j)
                    if mask_j.sum() < 2:
                        continue
                    Z_j  = h_m[mask_j]
                    pi_j = pi_m[mask_j]

                    a = _routing_weight(pi_i, pi_j, alpha)
                    loss = loss + a * _energy_distance(Z_i, Z_j)

    return loss