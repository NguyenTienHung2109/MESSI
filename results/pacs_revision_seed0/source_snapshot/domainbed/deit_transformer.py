# gmoe_utils.py
# Shared building blocks for GMoE variants:
#   - DeiTFeaturizer   — pretrained DeiT-small backbone (CLS token output)
#   - ExplicitMoEHead  — M expert MLPs + soft router + classifier
#   - loss functions   — inv_A, inv_B, sparse, balance, diversity, cond_independence

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed import vision_transformer as vit_module


_PRETRAINED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'domainbed', 'pretrained',
)


def resize_pos_embed_deit(pos_embed, old_grid_size, new_grid_size,
                          num_prefix_tokens=1):
    prefix_pos = pos_embed[:, :num_prefix_tokens, :]
    patch_pos = pos_embed[:, num_prefix_tokens:, :]

    bsz, num_patches, dim = patch_pos.shape
    if bsz != 1:
        raise ValueError(f"Expected batch size 1 in pos_embed, got {bsz}")
    if num_patches != old_grid_size * old_grid_size:
        raise ValueError(
            f"Expected {old_grid_size * old_grid_size} patch tokens for "
            f"old_grid_size={old_grid_size}, got {num_patches}"
        )

    patch_pos = patch_pos.reshape(1, old_grid_size, old_grid_size, dim)
    patch_pos = patch_pos.permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos, size=(new_grid_size, new_grid_size),
        mode='bicubic', align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1)
    patch_pos = patch_pos.reshape(1, new_grid_size * new_grid_size, dim)
    return torch.cat([prefix_pos, patch_pos], dim=1)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

_DEIT_CONFIGS = {
    # name -> (embed_dim, depth, num_heads, pretrained_url)
    'deit_tiny_patch16_224': (
        192, 12, 3,
        'https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
    ),
    'deit_small_patch16_224': (
        384, 12, 6,
        'https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
    ),
    'deit_base_patch16_224': (
        768, 12, 12,
        'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
    ),
}


class DeiTFeaturizer(nn.Module):
    """
    Wraps the repo's VisionTransformer (DeiT tiny/small/base config) to expose
    the CLS-token feature vector. All layers are dense ('F') — no MoE, no Tutel.

    Model is selected by `model_name`:
        'deit_tiny_patch16_224'  → embed_dim=192
        'deit_small_patch16_224' → embed_dim=384  (default)
        'deit_base_patch16_224'  → embed_dim=768
    """
    def __init__(self, pretrained=True, model_name='deit_small_patch16_224',
                 img_size=224, patch_size=16, in_chans=3):
        super().__init__()
        if model_name not in _DEIT_CONFIGS:
            raise ValueError(
                f"Unknown DeiT model '{model_name}'. "
                f"Supported: {list(_DEIT_CONFIGS)}"
            )
        embed_dim, depth, num_heads, ckpt_url = _DEIT_CONFIGS[model_name]

        self.model_name = model_name
        self.vit = vit_module.VisionTransformer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            num_classes=0,              # no head — returns raw features
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=4., qkv_bias=True,
            distilled=True,             # distilled checkpoints have dist token
            drop_path_rate=0.1,
            moe_layers=['F'] * depth,   # all-dense, no Tutel
            num_experts=1,
            router='cosine_top',
        )
        self.n_outputs = embed_dim

        if pretrained:
            shape_changed = (img_size, patch_size, in_chans) != (224, 16, 3)
            if shape_changed:
                print(f'[DeiTFeaturizer:{model_name}] shape override '
                      f'(img_size={img_size}, patch_size={patch_size}, '
                      f'in_chans={in_chans}); compatible pretrained weights '
                      f'will be reused when possible.')
            # Prefer a local checkpoint shipped with the repo over downloading
            # from the URL — needed on servers without outbound HTTPS access.
            local_ckpt = os.path.join(_PRETRAINED_DIR, os.path.basename(ckpt_url))
            if os.path.exists(local_ckpt):
                print(f'[DeiTFeaturizer:{model_name}] loading local checkpoint '
                      f'{local_ckpt}')
                checkpoint = torch.load(local_ckpt, map_location='cpu')
            else:
                checkpoint = torch.hub.load_state_dict_from_url(
                    ckpt_url, map_location='cpu', check_hash=True,
                )
            state = checkpoint.get('model', checkpoint)
            # Drop classifier head keys — not present when num_classes=0
            state = {k: v for k, v in state.items()
                     if not k.startswith('head')}

            own = self.vit.state_dict()
            if 'pos_embed' in state and state['pos_embed'].shape != own['pos_embed'].shape:
                if img_size % patch_size != 0:
                    raise ValueError(
                        f"img_size={img_size} must be divisible by patch_size={patch_size}"
                    )
                num_prefix_tokens = self.vit.num_tokens
                num_patch_tokens = state['pos_embed'].shape[1] - num_prefix_tokens
                old_grid_size = int(math.sqrt(num_patch_tokens))
                if old_grid_size * old_grid_size != num_patch_tokens:
                    raise ValueError(
                        f"Cannot infer square positional grid from pos_embed "
                        f"shape {tuple(state['pos_embed'].shape)} with "
                        f"num_prefix_tokens={num_prefix_tokens}"
                    )
                new_grid_size = img_size // patch_size
                print(
                    f'[DeiTFeaturizer:{model_name}] resolution={img_size}; '
                    f'num_prefix_tokens={num_prefix_tokens}; interpolating '
                    f'pos_embed {old_grid_size}x{old_grid_size} -> '
                    f'{new_grid_size}x{new_grid_size}; old_shape='
                    f'{list(state["pos_embed"].shape)}'
                )
                state['pos_embed'] = resize_pos_embed_deit(
                    state['pos_embed'], old_grid_size, new_grid_size,
                    num_prefix_tokens=num_prefix_tokens,
                )
                print(f'[DeiTFeaturizer:{model_name}] new pos_embed shape: '
                      f'{list(state["pos_embed"].shape)}')

            if shape_changed:
                # Drop remaining shape-incompatible keys explicitly so
                # load_state_dict doesn't error on size mismatch. Resolution-only
                # changes keep patch projection weights and resized pos_embed.
                state = {k: v for k, v in state.items()
                         if k in own and own[k].shape == v.shape}
            missing, _ = self.vit.load_state_dict(state, strict=False)
            if missing:
                print(f'[DeiTFeaturizer:{model_name}] missing keys '
                      f'(expected if head removed or shape changed): {missing}')

    def forward(self, x):
        # forward_features returns (cls, dist) tuple for distilled DeiT
        out = self.vit.forward_features(x)
        if isinstance(out, tuple):
            return out[0]   # CLS token → (B, embed_dim)
        return out


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