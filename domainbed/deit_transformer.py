import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed import vision_transformer as vit_module

_MODEL_CONFIG = {
    "deit_tiny_patch16_224": {"embed_dim": 192, "depth": 12},
    "deit_small_patch16_224": {"embed_dim": 384, "depth": 12},
    "deit_base_patch16_224": {"embed_dim": 768, "depth": 12},
    "deit_base_patch16_384": {"embed_dim": 768, "depth": 12},
    "deit_tiny_distilled_patch16_224": {"embed_dim": 192, "depth": 12},
    "deit_small_distilled_patch16_224": {"embed_dim": 384, "depth": 12},
    "deit_base_distilled_patch16_224": {"embed_dim": 768, "depth": 12},
    "deit_base_distilled_patch16_384": {"embed_dim": 768, "depth": 12},
    "vit_tiny_patch16_224": {"embed_dim": 192, "depth": 12},
    "vit_small_patch16_224": {"embed_dim": 384, "depth": 12},
    "vit_small_patch32_224": {"embed_dim": 384, "depth": 12},
    "vit_base_patch16_224": {"embed_dim": 768, "depth": 12},
    "vit_base_patch32_224": {"embed_dim": 768, "depth": 12},
    "vit_base_patch8_224": {"embed_dim": 768, "depth": 12},
    "vit_large_patch16_224": {"embed_dim": 1024, "depth": 24},
    "vit_large_patch32_224": {"embed_dim": 1024, "depth": 24},
    "vit_huge_patch14_224": {"embed_dim": 1280, "depth": 32},
}


class DeiTFeaturizer(nn.Module):
    def __init__(self, model_name="deit_small_patch16_224", pretrained=True):
        super().__init__()
        if model_name not in _MODEL_CONFIG:
            raise ValueError(
                f"Unknown model {model_name!r}. "
                f"Supported: {sorted(_MODEL_CONFIG.keys())}"
            )
        cfg = _MODEL_CONFIG[model_name]
        depth = cfg["depth"]
        factory = getattr(vit_module, model_name)
        self.vit = factory(
            pretrained=pretrained,
            num_classes=0,
            moe_layers=["F"] * depth,
            num_experts=1,
            gate_k=1,
            prune_ratio=0.0,
            router="cosine_top",
            is_tutel=False,
            expert_depth=2,
            drop_path_rate=0.1,
        )
        self.n_outputs = cfg["embed_dim"]
        self.model_name = model_name

    def forward(self, x):
        out = self.vit.forward_features(x)
        if isinstance(out, tuple):
            return out[0]
        return out


def supported_models():
    return sorted(_MODEL_CONFIG.keys())


class ExplicitMoEHead(nn.Module):
    def __init__(
        self,
        in_dim,
        expert_dim,
        num_experts,
        num_classes,
        mlp_ratio=None,
        prune_ratio=0.0,
        expert_depth=2,
    ):
        super().__init__()
        if expert_depth < 2:
            raise ValueError("expert_depth must be >= 2")
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.expert_depth = expert_depth
        if mlp_ratio is not None:
            hidden = max(1, int(in_dim * mlp_ratio * (1.0 - prune_ratio)))
        else:
            hidden = expert_dim
        self.experts = nn.ModuleList(
            [
                self._make_expert(in_dim, hidden, expert_dim, expert_depth)
                for _ in range(num_experts)
            ]
        )
        self.router = nn.Linear(in_dim, num_experts, bias=True)
        self.classifier = nn.Linear(expert_dim, num_classes)

    @staticmethod
    def _make_expert(in_dim, hidden, out_dim, depth):
        dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
        layers = []
        for i in range(depth):
            lin = nn.Linear(dims[i], dims[i + 1])
            layers.append(lin)
            if i < depth - 1:
                layers.append(nn.GELU())
        mlp = nn.Sequential(*layers)
        for idx in range(1, depth - 1):
            layer = mlp[2 * idx]
            if layer.weight.shape[0] == layer.weight.shape[1]:
                nn.init.eye_(layer.weight)
                nn.init.zeros_(layer.bias)
        return mlp

    def forward(self, z):
        h_list = [E(z) for E in self.experts]
        h_stack = torch.stack(h_list, dim=1)
        pi = F.softmax(self.router(z), dim=-1)
        h = (pi.unsqueeze(-1) * h_stack).sum(dim=1)
        logits = self.classifier(h)
        return logits, pi, h_stack


def _routed_class_mean(h_m, pi_m, y, num_classes):
    r = h_m.size(1)
    mu = h_m.new_zeros(num_classes, r)
    for c in range(num_classes):
        mask = (y == c).float()
        w = pi_m * mask
        denom = w.sum()
        if denom > 1e-8:
            mu[c] = (w.unsqueeze(-1) * h_m).sum(0) / denom
    return mu


def _routed_class_cov(h_m, pi_m, mu_m, y, num_classes):
    r = h_m.size(1)
    Sigma = h_m.new_zeros(num_classes, r, r)
    for c in range(num_classes):
        mask = (y == c).float()
        w = pi_m * mask
        denom = w.sum()
        if denom > 1e-8:
            diff = h_m - mu_m[c]
            Sigma[c] = (
                w.unsqueeze(-1).unsqueeze(-1) * diff.unsqueeze(-1) * diff.unsqueeze(-2)
            ).sum(0) / denom
    return Sigma


def loss_inv_A(h_stack, pi, y, num_classes, domain_ids, num_domains):
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(1).squeeze()
    for m in range(M):
        h_m = h_stack[:, m, :]
        pi_m = pi[:, m]
        mu_per_domain = []
        for d in range(num_domains):
            mask_d = domain_ids == d
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
                diff = mu_per_domain[d] - mu_per_domain[dp]
                loss = loss + (diff**2).sum()
    return loss


def loss_inv_B(h_stack, pi, y, num_classes, domain_ids, num_domains, alpha=1.0):
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(1).squeeze()
    for m in range(M):
        h_m = h_stack[:, m, :]
        pi_m = pi[:, m]
        mu_per_domain = []
        sigma_per_domain = []
        for d in range(num_domains):
            mask_d = domain_ids == d
            if mask_d.sum() == 0:
                mu_per_domain.append(None)
                sigma_per_domain.append(None)
                continue
            mu_d = _routed_class_mean(h_m[mask_d], pi_m[mask_d], y[mask_d], num_classes)
            sigma_d = _routed_class_cov(
                h_m[mask_d], pi_m[mask_d], mu_d, y[mask_d], num_classes
            )
            mu_per_domain.append(mu_d)
            sigma_per_domain.append(sigma_d)
        for d in range(num_domains):
            if mu_per_domain[d] is None:
                continue
            for dp in range(d + 1, num_domains):
                if mu_per_domain[dp] is None:
                    continue
                mu_diff = mu_per_domain[d] - mu_per_domain[dp]
                sigma_diff = sigma_per_domain[d] - sigma_per_domain[dp]
                loss = loss + (mu_diff**2).sum()
                loss = loss + alpha * (sigma_diff**2).sum()
    return loss


def loss_sparse(pi):
    entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)
    return entropy.mean()


def loss_balance(pi):
    M = pi.size(1)
    mean = pi.mean(dim=0)
    return ((mean - 1.0 / M) ** 2).sum()


def loss_diversity(h_stack):
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(1).squeeze()
    for m in range(M):
        for n in range(M):
            if m == n:
                continue
            C = (h_stack[:, m, :].T @ h_stack[:, n, :]) / B
            loss = loss + (C**2).sum()
    return loss


def loss_cond_independence(h_stack, y, num_classes):
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(1).squeeze()
    for c in range(num_classes):
        mask = y == c
        if mask.sum() < 2:
            continue
        H_c = h_stack[mask]
        B_c = H_c.size(0)
        for m in range(M):
            for n in range(M):
                if m == n:
                    continue
                Hm = H_c[:, m, :] - H_c[:, m, :].mean(0)
                Hn = H_c[:, n, :] - H_c[:, n, :].mean(0)
                C_mn = (Hm.T @ Hn) / max(B_c - 1, 1)
                loss = loss + (C_mn**2).sum()
    return loss


def _routing_weight(pi_m_i, pi_m_j, alpha):
    rho_i = pi_m_i.detach().mean()
    rho_j = pi_m_j.detach().mean()
    return torch.sigmoid(alpha * rho_i) * torch.sigmoid(alpha * rho_j)


def _pairwise_sq_dist(A, B):
    A2 = (A * A).sum(-1, keepdim=True)
    B2 = (B * B).sum(-1, keepdim=True).T
    return (A2 + B2 - 2.0 * A @ B.T).clamp_min(0.0)


def _mmd2_rbf(A, B, sigmas=(1.0, 2.0, 4.0, 8.0, 16.0)):
    if A.size(0) < 2 or B.size(0) < 2:
        return A.new_zeros(())
    d_AA = _pairwise_sq_dist(A, A)
    d_BB = _pairwise_sq_dist(B, B)
    d_AB = _pairwise_sq_dist(A, B)
    mmd2 = A.new_zeros(())
    for s in sigmas:
        k_AA = torch.exp(-d_AA / (2.0 * s * s))
        k_BB = torch.exp(-d_BB / (2.0 * s * s))
        k_AB = torch.exp(-d_AB / (2.0 * s * s))
        n = A.size(0)
        m = B.size(0)
        k_AA_nd = (k_AA.sum() - k_AA.diag().sum()) / (n * (n - 1))
        k_BB_nd = (k_BB.sum() - k_BB.diag().sum()) / (m * (m - 1))
        k_AB_m = k_AB.mean()
        mmd2 = mmd2 + (k_AA_nd + k_BB_nd - 2.0 * k_AB_m)
    return mmd2 / len(sigmas)


def loss_inv_MMD(
    h_stack,
    pi,
    y,
    num_classes,
    domain_ids,
    num_domains,
    alpha=4.0,
    sigmas=(1.0, 2.0, 4.0, 8.0, 16.0),
):
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(())
    for m in range(M):
        h_m = h_stack[:, m, :]
        pi_m = pi[:, m]
        for c in range(num_classes):
            mask_c = y == c
            if mask_c.sum() < 2:
                continue
            for i in range(num_domains):
                mask_i = mask_c & (domain_ids == i)
                if mask_i.sum() < 2:
                    continue
                Z_i = h_m[mask_i]
                pi_i = pi_m[mask_i]
                for j in range(i + 1, num_domains):
                    mask_j = mask_c & (domain_ids == j)
                    if mask_j.sum() < 2:
                        continue
                    Z_j = h_m[mask_j]
                    pi_j = pi_m[mask_j]
                    a = _routing_weight(pi_i, pi_j, alpha)
                    loss = loss + a * _mmd2_rbf(Z_i, Z_j, sigmas=sigmas)
    return loss


def _sinkhorn(A, B, epsilon=0.1, n_iter=50):
    n, m = A.size(0), B.size(0)
    if n == 0 or m == 0:
        return A.new_zeros(())
    C = _pairwise_sq_dist(A, B)
    log_mu = -torch.log(A.new_tensor(float(n)))
    log_nu = -torch.log(A.new_tensor(float(m)))
    log_u = A.new_zeros(n)
    log_v = A.new_zeros(m)
    log_K = -C / epsilon
    for _ in range(n_iter):
        log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_nu - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
    log_gamma = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    gamma = log_gamma.exp()
    return (gamma * C).sum()


def loss_inv_OT(
    h_stack,
    pi,
    y,
    num_classes,
    domain_ids,
    num_domains,
    alpha=4.0,
    epsilon=0.1,
    sinkhorn_iters=50,
):
    B, M, r = h_stack.shape
    loss = h_stack.new_zeros(())
    for m in range(M):
        h_m = h_stack[:, m, :]
        pi_m = pi[:, m]
        for c in range(num_classes):
            mask_c = y == c
            if mask_c.sum() < 2:
                continue
            for i in range(num_domains):
                mask_i = mask_c & (domain_ids == i)
                if mask_i.sum() == 0:
                    continue
                Z_i = h_m[mask_i]
                pi_i = pi_m[mask_i]
                for j in range(i + 1, num_domains):
                    mask_j = mask_c & (domain_ids == j)
                    if mask_j.sum() == 0:
                        continue
                    Z_j = h_m[mask_j]
                    pi_j = pi_m[mask_j]
                    a = _routing_weight(pi_i, pi_j, alpha)
                    loss = loss + a * _sinkhorn(
                        Z_i, Z_j, epsilon=epsilon, n_iter=sinkhorn_iters
                    )
    return loss
