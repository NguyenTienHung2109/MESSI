import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed import vision_transformer as vit_module


class ExplicitMoEHead(nn.Module):
    def __init__(self, in_dim, expert_dim, num_experts, num_classes):
        super().__init__()
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, expert_dim),
                    nn.GELU(),
                    nn.Linear(expert_dim, expert_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.router = nn.Linear(in_dim, num_experts, bias=True)
        self.classifier = nn.Linear(expert_dim, num_classes)

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


class LoadBalanceLoss(nn.Module):
    def __init__(self, num_experts, beta=0.99):
        super().__init__()
        self.num_experts = num_experts
        self.beta = beta
        self.register_buffer("ema_pi", torch.full((num_experts,), 1.0 / num_experts))
        self.register_buffer("initialised", torch.tensor(False))

    def forward(self, pi):
        M = pi.size(1)
        batch_mean = pi.mean(dim=0)
        if self.training:
            if not self.initialised:
                self.ema_pi.copy_(batch_mean.detach())
                self.initialised.fill_(True)
            else:
                self.ema_pi.mul_(self.beta).add_(
                    batch_mean.detach(), alpha=1.0 - self.beta
                )
        hat_pi = self.beta * self.ema_pi + (1.0 - self.beta) * batch_mean
        return ((hat_pi - 1.0 / M) ** 2).sum()


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


def _energy_distance(A, B):
    if A.size(0) < 2 or B.size(0) < 2:
        return A.new_zeros(())
    d_AA = _pairwise_sq_dist(A, A).clamp_min(1e-12).sqrt()
    d_BB = _pairwise_sq_dist(B, B).clamp_min(1e-12).sqrt()
    d_AB = _pairwise_sq_dist(A, B).clamp_min(1e-12).sqrt()
    n, m = A.size(0), B.size(0)
    mean_AA = (d_AA.sum() - d_AA.diag().sum()) / (n * (n - 1))
    mean_BB = (d_BB.sum() - d_BB.diag().sum()) / (m * (m - 1))
    mean_AB = d_AB.mean()
    return 2.0 * mean_AB - mean_AA - mean_BB


def loss_inv_ED(h_stack, pi, y, num_classes, domain_ids, num_domains, alpha=4.0):
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
                    loss = loss + a * _energy_distance(Z_i, Z_j)
    return loss
