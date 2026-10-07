"""Fixed-map predictive compatibility, relative to row-bounded linear heads.

All optimization bounds are convex first-order certificates, not gradient-norm
heuristics. Statistical statements require fresh iid audit samples per domain.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CandidateMap(nn.Module):
    def __init__(self, input_dim, dimension=16, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dimension))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def project(w, radius):
    return w / (w.norm(dim=1, keepdim=True) / radius).clamp_min(1)


def risk_gradient(w, cell):
    z, y, weight = cell
    logits = z @ w.T
    logp = logits.log_softmax(1)
    risk = -(weight * logp[torch.arange(len(y)), y]).sum()
    residual = logp.exp()
    residual[torch.arange(len(y)), y] -= 1
    grad = (residual * weight[:, None]).T @ z
    return risk, grad


def make_cell(z, y, weight=None):
    z = torch.as_tensor(z, dtype=torch.float64, device='cpu')
    y = torch.as_tensor(y, dtype=torch.long, device='cpu')
    if not len(y) or z.ndim != 2 or len(z) != len(y):
        raise ValueError('Nonempty aligned features and labels required')
    if not torch.isfinite(z).all() or (z.norm(dim=1) > 1 + 1e-6).any():
        raise ValueError('Features must be finite and have norm <= 1')
    weight = (torch.ones(len(y), dtype=torch.float64) if weight is None
              else torch.as_tensor(weight, dtype=torch.float64))
    if weight.shape != y.shape or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError('Invalid sample weights')
    return z, y, weight / weight.sum()


def fw_gap(w, grad, radius):
    # min_V <grad,V> = -radius sum_c ||grad_c||_2.
    return max(0., float((w * grad).sum() + radius * grad.norm(dim=1).sum()))


def _optimize(cells, classes, radius, steps, tolerance, offsets=None):
    """Projected accelerated descent; every iterate supplies valid bounds.

    For minimax, smoothing affects search only. Bounds use the unsmoothed
    objective and a feasible simplex dual witness, including baseline errors.
    """
    if radius <= 0 or steps < 1 or classes < 2:
        raise ValueError('Positive radius/steps and at least two classes required')
    w = torch.zeros(classes, cells[0][0].shape[1], dtype=torch.float64)
    v, best_w = w.clone(), w.clone()
    upper, lower, lr, momentum = float('inf'), 0., 1., 1.
    is_max = offsets is not None
    if is_max:
        base_lo = torch.tensor([b['lower'] for b in offsets], dtype=torch.float64)
        base_hi = torch.tensor([b['upper'] for b in offsets], dtype=torch.float64)

    def objective(point, temperature):
        pairs = [risk_gradient(point, c) for c in cells]
        risks = torch.stack([p[0] for p in pairs])
        grads = torch.stack([p[1] for p in pairs])
        if is_max:
            regrets = risks - base_hi
            q = (regrets / temperature).softmax(0)
            value = temperature * torch.logsumexp(regrets / temperature, 0)
        else:
            q = torch.ones(len(cells), dtype=torch.float64) / len(cells)
            value = (q * risks).sum()
        return value, (q[:, None, None] * grads).sum(0), risks, q

    for iteration in range(steps):
        # Restart acceleration at each deterministic smoothing stage.
        stage = min(3, 4 * iteration // steps)
        temperature = [0.1, 0.03, 0.01, 0.003][stage]
        if iteration == 0 or stage != previous_stage:
            v, momentum = w.clone(), 1.
        previous_stage = stage
        value, grad, _, _ = objective(v, temperature)
        for _ in range(60):
            trial = project(v - lr * grad, radius)
            trial_value, _, _, _ = objective(trial, temperature)
            diff = trial - v
            if trial_value <= value + (grad * diff).sum() + diff.square().sum() / (2 * lr) + 1e-13:
                break
            lr *= .5
        else:
            raise RuntimeError('Convex solver line search failed')
        _, trial_grad, risks, q = objective(trial, temperature)
        if is_max:
            candidate_upper = max(0., float((risks - base_lo).max()))
            candidate_lower = float((q * (risks - base_hi)).sum()) - fw_gap(trial, trial_grad, radius)
        else:
            candidate_upper = float(risks.mean())
            candidate_lower = candidate_upper - fw_gap(trial, trial_grad, radius)
        if candidate_upper < upper:
            upper, best_w = candidate_upper, trial.clone()
        lower = max(lower, candidate_lower)
        if upper - lower <= tolerance:
            break
        new_momentum = (1 + math.sqrt(1 + 4 * momentum ** 2)) / 2
        v = trial + (momentum - 1) / new_momentum * (trial - w)
        w, momentum, lr = trial, new_momentum, min(lr * 1.05, 100.)
    if lower > upper + 1e-8:
        raise RuntimeError('Inconsistent optimization certificate')
    return best_w, dict(lower=lower, upper=upper, gap=max(0., upper-lower),
                        iterations=iteration+1, converged=upper-lower <= tolerance)


def solve_head(cell, classes, radius=1., steps=1500, tolerance=1e-5):
    return _optimize([cell], classes, radius, steps, tolerance)


def solve_common(cells, classes, radius=1., steps=1500, tolerance=1e-5):
    individual = [solve_head(c, classes, radius, steps, tolerance)[1] for c in cells]
    w, certificate = _optimize(cells, classes, radius, steps, tolerance, individual)
    return w, dict(certificate, individual=individual)


def uniform_risk_bound(n, classes, dimension, radius, alpha):
    """Covering-number + Hoeffding bound for a fixed map, uniform over heads.

    Parameter radius = radius*sqrt(C), dimension=C*d, CE Lipschitz sqrt(2).
    A deterministic net radius minimizes the bound without looking at data.
    """
    if n < 1 or not 0 < alpha < 1:
        raise ValueError('Invalid sample size/confidence')
    loss_bound = math.log(classes) + 2 * radius
    parameter_radius = radius * math.sqrt(classes)
    candidates = []
    for r in np.geomspace(1e-6, 2 * parameter_radius, 160):
        log_cover = classes * dimension * math.log1p(2 * parameter_radius / r)
        candidates.append(loss_bound * math.sqrt((math.log(2 / alpha) + log_cover) / (2*n))
                          + 2 * math.sqrt(2) * r)
    return float(min(loss_bound, min(candidates)))


def entropy_lower(counts, alpha):
    counts = np.asarray(counts, dtype=float)
    n, classes = counts.sum(), len(counts)
    p = counts / n
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    tv = min(1., .5 * classes * math.sqrt(math.log(2 * classes / alpha) / (2*n)))
    correction = math.log(classes)
    if tv < 1 - 1/classes:
        binary = -tv*math.log(tv) - (1-tv)*math.log1p(-tv) if tv else 0.
        correction = binary + tv*math.log(classes-1)
    return entropy, max(0., entropy-correction)


def audit_head(w, cells, counts, radius, tau, delta, alpha_per_event,
               steps=1500, tolerance=1e-5):
    """Certify the deployed, calibration-fitted head on independent audit data.

    Domain-optimum lower bounds include numerical error. Statistical regret
    upper bounds add 2*epsilon; no error is silently dropped or double counted.
    Weighted compressed cells require genuine iid counts for n and entropy.
    """
    if len(cells) != len(counts) or not cells:
        raise ValueError('Audit cells and class counts must align')
    if not torch.isfinite(w).all() or (w.norm(dim=1) > radius + 1e-8).any():
        raise ValueError('Audit head must be feasible in the declared class')
    rows = []
    for cell, count in zip(cells, counts):
        _, optimum = solve_head(cell, len(w), radius, steps, tolerance)
        risk = float(risk_gradient(w, cell)[0])
        n = int(np.sum(count))
        epsilon = uniform_risk_bound(n, len(w), w.shape[1], radius, alpha_per_event)
        entropy, h_lower = entropy_lower(count, alpha_per_event)
        regret_upper = max(0., risk-optimum['lower']) + 2*epsilon
        risk_upper = min(math.log(len(w))+2*radius, risk+epsilon)
        rows.append(dict(n=n, risk=risk, optimum=optimum, epsilon=epsilon,
                         empirical_regret_upper=max(0., risk-optimum['lower']),
                         regret_upper=regret_upper, entropy=entropy,
                         entropy_lower=h_lower, risk_upper=risk_upper,
                         compatible=regret_upper <= tau,
                         predictive=risk_upper <= h_lower-delta))
    accepted = all(r['compatible'] and r['predictive'] for r in rows)
    return dict(accepted=accepted, status='certified' if accepted else 'insufficient_evidence',
                domains=rows, tau=tau, delta=delta)


def gaussian_signal(means, covariance, support):
    """Analytic SSI existence diagnostic; distinct from neural compatibility."""
    values, vectors = np.linalg.eigh(np.asarray(covariance))
    if np.any(values <= 0):
        raise ValueError('Covariance must be positive definite')
    whiten = (vectors / np.sqrt(values)) @ vectors.T
    nu = np.asarray(means)[list(support)] @ whiten
    differences = nu[1:] - nu[0]
    if len(differences):
        _, s, vh = np.linalg.svd(differences, full_matrices=True)
        rank = np.count_nonzero(s > max(differences.shape)*np.finfo(float).eps*max(1., s[0]))
        null = vh[rank:]
        projection = null.T @ (null @ nu[0])
    else:
        projection = nu[0]
    gamma = float(np.linalg.norm(projection))
    return dict(gamma=gamma, optimal_error=.5*math.erfc(gamma/math.sqrt(2)),
                predictive=gamma > 1e-10)
