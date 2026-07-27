# domainbed/losses/cirl_utils.py
"""CIRL Fourier mixing, feature masking, and factorization utilities."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ImageNet normalization values used around the Fourier transform.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return x * std + mean


def _normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def fourier_amplitude_mix_batched(
    x: torch.Tensor,
    alpha: float = 1.0,
    ratio: float = 1.0,
) -> torch.Tensor:
    """
    Args:
        x: (B, C, H, W) normalized image tensor (ImageNet stats).
        alpha: upper bound of the per-sample mixing coefficient lam ~ U(0, alpha).
            CIRL's PACS default is 1.0.
        ratio: fraction of the (centered) amplitude spectrum to mix.
            CIRL's default is 1.0 (mix the whole spectrum).

    Returns:
        x_aug: (B, C, H, W) same-shape augmented batch where each sample i has
            had its amplitude spectrum mixed with sample perm[i]'s, while
            keeping its own phase. The label of sample i is preserved.
    """
    if x.dim() != 4:
        raise ValueError(f'expected 4-D (B,C,H,W), got {tuple(x.shape)}')

    B, C, H, W = x.shape
    if B < 2:
        return x.clone()

    # Select a different partner for each image.
    perm = torch.randperm(B, device=x.device)
    same = (perm == torch.arange(B, device=x.device))
    if same.any():
        # Avoid self-pairing.
        perm[same] = (perm[same] + 1) % B

    # Convert normalized inputs to non-negative image values.
    x_img = _denormalize(x).clamp(0.0, 1.0)
    x_partner = x_img[perm]

    # Transform both batches to the frequency domain.
    fft_a = torch.fft.fft2(x_img, dim=(-2, -1))
    fft_b = torch.fft.fft2(x_partner, dim=(-2, -1))

    abs_a = fft_a.abs()
    abs_b = fft_b.abs()
    pha_a = torch.angle(fft_a)

    # Mix the centered low-frequency block.
    abs_a_c = torch.fft.fftshift(abs_a, dim=(-2, -1))
    abs_b_c = torch.fft.fftshift(abs_b, dim=(-2, -1))

    h_crop = max(1, int(H * math.sqrt(ratio)))
    w_crop = max(1, int(W * math.sqrt(ratio)))
    h0 = H // 2 - h_crop // 2
    w0 = W // 2 - w_crop // 2

    # Draw one broadcastable mixing coefficient per sample.
    lam = torch.rand(B, 1, 1, 1, device=x.device, dtype=x.dtype) * alpha

    abs_mixed = abs_a_c.clone()
    block_a = abs_a_c[..., h0:h0 + h_crop, w0:w0 + w_crop]
    block_b = abs_b_c[..., h0:h0 + h_crop, w0:w0 + w_crop]
    abs_mixed[..., h0:h0 + h_crop, w0:w0 + w_crop] = (
        lam * block_b + (1.0 - lam) * block_a
    )

    # Restore the original phase and return to image space.
    abs_mixed = torch.fft.ifftshift(abs_mixed, dim=(-2, -1))
    fft_mixed = abs_mixed * torch.exp(1j * pha_a)
    x_aug_img = torch.fft.ifft2(fft_mixed, dim=(-2, -1)).real
    x_aug_img = x_aug_img.clamp(0.0, 1.0)

    return _normalize(x_aug_img)


# Differentiable top-k feature selector.

class Masker(nn.Module):
    """Learn an approximately k-hot feature mask with Gumbel softmax."""

    def __init__(
        self,
        in_dim: int,
        k: int | None = None,
        middle_ratio: int = 4,
        dropout: float = 0.5,
        tau: float = 0.5,
    ):
        super().__init__()
        if k is None:
            k = max(1, int(round(0.6 * in_dim)))
        if not (1 <= k < in_dim):
            raise ValueError(f'k={k} must satisfy 1 <= k < in_dim={in_dim}')

        self.in_dim = in_dim
        self.k = k
        self.tau = tau
        middle = middle_ratio * in_dim

        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(middle, in_dim),
        )
        self.bn = nn.BatchNorm1d(in_dim, affine=False)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Return an approximately k-hot mask for ``[B, in_dim]`` features."""
        score = self.bn(self.mlp(f))
        mask = torch.zeros_like(score)
        # Repeated samples allow different dimensions to be selected.
        cur = score
        for _ in range(self.k):
            soft = F.gumbel_softmax(cur, tau=self.tau, hard=False, dim=-1)
            mask = torch.maximum(mask, soft)
        return mask


# Feature-factorization loss.

def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Flatten the off-diagonal entries of a square matrix."""
    n, m = x.shape
    assert n == m, f'expected square matrix, got {(n, m)}'
    # The classic flatten-pop-reshape trick from the Barlow-Twins repo.
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def factorization_loss(
    f_a: torch.Tensor,
    f_b: torch.Tensor,
    off_diag_weight: float = 5e-3,
) -> torch.Tensor:
    """Compute the Barlow Twins cross-correlation objective."""
    if f_a.shape != f_b.shape:
        raise ValueError(f'shape mismatch: {f_a.shape} vs {f_b.shape}')

    # Per-dim z-score over the batch.
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0) + 1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0) + 1e-6)

    # Cross-correlation: (D, D)
    c = (f_a_norm.T @ f_b_norm) / f_a_norm.size(0)

    on_diag = (c.diagonal() - 1.0).pow(2).mean()
    off_diag = _off_diagonal(c).pow(2).mean()

    return on_diag + off_diag_weight * off_diag


# Factorization-weight schedule.

def sigmoid_rampup(current: float, rampup_length: float) -> float:
    """Ramp a weight from zero to one over the requested interval."""
    if rampup_length <= 0:
        return 1.0
    current = float(np.clip(current, 0.0, rampup_length))
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))