"""Shared selector and ridge functions for the ImageNet reproduction."""
import numpy as np
import torch


def unit(x):
    x = np.asarray(x)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def transform(x, basis, factors):
    return unit(x + ((x @ basis) * factors) @ basis.T)


def select_gap(scores, cap=10):
    owner = np.argmax(scores, axis=1)
    best = scores[np.arange(len(scores)), owner]
    margin = best - np.partition(scores, -2, axis=1)[:, -2]
    tolerance = 1e-12 * np.maximum(1., np.max(np.abs(scores), axis=1))
    selected = []
    for c in range(scores.shape[1]):
        ix = np.flatnonzero((owner == c) & (margin > tolerance))
        selected.append(ix[np.argsort(-margin[ix], kind='stable')[:cap]])
    return np.concatenate(selected).astype(np.int32), np.repeat(np.arange(scores.shape[1], dtype=np.int32), list(map(len, selected)))


def fixed_owner_select(base, scores):
    owner = base.argmax(axis=1)
    selected = []
    for c in range(scores.shape[1]):
        ix = np.flatnonzero(owner == c)
        if len(ix):
            margin = scores[ix, c] - np.max(np.delete(scores[ix], c, axis=1), axis=1)
            ix = ix[np.argsort(-margin, kind='stable')[:10]]
        selected.append(ix)
    return np.concatenate(selected).astype(np.int32), np.repeat(np.arange(scores.shape[1], dtype=np.int32), [len(ix) for ix in selected])


def gpu_ridge(sx, sy, tx, ix, ps, nclass):
    x = np.concatenate([unit(sx), tx[ix]]).astype(np.float64)
    y = np.eye(nclass, dtype=np.float64)[np.r_[sy, ps]]
    xm, ym = x.mean(0), y.mean(0)
    z = torch.as_tensor(x - xm, device='cuda', dtype=torch.float64)
    yy = torch.as_tensor(y - ym, device='cuda', dtype=torch.float64)
    w = torch.linalg.solve(z.T @ z + torch.eye(z.shape[1], device='cuda', dtype=torch.float64), z.T @ yy)
    my = torch.as_tensor(ym, device='cuda', dtype=torch.float64)
    pred = np.empty(len(tx), dtype=np.uint16)
    for lo in range(0, len(tx), 2048):
        q = torch.as_tensor(tx[lo:lo+2048] - xm, device='cuda', dtype=torch.float64)
        pred[lo:lo+2048] = (q @ w + my).argmax(1).cpu().numpy().astype(np.uint16)
    return pred
