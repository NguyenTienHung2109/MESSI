"""Run frozen Ours and VLCS patch-local for ImageNet -> V2 / A.

ImageNet-A uses its official known 200-class output space, matching Table 2.
Target labels are read only after each seed's predictions are frozen.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn.functional as F

from experiments.fx_research.imagenet_ours_math import unit, gpu_ridge, select_gap, fixed_owner_select, transform
from experiments.fx_research.imagenet_ours_extract import BASE, OUT, sha


def support(seed, allowed, source_ids):
    lookup = {(Path(x).parent.name, Path(x).name): i for i, x in enumerate(source_ids)}
    root = BASE / f'data/imagenet-seed{seed}/imagenet/images/train'
    rows = []
    for wnid in allowed:
        paths = sorted((root / wnid).glob('*.JPEG'))
        if len(paths) != 16:
            raise RuntimeError(f'Support incomplete: {seed}, {wnid}')
        for path in paths:
            ordinal = int(path.stem.split('_')[-1])
            rows.append(lookup[(wnid, f'{ordinal:06}.JPEG')])
    return np.asarray(rows, dtype=np.int64)


def covariance_basis(source_top, nclass):
    # Eigen-decomposition of X^T X / N equals the SVD-based original transform.
    z = source_top.reshape(nclass, 16, -1)
    residual = z - z.mean(axis=1, keepdims=True)
    x = torch.as_tensor(residual.reshape(-1, 768), dtype=torch.float64, device='cuda')
    with torch.inference_mode():
        eig, vec = torch.linalg.eigh((x.T @ x) / len(x))
    return vec.cpu().numpy(), np.maximum(eig.cpu().numpy(), 0.0)


def predict(target, seed):
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not physical or not physical.isdigit() or torch.cuda.device_count() != 1:
        raise RuntimeError('Select exactly one idle physical GPU')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    source_dir, target_dir = OUT / 'source', OUT / target
    for folder in (source_dir, target_dir):
        if not (folder / 'FEATURES_COMPLETE.json').exists():
            raise RuntimeError(f'Feature bank incomplete: {folder}')
    out = OUT / 'predictions' / target
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f'seed{seed}.npz'
    freeze = out / f'seed{seed}_FROZEN.json'
    if freeze.exists():
        if sha(archive) != json.loads(freeze.read_text())['prediction_sha256']:
            raise RuntimeError('Frozen predictions changed')
        return
    all_ids = json.loads((source_dir / 'inventory.json').read_text())
    target_ids = json.loads((target_dir / 'inventory.json').read_text())
    wnids = json.loads((BASE / 'imagenet_mirror_index/sampling_plan.json').read_text())['class_names']
    allowed = wnids if target == 'v2' else sorted({Path(x).parent.name for x in target_ids})
    if len(allowed) != (1000 if target == 'v2' else 200) or not set(allowed) <= set(wnids):
        raise RuntimeError('Unexpected official target class space')
    nclass = len(allowed)
    sr = support(seed, allowed, all_ids)
    sy = np.repeat(np.arange(nclass, dtype=np.int64), 16)
    sc = unit(np.asarray(np.load(source_dir / 'cls.npy', mmap_mode='r')[sr]))
    sm = unit(np.asarray(np.load(source_dir / 'mean.npy', mmap_mode='r')[sr]))
    st = unit(np.asarray(np.load(source_dir / 'top.npy', mmap_mode='r')[sr]))
    tc = unit(np.load(target_dir / 'cls.npy', mmap_mode='r'))
    tm = unit(np.load(target_dir / 'mean.npy', mmap_mode='r'))

    # Ours: centered CLS cosine, top-two margin, <=10 pseudo/class, Ridge(alpha=1).
    proto = unit(sc.reshape(nclass, 16, 768).mean(axis=1))
    scores = (tc - tc.mean(axis=0)) @ proto.T
    ix0, labels0 = select_gap(scores)
    pred0 = gpu_ridge(sc, sy, tc, ix0, labels0, nclass)
    print(target, seed, 'Ours complete; pseudo', len(ix0), flush=True)

    # Frozen patch-local policy from transfer_vlcs_patchlocal_predict.
    basis, eig = covariance_basis(st, nclass)
    factors = 1 / np.sqrt(1 + 100 * eig) - 1
    sp = transform(st, basis, factors)
    tp = transform(tm, basis, factors)
    sx = np.concatenate((sc, sp), axis=1) / np.sqrt(2)
    tx = np.concatenate((tc, tp), axis=1) / np.sqrt(2)
    prototypes = unit(sx.reshape(nclass, 16, -1).mean(axis=1))
    base = tx @ prototypes.T
    source_raw = np.load(source_dir / 'patch.npy', mmap_mode='r')
    target_raw = np.load(target_dir / 'patch.npy', mmap_mode='r')
    with torch.inference_mode():
        src = F.normalize(torch.as_tensor(np.asarray(source_raw[sr]), device='cuda', dtype=torch.float32), dim=-1).reshape(-1, 768)
        parts = []
        for start in range(0, len(tc), 4):
            block = np.asarray(target_raw[start:start + 4]).copy()
            q = F.normalize(torch.as_tensor(block, device='cuda', dtype=torch.float32), dim=-1)
            sim = (q @ src.T).reshape(len(q), 196, nclass, 256).amax(-1)
            values, which = sim.topk(2, dim=2)
            other = torch.where(which[:, :, :1] == torch.arange(nclass, device='cuda')[None, None, :],
                                values[:, :, 1:2], values[:, :, 0:1])
            parts.append((sim - other).topk(16, dim=1).values.mean(1).cpu().numpy())
            if start % 256 == 0:
                print(target, seed, 'patch', start, '/', len(tc), flush=True)
        local = np.concatenate(parts)
    ix1, labels1 = fixed_owner_select(base, base + 0.5 * local)
    src_head = np.concatenate((sc, sm), axis=1) / np.sqrt(2)
    tgt_head = np.concatenate((tc, tm), axis=1) / np.sqrt(2)
    pred1 = gpu_ridge(src_head, sy, tgt_head, ix1, labels1, nclass)
    np.savez_compressed(archive, ours=pred0, patch_local=pred1, ours_selected=ix0,
                        ours_pseudo=labels0.astype(np.uint16), patch_selected=ix1,
                        patch_pseudo=labels1.astype(np.uint16))
    freeze.write_text(json.dumps({'target': target, 'seed': seed, 'shots': 16, 'nclass': nclass,
                                  'allowed_wnids': allowed, 'physical_gpu': int(physical),
                                  'prediction_sha256': sha(archive),
                                  'source_features_sha256': sha(source_dir / 'FEATURES_COMPLETE.json'),
                                  'target_features_sha256': sha(target_dir / 'FEATURES_COMPLETE.json'),
                                  'support_ordinals_sha256': hashlib.sha256(sr.tobytes()).hexdigest(),
                                  'target_labels_used_before_freeze': False,
                                  'support_original_filenames_available': False}, indent=2) + '\n')
    print(target, seed, 'predictions frozen', flush=True)


def evaluate(target):
    folder = OUT / 'predictions' / target
    ids = json.loads((OUT / target / 'inventory.json').read_text())
    records = []
    for seed in (1,):
        meta = json.loads((folder / f'seed{seed}_FROZEN.json').read_text())
        archive = folder / f'seed{seed}.npz'
        if sha(archive) != meta['prediction_sha256']:
            raise RuntimeError('Frozen prediction mismatch')
        allowed = meta['allowed_wnids']
        y = np.array([int(Path(x).parent.name) if target == 'v2'
                      else allowed.index(Path(x).parent.name) for x in ids], dtype=np.uint16)
        with np.load(archive) as z:
            for method, pred, sel, pseudo in (('ours', 'ours', 'ours_selected', 'ours_pseudo'),
                                              ('patch_local', 'patch_local', 'patch_selected', 'patch_pseudo')):
                chosen = z[sel]; lab = z[pseudo]
                records.append({'target': target, 'seed': seed, 'method': method, 'n': len(y),
                                'accuracy_pct': float(100 * np.mean(z[pred] == y)),
                                'pseudo_count': len(chosen),
                                'pseudo_precision_pct': float(100 * np.mean(y[chosen] == lab)) if len(chosen) else None})
    output = OUT / f'{target}_results.csv'
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    for method in ('ours', 'patch_local'):
        values = [r['accuracy_pct'] for r in records if r['method'] == method]
        print(target, method, 'accuracy', mean(values), 'sd', stdev(values) if len(values) > 1 else 'N/A (one seed)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target', choices=('v2', 'a'))
    parser.add_argument('--seed', type=int, choices=(1, 2, 3))
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.evaluate:
        evaluate(args.target)
    elif args.seed:
        predict(args.target, args.seed)
    else:
        parser.error('--seed is required for prediction')
