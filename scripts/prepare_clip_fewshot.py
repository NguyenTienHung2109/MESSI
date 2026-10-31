"""Build the CLIP caches used by the single-source few-shot OOD runner."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from domainbed.fewshot.protocols import SourceRecord, build_manifest, candidate_pool, folds, save_manifest
from scripts.verify_fewshot_data import DATASETS, IMAGE_SUFFIXES, audit_dataset

CLIP_SHA = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
TEMPLATES = ('a photo of a {}.', 'a good photo of a {}.', 'a bad photo of a {}.',
             'a close-up photo of a {}.', 'a cropped photo of a {}.', 'an image of a {}.',
             'a picture of a {}.', 'a rendering of a {}.')
ImageFile.LOAD_TRUNCATED_IMAGES = True

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=DATASETS)
    p.add_argument('--data-dir', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4, 8, 16), default=[1, 2, 4, 8, 16])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--views', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda:0', help='Logical device after setting CUDA_VISIBLE_DEVICES')
    p.add_argument('--allow-nonstandard-dataset', action='store_true', help='Smoke tests only; results are not paper benchmarks')
    a = p.parse_args()
    if a.views < 2 or a.batch_size < 1:
        p.error('Need at least two views and batch size >= 1')
    if digest(a.checkpoint) != CLIP_SHA:
        raise RuntimeError('Expected the official OpenAI CLIP ViT-B/16 checkpoint SHA256')
    cache = ROOT / 'results/fewshot_ood/features' / a.dataset / 'clip_vit_b16_224'
    views_dir = ROOT / 'results/fewshot_ood/support_views' / a.dataset / 'clip_vit_b16'
    if cache.exists() or views_dir.exists():
        raise RuntimeError('Cache directory already exists; use a fresh worktree or move the old cache aside')
    report = audit_dataset(a.data_dir, a.dataset, hash_content=(a.dataset == 'OfficeHome'))
    if not report.get('present'):
        raise RuntimeError(f'Dataset directory is missing: {a.data_dir / DATASETS[a.dataset][0]}')
    if not report['valid'] and not a.allow_nonstandard_dataset:
        raise RuntimeError(f"Dataset audit failed: {report['errors']}")
    import clip
    import torch
    from torchvision import transforms
    torch.set_num_threads(1)
    model, preprocess = clip.load(str(a.checkpoint), device=a.device, jit=False)
    model.eval().requires_grad_(False)
    augment = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    root = a.data_dir / DATASETS[a.dataset][0]
    domains, classes = sorted(report['domains']), report['canonical_classes']
    excluded = {Path(x).relative_to(DATASETS[a.dataset][0]).as_posix() for x in report['leakage_exclusions']}
    paths, labels, envs = [], [], []
    for d, domain in enumerate(domains):
        for c, category in enumerate(classes):
            for path in sorted((root / domain / category).rglob('*')):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    paths.append(path.relative_to(root).as_posix()); labels.append(c); envs.append(d)
    paths, labels, envs = np.array(paths), np.array(labels), np.array(envs)
    selected = set()
    for seed in a.seeds:
        for fold in folds(domains, 'single_source'):
            records = [SourceRecord(str(paths[i]), domains[int(envs[i])], int(labels[i]))
                       for i in range(len(paths)) if domains[int(envs[i])] in fold.source_domains and paths[i] not in excluded]
            pool = candidate_pool(records, fold.source_domains, len(classes), seed)
            for k in a.shots:
                try:
                    manifest = build_manifest(a.dataset, fold, pool, len(classes), k, 'total', seed)
                except ValueError as exc:
                    if not str(exc).startswith('Infeasible:'):
                        raise
                    print(f'Skipping infeasible source: {fold.identifier}, K={k}, seed={seed}: {exc}', flush=True)
                    continue
                save_manifest(ROOT / 'results/fewshot_ood/manifests' / a.dataset / 'single_source/total' / str(k) / str(seed) / f'{fold.identifier}.json', manifest)
                selected.update(x['sample_id'] for x in manifest['support'])
    cache.mkdir(parents=True); views_dir.mkdir(parents=True)
    started = time.monotonic()
    def encode_items(items, transform, output_path, augmented=False, shape=None):
        count = len(items)
        if not count:
            np.save(output_path, np.empty(shape or (0, 512), dtype=np.float32))
            return
        feature = np.lib.format.open_memmap(output_path, mode='w+', dtype='float32',
                                           shape=shape or (count, 512))
        flat_feature = feature.reshape(count, 512)
        for start in range(0, count, a.batch_size):
            batch = []
            for item in items[start:start+a.batch_size]:
                sample, view = item if augmented else (item, None)
                if augmented:
                    value = hashlib.sha256(f'fewshot_views_v1\0{sample}\0{view}'.encode()).digest()
                    torch.manual_seed(int.from_bytes(value[:8], 'big') % (2**63))
                with Image.open(root / sample) as image:
                    batch.append(transform(image.convert('RGB')))
            with torch.inference_mode():
                flat_feature[start:start+len(batch)] = model.encode_image(torch.stack(batch).to(a.device)).float().cpu().numpy()
        feature.flush()
        del flat_feature, feature
    print(f'Encoding {len(paths)} images and {len(selected)} x {a.views} support views', flush=True)
    np.savez_compressed(cache / 'inventory.npz', sample_id=paths, label=labels.astype('int16'), environment=envs.astype('int8'))
    encode_items(paths, preprocess, cache / 'image_features.npy')
    prompts = [[t.format(c.replace('_', ' ').lower()) for t in TEMPLATES] for c in classes]
    def text_encode(values):
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(values), a.batch_size):
                tokens = clip.tokenize(values[start:start+a.batch_size]).to(a.device)
                chunks.append(model.encode_text(tokens).float().cpu().numpy())
        return np.concatenate(chunks)
    raw = text_encode([row[0] for row in prompts])
    prompt_bank = text_encode([x for row in prompts for x in row])
    normal = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    np.save(cache / 'text_features.npy', normal(raw)); np.save(cache / 'raw_text_features.npy', raw)
    np.save(cache / 'prompt_features.npy', normal(prompt_bank).reshape(len(classes), len(TEMPLATES), -1))
    base = {'classes': classes, 'checkpoint_sha256': CLIP_SHA}
    for stem in ('text', 'raw_text', 'prompt'):
        filename = stem + '_features.npy'
        write_json(cache / (stem + '_metadata.json'), {**base, stem + '_features_sha256': digest(cache / filename)})
    ordered = sorted(selected)
    np.savez_compressed(views_dir / 'inventory.npz', sample_id=np.array(ordered))
    encode_items([(sample, v) for sample in ordered for v in range(a.views)], augment,
                 views_dir / 'support_views.npy', True, (len(ordered), a.views, 512))
    write_json(views_dir / 'complete.json', {'checkpoint_sha256': CLIP_SHA, 'n_views': a.views,
        'support_views_sha256': digest(views_dir / 'support_views.npy'),
        'rank_progress': [{'elapsed_seconds': time.monotonic()-started}]})
    write_json(cache / 'complete.json', {'domains': domains, **base, 'n': len(paths),
        'inventory_sha256': digest(cache / 'inventory.npz'), 'image_features_sha256': digest(cache / 'image_features.npy'),
        'elapsed_seconds': time.monotonic()-started, 'smoke_only': bool(a.allow_nonstandard_dataset)})
    audit_path = ROOT / 'results/fewshot_ood/data_audit.json'
    audits = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    audits[a.dataset] = report
    write_json(audit_path, audits)
    print(f'Cache ready: {cache}; GPU mapping CUDA_VISIBLE_DEVICES={__import__("os").environ.get("CUDA_VISIBLE_DEVICES", "unset")}', flush=True)

if __name__ == '__main__':
    main()
