"""Extract DINOv3 descriptors for ImageNet 16-shot -> V2 / A.

The full target patch bank is temporary. The source retains only 16 patches
per image selected by CLS similarity. Images use the same 224 resize as the
existing Ours/patch-local transfer experiments. Labels are not consumed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
import timm
from PIL import Image, ImageFile
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CHECKPOINT = Path(os.environ.get('DINOV3_CHECKPOINT', 'results/representation_study/checkpoints/dinov3_b16_timm/model.safetensors'))
MODEL_NAME = 'vit_base_patch16_dinov3.lvd1689m'

BASE = Path('results/reproduction_ood_table2')
OUT = BASE / 'ours_imagenet16'
CAP = 285_000_000_000


def source_items():
    p = BASE / 'train_materialize_progress.json'
    if not (BASE / 'train_support_complete.json').exists() and (not p.exists() or json.loads(p.read_text())['next_shard'] != 296):
        raise RuntimeError('ImageNet 16-shot support is not ready')
    root = BASE / 'data/imagenet-common/train'
    items = sorted(p for p in root.glob('*/*.JPEG') if p.is_file())
    if len(items) < 16000:
        raise RuntimeError(f'Incomplete source inventory: {len(items)}')
    return items


def target_items(target):
    folder, image_dir, expected = {
        'v2': ('imagenetv2', 'imagenetv2-matched-frequency-format-val', 10000),
        'a': ('imagenet-adversarial', 'imagenet-a', 7500),
    }[target]
    root = BASE / 'data' / folder
    if not (root / 'REPRODUCTION_DATA_COMPLETE.json').exists():
        raise RuntimeError(f'Target not prepared: {target}')
    images = root / image_dir
    items = sorted(p for p in images.glob('*/*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if len(items) != expected:
        raise RuntimeError(f'Unexpected target count for {target}: {len(items)}')
    return items


class Images(Dataset):
    def __init__(self, paths, start):
        self.paths, self.start = paths, start
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    def __len__(self):
        return len(self.paths) - self.start

    def __getitem__(self, i):
        idx = self.start + i
        with Image.open(self.paths[idx]) as image:
            return self.transform(image.convert('RGB')), idx


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def main(part):
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not gpu or not gpu.isdigit() or torch.cuda.device_count() != 1:
        raise RuntimeError('Select exactly one physically idle GPU')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    paths = source_items() if part == 'source' else target_items(part)
    folder = OUT / part
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'FEATURES_COMPLETE.json'
    if done.exists():
        print(part, 'already completed', flush=True)
        return
    inventory = folder / 'inventory.json'
    listed = [str(x) for x in paths]
    if inventory.exists():
        if json.loads(inventory.read_text()) != listed:
            raise RuntimeError('Inventory changed since extraction began')
    else:
        inventory.write_text(json.dumps(listed) + '\n')
    n, n_patch = len(paths), 16 if part == 'source' else 196
    specs = {'cls': ((n, 768), np.float32), 'mean': ((n, 768), np.float32),
             'top': ((n, 768), np.float32), 'patch': ((n, n_patch, 768), np.float16)}
    missing = sum(np.prod(shape) * np.dtype(dtype).itemsize for key, (shape, dtype) in specs.items()
                  if not (folder / (key + '.npy')).exists())
    quota_root = os.environ.get('IMAGENET_QUOTA_ROOT', str(Path.home()))
    used = int(subprocess.check_output(['du', '-sb', quota_root]).split()[0])
    if used + missing >= CAP:
        raise RuntimeError(f'Not enough storage headroom: {used}+{missing}')
    for key, (shape, dtype) in specs.items():
        path = folder / (key + '.npy')
        if not path.exists():
            np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape).flush()
    arrays = {key: np.load(folder / (key + '.npy'), mmap_mode='r+') for key in specs}
    progress = folder / 'progress.json'
    start = json.loads(progress.read_text())['done'] if progress.exists() else 0
    count = start
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, img_size=224, drop_path_rate=.1)
    model.load_state_dict(load_file(str(CHECKPOINT)), strict=True)
    model.cuda().eval().requires_grad_(False)
    if model.num_prefix_tokens != 5:
        raise RuntimeError('Unexpected DINOv3 token layout')
    loader = DataLoader(Images(paths, start), batch_size=64, num_workers=0, pin_memory=True)
    for images, rows in loader:
        with torch.inference_mode():
            tokens = model.forward_features(images.cuda(non_blocking=True))
            cls = tokens[:, 0].float()
            patch = torch.nn.functional.normalize(tokens[:, 5:].float(), dim=-1)
            u = torch.nn.functional.normalize(cls, dim=-1)
            ind = torch.argsort((patch * u[:, None]).sum(-1), descending=True, stable=True)[:, :16]
            chosen = patch.gather(1, ind[:, :, None].expand(-1, -1, 768))
            top = torch.nn.functional.normalize(chosen.mean(1), dim=-1)
            mean = torch.nn.functional.normalize(patch.mean(1), dim=-1)
        ids = rows.numpy()
        arrays['cls'][ids] = cls.cpu().numpy()
        arrays['mean'][ids] = mean.cpu().numpy()
        arrays['top'][ids] = top.cpu().numpy()
        arrays['patch'][ids] = (chosen if part == 'source' else patch).cpu().numpy().astype(np.float16)
        count = int(ids[-1]) + 1
        if count % 1024 < 64 or count == n:
            for array in arrays.values():
                array.flush()
            tmp = progress.with_suffix('.tmp')
            tmp.write_text(json.dumps({'done': count}) + '\n'); tmp.replace(progress)
            print(part, count, '/', n, 'physical GPU', gpu, flush=True)
    if n and count != n:
        raise RuntimeError('Feature extraction incomplete')
    for array in arrays.values():
        array.flush()
    done.write_text(json.dumps({'part': part, 'n': n, 'resolution': 224,
                                'gpu_physical': int(gpu), 'checkpoint_sha256': sha(Path(CHECKPOINT)),
                                'inventory_sha256': sha(inventory),
                                'features_sha256': {key: sha(folder / (key + '.npy')) for key in specs},
                                'target_labels_used_for_features': False}, indent=2) + '\n')
    print(part, 'feature bank complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=('source', 'v2', 'a'))
    main(parser.parse_args().part)
