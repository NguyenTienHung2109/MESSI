"""Materialize planned ImageNet support through dataset-server image assets.

Dataset-server returns re-encoded JPEGs, not the original bytes. All methods
must use the same materialized support and report this provenance.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import threading
import time

import requests

BASE = Path('results/reproduction_ood_table2')
DATA = BASE / 'data'
API = 'https://datasets-server.huggingface.co/rows'
REPO = 'mrm8488/ImageNet1K-train'
LOCAL = threading.local()
SEEDS = ('1',)
GATE = threading.Lock()
NEXT_REQUEST = 0.0


def wait_turn():
    global NEXT_REQUEST
    while True:
        with GATE:
            now = time.monotonic()
            if now >= NEXT_REQUEST:
                NEXT_REQUEST = now + 0.5
                return
            delay = NEXT_REQUEST - now
        time.sleep(delay)


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    return LOCAL.session


def get(url, **kwargs):
    global NEXT_REQUEST
    attempt = 0
    while True:
        try:
            wait_turn()
            response = session().get(url, timeout=(20, 90), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            code = getattr(getattr(exc, 'response', None), 'status_code', None)
            print('Retry', type(exc).__name__, 'HTTP', code, 'attempt', attempt + 1, flush=True)
            if code == 429:
                raw = exc.response.headers.get('Retry-After', '')
                delay = max(300, int(raw)) if raw.isdigit() else 300
                with GATE:
                    NEXT_REQUEST = max(NEXT_REQUEST, time.monotonic() + delay)
            time.sleep(min(60, 2 + 3 * attempt))
            attempt += 1


def plans():
    plan = json.loads((BASE / 'imagenet_mirror_index/sampling_plan.json').read_text())
    classes, selected = plan['class_names'], plan['selected_row_ordinals']
    positions = {}
    for seed in SEEDS:
        rng = random.Random(int(seed))
        for label, wnid in enumerate(classes):
            for ordinal, slot in zip(selected[seed][wnid], rng.sample(range(16), 16)):
                positions[(seed, label, ordinal)] = slot
    jobs = {}
    offset = 0
    for label, wnid in enumerate(classes):
        ordinals = set().union(*(selected[seed][wnid] for seed in SEEDS))
        for ordinal in sorted(ordinals):
            row = offset + ordinal
            jobs.setdefault(row // 100 * 100, []).append((row, label, ordinal, wnid))
        offset += plan['class_counts'][str(label)]
    return plan, positions, jobs


def block(start, entries, positions):
    pending = [(row, label, ordinal, wnid) for row, label, ordinal, wnid in entries
               if not (DATA / 'imagenet-common/train' / wnid / f'{ordinal:06}.JPEG').exists()]
    rows = {}
    if pending:
        params = {'dataset': REPO, 'config': 'default', 'split': 'train', 'offset': start, 'length': 100}
        body = get(API, params=params).json()
        rows = {entry['row_idx']: entry['row'] for entry in body['rows']}
    for row, label, ordinal, wnid in entries:
        folder = DATA / 'imagenet-common/train' / wnid
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f'{ordinal:06}.JPEG'
        if not dest.exists():
            item = rows[row]
            if item['label'] != label:
                raise RuntimeError(f'Label mismatch at row {row}')
            image = get(item['image']['src']).content
            if len(image) < 4 or image[:2] != b'\xff\xd8' or image[-2:] != b'\xff\xd9':
                raise RuntimeError(f'Invalid JPEG at row {row}')
            tmp = dest.with_suffix('.tmp')
            tmp.write_bytes(image)
            tmp.replace(dest)
        for seed in SEEDS:
            slot = positions.get((seed, label, ordinal))
            if slot is None:
                continue
            link = DATA / f'imagenet-seed{seed}/imagenet/images/train' / wnid / f'{slot:02}_{ordinal:06}.JPEG'
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                os.link(dest, link)
    return len(entries)


def main():
    if (BASE / 'train_support_complete.json').exists():
        print('API support already complete', flush=True)
        return
    plan, positions, jobs = plans()
    print('API JPEG support:', sum(map(len, jobs.values())), 'unique images,', len(jobs), 'API blocks', flush=True)
    completed, images = 0, 0
    def check_quota():
        quota_root = os.environ.get('IMAGENET_QUOTA_ROOT', str(Path.home()))
        used = int(subprocess.check_output(['du', '-sb', quota_root]).split()[0])
        if used >= 285_000_000_000:
            raise RuntimeError(f'Storage safety threshold exceeded: {used} bytes')
    check_quota()
    # Four I/O workers leave CPU capacity for the existing queue and monitors.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(block, key, entries, positions) for key, entries in sorted(jobs.items())]
        for future in as_completed(futures):
            images += future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(jobs):
                progress = {'completed_blocks': completed, 'total_blocks': len(jobs),
                            'images_verified': images, 'unique_images': sum(map(len, jobs.values()))}
                tmp = BASE / 'train_api_progress.tmp'
                tmp.write_text(json.dumps(progress) + '\n')
                tmp.replace(BASE / 'train_api_progress.json')
                print('API blocks', completed, '/', len(jobs), 'images', images, flush=True)
            if completed % 200 == 0:
                check_quota()
    for seed in SEEDS:
        for wnid in plan['class_names']:
            if len(list((DATA / f'imagenet-seed{seed}/imagenet/images/train' / wnid).glob('*.JPEG'))) != 16:
                raise RuntimeError(f'Incomplete support: seed={seed}, class={wnid}')
    digest = hashlib.sha256((BASE / 'imagenet_mirror_index/sampling_plan.json').read_bytes()).hexdigest()
    (BASE / 'train_support_complete.json').write_text(json.dumps({
        'source': 'Hugging Face datasets-server cached-assets (re-encoded JPEG)',
        'original_jpeg_bytes_preserved': False, 'support_original_filenames_available': False,
        'seeds': [int(seed) for seed in SEEDS], 'shots': 16, 'unique_images': images, 'sampling_plan_sha256': digest,
    }, indent=2) + '\n')
    print('API support complete', flush=True)


if __name__ == '__main__':
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    main()
