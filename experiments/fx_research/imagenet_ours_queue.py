"""Resume the ImageNet Ours comparison when inputs and an exclusive GPU are ready."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.fx_research.imagenet_v2_idle_gpu_queue import available

BASE = Path('results/reproduction_ood_table2')
OUT = BASE / 'ours_imagenet16'


def own_gpu_count():
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                    '--format=csv,noheader'], text=True)
    uuids = set()
    for row in rows.splitlines():
        fields = [x.strip() for x in row.split(',')]
        if len(fields) != 2:
            continue
        try:
            if Path('/proc', fields[0]).stat().st_uid == os.getuid():
                uuids.add(fields[1])
        except FileNotFoundError:
            pass
    return len(uuids)


def shared_gpu():
    """Only share GPU IDs explicitly configured by the user."""
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in rows.splitlines()}
    allowed = [int(x) for x in os.environ.get('IMAGENET_SHARED_GPUS', '').split(',') if x.strip()]
    return next((gpu for gpu in allowed if free.get(gpu, 0) >= 12000), None)


def execute(module, args, done):
    while not done.exists():
        # Serialize our GPU tasks to avoid contention for the user's CPU/RAM quota.
        gpu = None
        if own_gpu_count() == 0:
            gpu = available()
            if gpu is None:
                gpu = shared_gpu()
        if gpu is None:
            print('Waiting for an idle GPU or sufficient free VRAM on GPU 2/4.', flush=True)
            time.sleep(30)
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   ARROW_NUM_THREADS='1', TABLE2_GUARDED_MODULE=module,
                   TABLE2_DONE_PATH=str(done))
        print('Starting', module, *args, 'on physical GPU', gpu, flush=True)
        # The exclusive-GPU guard is inappropriate on GPUs the user allowed us to share.
        # Keep it on other GPUs, which must still be completely idle.
        shared = {int(x) for x in os.environ.get('IMAGENET_SHARED_GPUS', '').split(',') if x.strip()}
        guard = None if gpu in shared else subprocess.Popen(
            [sys.executable, '-u', '-m', 'experiments.fx_research.imagenet_table2_exclusive_gpu'], env=env)
        try:
            run = subprocess.run([sys.executable, '-u', '-m', module, *args], env=env)
        finally:
            if guard:
                guard.terminate()
                guard.wait()
        if run.returncode == 0 and done.exists():
            return
        if run.returncode not in (130, -2):
            raise RuntimeError(f'{module} {args} failed with exit {run.returncode}')
        print('GPU became shared; retrying when idle.', flush=True)
        time.sleep(30)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for target in ('v2', 'a'):
        execute('experiments.fx_research.imagenet_ours_extract', [target],
                OUT / target / 'FEATURES_COMPLETE.json')
    progress = BASE / 'train_materialize_progress.json'
    while not (BASE / 'train_support_complete.json').exists() and (not progress.exists() or json.loads(progress.read_text())['next_shard'] != 296):
        print('Waiting for all 16-shot source images from the ImageNet API.', flush=True)
        time.sleep(60)
    execute('experiments.fx_research.imagenet_ours_extract', ['source'],
            OUT / 'source' / 'FEATURES_COMPLETE.json')
    for target in ('v2', 'a'):
        for seed in (1,):
            execute('experiments.fx_research.imagenet_ours_predict',
                    [target, '--seed', str(seed)],
                    OUT / 'predictions' / target / f'seed{seed}_FROZEN.json')
        subprocess.run([sys.executable, '-u', '-m', 'experiments.fx_research.imagenet_ours_predict',
                        target, '--evaluate'], check=True)
    print('Ours ImageNetV2 / ImageNet-A comparison complete.', flush=True)


if __name__ == '__main__':
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:8])
    main()
