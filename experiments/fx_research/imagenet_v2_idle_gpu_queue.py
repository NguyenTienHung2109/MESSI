"""Wait for an entirely idle GPU, then rerun the ImageNetV2 prompt audit."""
import os
from pathlib import Path
import subprocess
import sys
import time

RESULT = Path('results/reproduction_ood_table2/imagenetv2_clip_coop_prompt_audit.json')
ORDER = (1, 5, 3, 2, 0, 4, 6)


def available(order=ORDER):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                    '--format=csv,noheader'], text=True).splitlines()
    occupied = set(x.strip() for x in apps)
    idle = {}
    for row in rows.splitlines():
        index, uuid, memory, utilization = [x.strip() for x in row.split(',')]
        if uuid not in occupied and int(memory) <= 10 and int(utilization) == 0:
            idle[int(index)] = uuid
    return next((x for x in (*order, *sorted(idle)) if x in idle), None)


def main():
    while not RESULT.exists():
        # Leave CPU capacity for the resumable downloads and data preparation.
        if not Path('results/reproduction_ood_table2/downloads/SIZES_COMPLETE.json').exists():
            time.sleep(30)
            continue
        gpu = available()
        if gpu is None:
            print('No fully idle GPU; waiting.', flush=True)
            time.sleep(30)
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        print('Selected verified idle physical GPU', gpu, flush=True)
        guard = subprocess.Popen([sys.executable, '-u', '-m',
                                  'experiments.fx_research.imagenet_table2_exclusive_gpu'], env=env)
        try:
            run = subprocess.run([sys.executable, '-u', '-m',
                                  'experiments.fx_research.imagenet_table2_eval'], env=env)
        finally:
            guard.terminate()
            guard.wait()
        if run.returncode == 0 and RESULT.exists():
            return
        if run.returncode not in (130, -2):
            raise RuntimeError(f'Prompt audit failed: exit {run.returncode}')
        print('Interrupted because GPU became occupied; waiting for an idle GPU.', flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
