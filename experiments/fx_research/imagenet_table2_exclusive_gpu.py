"""Interrupt only our ImageNet evaluation if GPU 5 becomes shared."""
import os
from pathlib import Path
import signal
import subprocess
import time

physical = os.environ.get('CUDA_VISIBLE_DEVICES')
if not physical or not physical.isdigit():
    raise RuntimeError('Select exactly one physical GPU')
devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
UUID = next(row.split(',')[1].strip() for row in devices.splitlines() if row.split(',')[0].strip() == physical)
module = os.environ.get('TABLE2_GUARDED_MODULE', 'experiments.fx_research.imagenet_table2_eval')
done = Path(os.environ.get('TABLE2_DONE_PATH', 'results/reproduction_ood_table2/imagenetv2_clip_coop_prompt_audit.json'))
while True:
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                    '--format=csv,noheader'], text=True)
    ours, others = [], []
    for row in rows.splitlines():
        fields = [part.strip() for part in row.split(',')]
        if len(fields) != 2 or fields[1] != UUID:
            continue
        pid = int(fields[0])
        try:
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        except (FileNotFoundError, PermissionError):
            cmd = ''
        (ours if module in cmd else others).append(pid)
    if ours and others:
        for pid in ours:
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        print(f'Stopped our evaluation because another compute process joined GPU {physical}.', flush=True)
        break
    if done.exists():
        break
    time.sleep(5)
