"""Frozen PACS protocol for the router-independent predictive hinge revision."""
import argparse
from copy import deepcopy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


def experiment_arms(base):
    learned = deepcopy(base)
    # First isolate router removal while preserving the old neural objective,
    # coefficient and warm-up. Then isolate hinge+mixture before scheduling.
    decoupled = deepcopy(base)
    decoupled['hparams'].update(support_predictive_objective='max_ce',
                                support_lambda_adm=math.log(7) + 10,
                                support_adm_ramp_steps=0)
    matched = deepcopy(base)
    matched['hparams'].update(support_lambda_adm=math.log(7) + 10,
                              support_adm_ramp_steps=0)
    return {
        'v2_learned': (learned, 'learned'),
        'decoupled_max': (decoupled, 'learned'),
        'hinge_matched_adm': (matched, 'learned'),
        'v2_no_discrepancy': (deepcopy(base), 'no_discrepancy'),
        'v2_fixed': (deepcopy(base), 'fixed'),
        'v2_random': (deepcopy(base), 'random'),
    }


def plan_jobs(arms, seeds):
    # Seed 0 runs the six mechanism arms. Additional seeds replicate learned v2.
    jobs = [(arm, 0, target) for arm in arms for target in range(4)]
    jobs += [('v2_learned', seed, target) for seed in seeds if seed != 0 for target in range(4)]
    return jobs


def summarize(root, jobs):
    groups = {}
    for job in jobs:
        path = Path(job['output']) / 'results.json'
        if job['state'] != 'completed' or not path.exists():
            continue
        result = json.loads(path.read_text())
        group = groups.setdefault(job['arm'], {})
        group.setdefault(str(job['seed']), {})[str(job['target_env'])] = {
            'target_accuracy': result['target'][0]['accuracy'],
            'selected_step': result['best_step'] + 1,
            'source_selected_accuracy': result['best_source_accuracy'],
        }
    summary = {}
    for arm, seeds in groups.items():
        means = {seed: statistics.mean(row['target_accuracy'] for row in targets.values())
                 for seed, targets in seeds.items() if len(targets) == 4}
        values = list(means.values())
        summary[arm] = {'seeds': seeds, 'complete_seed_means': means,
                        'mean_accuracy': statistics.mean(values) if values else None,
                        'std_across_seed_means': statistics.stdev(values) if len(values) > 1 else None}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))


def run(args):
    root = Path(args.output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'use a fresh output directory: {root}')
    root.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(args.config).read_text())
    if base['dataset'] != 'PACS':
        raise ValueError('this suite is a PACS-only protocol')
    if args.smoke:
        base.update(steps=4, checkpoint_freq=4, structure_interval=1,
                    structure_prediction_samples=8, structure_cell_samples=2,
                    alignment_pair_draws=1, alignment_cell_samples=2)
        base['hparams'].update(model='deit_tiny_patch16_224', pretrained=False,
                              batch_size=2, support_warmup_steps=1,
                              support_adm_ramp_steps=2, support_gradient_interval=1)
    arms = experiment_arms(base)
    seeds = [0] if args.smoke else [0, 1, 2]
    plan = plan_jobs(arms, seeds)
    if args.smoke:
        # One target per arm validates all branches without six full benchmarks.
        plan = [(arm, 0, 0) for arm in arms]
    repository = Path(__file__).resolve().parents[2]
    # Later edits to the workspace cannot silently change pending experiments.
    snapshot = root / 'source_snapshot'
    shutil.copytree(repository / 'domainbed', snapshot / 'domainbed',
                    ignore=shutil.ignore_patterns('data', 'pretrained', '__pycache__', '*.pyc'))
    hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*.py'))}
    (root / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))
    configs = root / 'configs'
    configs.mkdir()
    for name, (config, _) in arms.items():
        config['experiment_arm'] = name
        (configs / f'{name}.json').write_text(json.dumps(config, indent=2))
    status = {'pid': os.getpid(), 'state': 'running', 'seeds': seeds,
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'protocol': 'six seed-0 arms; learned v2 repeated at seeds 1 and 2',
              'smoke': args.smoke, 'jobs': []}
    for arm, seed, target in plan:
        name = f'{arm}_env{target}_seed{seed}'
        status['jobs'].append({'arm': arm, 'seed': seed, 'target_env': target,
                               'state': 'queued', 'output': str(root / name),
                               'log': str(root / f'{name}.log')})
    def save():
        temp = root / 'queue.tmp'
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(root / 'queue.json')
    save()
    env = dict(os.environ, PYTHONPATH=str(snapshot), PYTHONWARNINGS='ignore', OMP_NUM_THREADS='4')
    for job in status['jobs']:
        mode = arms[job['arm']][1]
        command = [sys.executable, '-u', '-m', 'domainbed.scripts.train_predictive_support',
                   '--config', str(configs / f'{job["arm"]}.json'),
                   '--output-dir', job['output'], '--target-env', str(job['target_env']),
                   '--seed', str(job['seed']), '--support-mode', mode,
                   '--data-dir', str(Path(args.data_dir).resolve()), '--device', args.device]
        with open(job['log'], 'w') as log:
            process = subprocess.Popen(command, cwd=snapshot, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            job.update(state='running', pid=process.pid)
            save()
            code = process.wait()
        job.update(state='completed' if code == 0 else 'failed', returncode=code)
        if code:
            status['state'] = 'failed'
            save()
            return code
        save()
        summarize(root, status['jobs'])
    status['state'] = 'completed'
    save()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/predictive_support_v2_pacs.json')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--data-dir', default='domainbed/data')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    sys.exit(run(parser.parse_args()))
