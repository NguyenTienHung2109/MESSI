"""Sequential, source-first PACS experiment launcher. Does not modify legacy code."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODULE = 'experiments.sirm_h1_h6.run'
VARIANTS = ['native', 'balanced_views', 'border', 'top1', 'Q_Random', 'Q_Risk', 'Q_Grad', 'Q_Regret', 'Q_FixedPairs', 'Q_Global']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--output', required=True)
    p.add_argument('--data-dir', default='domainbed/data')
    p.add_argument('--mode', choices=['smoke', 'pilot', 'confirm'], default='pilot')
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=['native'])
    p.add_argument('--targets', nargs='+', type=int, choices=range(4), default=list(range(4)))
    p.add_argument('--seeds', nargs='+', type=int, choices=range(5))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    seeds = args.seeds or ([0] if args.mode == 'smoke' else list(range(3 if args.mode == 'pilot' else 5)))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('suite output must be new; resume individual train stages with run.py train --resume')
    variants = ['native'] + [v for v in args.variants if v != 'native']
    config = json.loads(Path(args.config).read_text())
    commands = []
    config_path = str(Path(args.config).resolve())
    if args.mode == 'smoke':
        config.update(checkpoint_freq=2, router_epochs=8, temperatures=[1.], bootstrap_repeats=25)
        config_path = str((output / 'smoke_config.json').resolve())
    def command(stage, *options):
        return [sys.executable, '-m', MODULE, '--config', config_path, stage, *map(str, options)]
    for env in args.targets:
        for seed in seeds:
            # All training, source-only tuning and assignments are locked BEFORE any target pixels are read.
            for variant in variants:
                bank = output / variant / f'env{env}' / f'seed{seed}'
                train = command('train', '--output', bank, '--data-dir', args.data_dir, '--target-env', env,
                                '--seed', seed, '--variant', variant)
                if variant in ['balanced_views', 'border']:
                    train += ['--cohort', variant]
                if variant == 'top1':
                    train += ['--topk', '1']
                if variant.startswith('Q_'):
                    criterion = variant[2:]
                    train += ['--criterion', criterion]
                    # Filled from frozen, source-selected baseline router, never from target scores.
                    train += ['--assignment-weight', f'FROM_SOURCE:{output / "native" / f"env{env}" / f"seed{seed}" / "routers.pt"}:{criterion}']
                if args.mode == 'smoke':
                    train += ['--smoke', '--steps', '4', '--limit', '56']
                commands.append(train)
                cmd = command('cache', '--source-only', '--bank', bank, '--data-dir', args.data_dir, '--output', bank / 'cache')
                if args.mode == 'smoke':
                    cmd += ['--limit', '56']
                commands.append(cmd)
                commands.append(command('fit', '--cache', bank / 'cache', '--output', bank / 'routers.pt'))
                commands.append([sys.executable, '-m', 'experiments.sirm_h1_h6.source_review', '--cache', str(bank / 'cache'),
                                 '--routers', str(bank / 'routers.pt'), '--output', str(bank / 'source_review.json')])
                commands.append([sys.executable, '-m', 'experiments.sirm_h1_h6.profile', '--bank', str(bank),
                                 '--data-dir', args.data_dir, '--output', str(bank / 'compute.json')])
            for variant in variants:
                bank = output / variant / f'env{env}' / f'seed{seed}'
                cmd = command('cache', '--target-only', '--bank', bank, '--data-dir', args.data_dir,
                              '--routers', bank / 'routers.pt', '--output', bank / 'cache')
                if args.mode == 'smoke':
                    cmd += ['--limit', '56']
                commands.append(cmd)
                commands.append(command('eval', '--cache', bank / 'cache', '--routers', bank / 'routers.pt', '--output', bank / 'evaluation'))
    commands.append(command('report', '--root', output, '--output', output / 'report.json'))
    commands.append([sys.executable, '-m', 'experiments.sirm_h1_h6.summarize', '--root', str(output),
                     '--output', str(output / 'hypotheses_report.json')])
    plan = dict(mode=args.mode, variants=variants, targets=args.targets, seeds=seeds,
                full_training_runs=len(variants)*len(args.targets)*len(seeds), commands=commands)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    output.mkdir(parents=True)
    if args.mode == 'smoke':
        Path(config_path).write_text(json.dumps(config, indent=2)+'\n')
    (output / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    environment = dict(os.environ, WANDB_MODE='disabled', PYTHONWARNINGS='ignore')
    import hashlib
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob('*.py')}
    (output / 'source_hashes.json').write_text(json.dumps(source_hashes, indent=2)+'\n')
    for index, cmd in enumerate(commands):
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != value for path, value in source_hashes.items()):
            raise RuntimeError('suite source changed during execution; refuse to mix implementations')
        for i, option in enumerate(cmd):
            if option.startswith('FROM_SOURCE:'):
                import torch
                _, file, criterion = option.split(':', 2)
                routers = torch.load(file, map_location='cpu', weights_only=False)
                cmd[i] = str(routers[f'Q_{criterion}']['assignment_weight'])
        print(f'[{index+1}/{len(commands)}] {" ".join(cmd)}', flush=True)
        with (output / f'stage_{index:03d}.log').open('w') as log:
            subprocess.run(cmd, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'finished: {output / "report.json"}')


if __name__ == '__main__':
    main()
