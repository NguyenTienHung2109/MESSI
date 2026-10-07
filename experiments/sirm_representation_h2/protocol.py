"""Factorial configuration and evaluation-only transforms using the existing SIRM pipeline."""
import copy
import json
from pathlib import Path
import torch
from torchvision.transforms import functional as TF
from experiments.sirm_h1_h6 import run as base

CONDITIONS = {'A': ('baseline', False), 'B': ('h2_balanced', False),
              'C': ('baseline', True), 'D': ('h2_balanced', True)}
VIEWS = ['original', 'grayscale', 'blur', 'gray_blur']
SIRM_KEY = 'subset_irm_lambda_sirm'

def config_for(enabled):
    config = json.loads(Path('experiments/sirm_h1_h6/config.json').read_text())
    config.update(config['interventions'])
    if not enabled:
        config['hparams'][SIRM_KEY] = 0.
    return config

def assert_only_sirm_changed(on, off):
    expected = copy.deepcopy(on)
    expected['hparams'][SIRM_KEY] = 0.
    if off != expected:
        raise ValueError('OFF may change only subset_irm_lambda_sirm')
    if not off['hparams']['subset_irm_enabled']:
        raise ValueError('OFF must retain SIRM architecture')

class ViewTransform(base.ViewTransform):
    """Baseline/H2 are inherited exactly; only the evaluation gray+blur view is new."""
    def __call__(self, image):
        if self.view != 'gray_blur':
            return super().__call__(image)
        if self.train:
            raise ValueError('gray_blur is evaluation-only')
        image = self.prepare(image)
        image = TF.rgb_to_grayscale(image, num_output_channels=3)
        image = TF.gaussian_blur(image, self.config['blur_kernel'], self.config['blur_sigma'])
        return self.normalize(image)

def validate_source(cache, target):
    if cache['metadata']['kind'] != 'source':
        raise ValueError('target data cannot fit weights or select a model')
    if int(target) in cache['physical_domains'].unique().tolist():
        raise ValueError('held-out target leaked into source cache')

def train(condition, output, target, seed, smoke=False, resume=False):
    import argparse
    aug, enabled = CONDITIONS[condition]
    config = config_for(enabled)
    assert_only_sirm_changed(config_for(True), config_for(False))
    args = argparse.Namespace(output=str(output), data_dir='domainbed/data', seed=seed, target_env=target,
        criterion=None, assignment_weight=1., variant='factorial_'+condition,
        cohort='balanced_views' if aug == 'h2_balanced' else 'standard', topk=None,
        steps=6 if smoke else None, smoke=smoke, limit=8 if smoke else None,
        resume=resume, stop_after=None)
    base.train(args, config)
    (Path(output)/'source_snapshot/factorial_protocol.py').write_bytes(Path(__file__).read_bytes())

def validate_reuse(bank, condition, target, seed):
    aug, enabled = CONDITIONS[condition]
    if not enabled:
        raise ValueError('no OFF pilot bank has been declared reusable')
    eff = json.loads((bank/'effective_config.json').read_text())
    manifest = json.loads((bank/'manifest.json').read_text())
    config = config_for(enabled)
    if eff['config'] != config or eff['hparams'] != config['hparams']:
        raise ValueError('reuse config is not an exact match')
    expected_cohort = 'balanced_views' if aug == 'h2_balanced' else 'standard'
    if (eff['target_env'],eff['seed'],eff['cohort'],eff['steps'],eff['smoke'],eff['limit'],eff['criterion']) != (target,seed,expected_cohort,5001,False,None,None):
        raise ValueError('reuse arguments are not an exact match')
    for source in (bank/'source_snapshot').glob('*.py'):
        current = Path('domainbed/subset_irm.py') if source.name == 'subset_irm.py' else Path('experiments/sirm_h1_h6')/source.name
        if source.read_bytes() != current.read_bytes():
            raise ValueError(f'reuse source mismatch: {current}')
    import subprocess
    if manifest['git_sha'] != subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip():
        raise ValueError('reuse commit changed')
    split=json.loads((bank/'split_manifest.json').read_text())
    if base.canonical_hash(split)!=eff['split_hash'] or str(target) in split['sources']:
        raise ValueError('split mismatch or target leakage')
    for key in ['source_validation','source_mean']:
        if key not in json.loads((bank/'train_log.jsonl').read_text().splitlines()[-1]):
            raise ValueError('missing source-only selection log')
    summary=json.loads((bank/'summary.json').read_text())
    if summary['target_evaluated']:
        raise ValueError('target used during training')
    rows=[json.loads(line) for line in (bank/'train_log.jsonl').read_text().splitlines()]
    expected_sources=set(split['sources'])
    if any(set(row['source_validation'])!=expected_sources for row in rows):
        raise ValueError('validation log does not contain exactly the source domains')
    best=max(rows,key=lambda row:row['source_mean'])
    if (best['step'],best['source_mean'])!=(summary['selected_step'],summary['source_mean']):
        raise ValueError('reused checkpoint is not earliest best source validation')
    return dict(effective_config_hash=base.canonical_hash(eff), checkpoint_sha256=base.digest(bank/'checkpoints/best.pkl'),
                split_hash=eff['split_hash'], source_snapshot_verified=True, exact_match=True)

def branch_weights(global_router):
    if global_router['kind'] != 'global':
        raise ValueError('expected source-fitted global weights')
    weights=global_router['state'].softmax(0)
    torch.testing.assert_close(weights.sum(), torch.tensor(1.), rtol=1e-6, atol=1e-6)
    if not torch.isfinite(weights).all():
        raise ValueError('nonfinite global weights')
    return weights
