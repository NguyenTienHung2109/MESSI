"""Read-only environment, data, protocol and test checks before full training."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from experiments.sirm_h1_h6.run import DOMAINS, canonical_hash
from domainbed.scripts.train_subset_irm_pacs import split_keys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--data-dir', default='domainbed/data')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text())
    h = c['hparams']
    required = {'model': 'deit_small_patch16_224', 'legacy_non_distilled_deit': False,
                'pretrained': True, 'num_experts': 6, 'batch_size': 32,
                'subset_irm_prediction_mode': 'expert_logit_mix', 'subset_irm_use_expert_heads': True,
                'subset_irm_router_topk': 2, 'subset_irm_topk_warmup_steps': 500,
                'subset_irm_responsibility_detach': True, 'lr': 3e-5, 'weight_decay': 1e-6}
    for key, value in required.items():
        if h[key] != value:
            raise ValueError(f'protocol mismatch {key}: expected {value}, found {h[key]}')
    if c['steps'] != 5001 or c['assignment_interval'] != 500:
        raise ValueError('official protocol requires 5001 steps, assignment interval 500')
    if set(c['noise_sigmas']) != {0, .25, .5, 1}:
        raise ValueError('H1 noise controls are incomplete')
    if len(c['mixture_weights']) != 4 or any(abs(sum(rho)-1)>1e-6 for rho in c['mixture_weights']):
        raise ValueError('H2 mixtures invalid')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    x = torch.randn(64,64,device='cuda',requires_grad=True)
    (x@x).square().mean().backward()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError('CUDA backward failed')
    classes, inventory = None, {}
    for env, domain in enumerate(DOMAINS):
        data = ImageFolder(Path(args.data_dir)/'PACS'/domain)
        if classes is None:
            classes = data.classes
        if data.classes != classes or len(classes) != 7:
            raise ValueError('PACS must have identical class mapping in every domain')
        inventory[domain] = dict(images=len(data), classes={name:data.targets.count(i) for i,name in enumerate(classes)})
        for seed in range(5):
            in_, out_ = split_keys(len(data), .2, seed, env)
            for ids, minimum in [(in_,4), (out_,2)]:
                if any(sum(data.targets[i] == c for i in ids) < minimum for c in range(7)):
                    raise ValueError(f'insufficient per-class support: {domain}, seed {seed}')
    versions = subprocess.run([sys.executable,'-m','pip','check'],text=True,capture_output=True,check=True).stdout
    test = subprocess.run([sys.executable,'-m','unittest','-q','experiments.sirm_h1_h6.test_protocol',
                           'domainbed.test.test_subset_irm'],text=True,capture_output=True,check=True)
    result = dict(status='PASS', config_sha256=canonical_hash(c), python=sys.executable,
                  torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  classes=classes, inventory=inventory, dependencies=versions.strip(), tests=test.stderr.strip(),
                  scope='environment/data/numerical contracts; end-to-end smoke recorded separately')
    destination=Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
