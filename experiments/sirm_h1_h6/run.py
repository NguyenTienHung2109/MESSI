"""PACS H1-H6 CLI. Run --help; see experiments/sirm_h1_h6/README.md."""
import argparse
import copy
import hashlib
import itertools
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from experiments.sirm_h1_h6.model import HypothesesSIRM
from domainbed.lib import misc
from domainbed.subset_irm import per_sample_topk
from experiments.sirm_h1_h6.statistics import (expert_losses, choose_assignment, classifier_gradient_gaps, fit_gap_probes, evaluate_gaps)
from experiments.sirm_h1_h6.controls import (fit_router, route, metrics, mixture, frozen_controls,
    paired_delta, subset_cache)
from domainbed.scripts.train_subset_irm_pacs import seed_everything, split_keys, make_schedule, FixedBatchSampler, save_checkpoint

DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
CRITERIA = ['Random', 'Risk', 'Grad', 'Regret', 'FixedPairs', 'Global']
COLORS = [(220, 30, 30), (30, 180, 30), (30, 30, 220), (180, 100, 20)]
MEAN, STD = [.485, .456, .406], [.229, .224, .225]


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


class ScheduledImages(torch.utils.data.Dataset):
    def __init__(self, subset, seed, env):
        self.subset, self.seed, self.env = subset, seed, env

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, key):
        index, step, position = key
        # Stable augmentation independent of worker scheduling, and exactly resumable.
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        seed = misc.seed_hash(self.seed, self.env, step, position, 'augmentation')
        random.seed(seed); np.random.seed(seed); torch.random.default_generator.manual_seed(seed)
        try:
            return self.subset[index]
        finally:
            random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


class ScheduledBatches:
    def __init__(self, schedule, start_step):
        self.schedule, self.start_step = schedule, start_step

    def __iter__(self):
        for step in range(self.start_step, len(self.schedule)):
            yield [(int(index), step, position) for position, index in enumerate(self.schedule[step])]

    def __len__(self):
        return len(self.schedule) - self.start_step


def root_domain(root, env):
    return Path(root) / 'PACS' / DOMAINS[env]


class ViewTransform:
    def __init__(self, config, view='original', train=False, env=0, cohort='standard', parent_indices=None):
        self.config, self.view, self.train, self.env, self.cohort = config, view, train, env, cohort
        self.prepare = T.Compose([T.RandomResizedCrop(224, scale=(.7, 1.)), T.RandomHorizontalFlip(),
                                  T.ColorJitter(.3, .3, .3, .3), T.RandomGrayscale(p=.1)]) if train else T.Resize((224, 224))
        self.normalize = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __call__(self, image):
        independent_color = int(hashlib.sha256(image.tobytes()).hexdigest()[:8], 16) % 4
        image = self.prepare(image)
        view = random.choice(['original', 'grayscale', 'blur']) if self.train and self.cohort == 'balanced_views' else self.view
        if view == 'grayscale':
            image = TF.rgb_to_grayscale(image, num_output_channels=3)
        elif view == 'blur':
            image = TF.gaussian_blur(image, self.config['blur_kernel'], self.config['blur_sigma'])
        elif view == 'style':
            image = TF.adjust_brightness(image, self.config['brightness'])
            image = TF.adjust_contrast(image, self.config['contrast'])
            image = TF.adjust_saturation(image, self.config['saturation'])
        if self.cohort == 'border' or view.startswith('border_'):
            color = independent_color if self.env < 0 or view == 'border_random' else self.env
            if view == 'border_swapped':
                color = (color + 1) % 4
            width = self.config['border_width']
            # Add a frame outside resized content; preserve the full original image.
            content = TF.resize(image, [224 - 2 * width, 224 - 2 * width])
            frame = Image.new('RGB', (224, 224), (128, 128, 128) if view == 'border_removed' else COLORS[color])
            frame.paste(content, (width, width))
            image = frame
        return self.normalize(image)


def inventory(root, envs, seed, config):
    result = {'protocol': 'DomainBed80_20', 'seed': seed, 'holdout_fraction': .2,
              'sources': {}, 'class_names': None}
    for env in envs:
        data = ImageFolder(root_domain(root, env))
        if result['class_names'] is None:
            result['class_names'] = data.classes
        if data.classes != result['class_names'] or len(data.classes) != 7:
            raise ValueError('PACS source domains must share the same seven classes')
        in_ids, out_ids = split_keys(len(data), .2, seed, env)
        samples = [{'id': str(Path(p).relative_to(Path(root) / 'PACS')), 'class': y, 'sha256': digest(p)}
                   for p, y in data.samples]
        result['sources'][str(env)] = {'in': in_ids, 'out': out_ids, 'images': samples}
        if set(in_ids) & set(out_ids) or len(in_ids) + len(out_ids) != len(data):
            raise AssertionError('split overlap or missing image')
    return result


def datasets_for(root, split, config, view='original', train=False, cohort='standard', limit=None):
    result = {}
    for key, entry in split['sources'].items():
        env = int(key)
        data = ImageFolder(root_domain(root, env), ViewTransform(config, view, train, env, cohort))
        actual = [str(Path(p).relative_to(Path(root) / 'PACS')) for p, _ in data.samples]
        if actual != [s['id'] for s in entry['images']]:
            raise ValueError('dataset inventory changed; refuse to reuse split')
        indices = entry['in' if train else 'out']
        if limit:
            indices = indices[:limit]
        result[env] = Subset(data, indices)
    return result


def source_data(root, split, config, split_name, view='original', cohort='standard', limit=None):
    result = {}
    for key, entry in split['sources'].items():
        env = int(key)
        data = ImageFolder(root_domain(root, env), ViewTransform(config, view, False, env, cohort))
        if [(str(Path(p).relative_to(Path(root) / 'PACS')), y) for p, y in data.samples] != [(s['id'], s['class']) for s in entry['images']]:
            raise ValueError('dataset inventory changed')
        ids = entry[split_name]
        if limit:
            # smoke samples cover classes for numerical/gradient statistics.
            groups = {c: [i for i in ids if data.targets[i] == c] for c in range(7)}
            ids = [i for pos in range(max(map(len, groups.values()))) for c in range(7)
                   for i in groups[c][pos:pos + 1]][:limit]
        result[env] = Subset(data, ids)
    return result


@torch.no_grad()
def collect(model, data, config, metadata):
    model.eval()
    columns = {k: [] for k in ['expert_logits', 'expert_features', 'projected_feature', 'router_dense', 'router_topk', 'labels', 'domains']}
    image_ids, physical_domains = [], []
    for local_domain, (env, subset) in enumerate(sorted(data.items())):
        loader = DataLoader(subset, batch_size=config['eval_batch_size'], num_workers=config['num_workers'], shuffle=False)
        offset = 0
        for x, y in loader:
            x = x.cuda()
            z = model.moe_head.input_proj(model.featurizer(x))
            features = torch.stack([expert(z) for expert in model.moe_head.experts], 1)
            logits = model.expert_heads(features)
            dense = model.moe_head.router(z).softmax(1)
            topk = per_sample_topk(dense, model.router_topk)
            direct = model._subset_forward(x)[0]
            torch.testing.assert_close(direct, mixture(logits, model._routing_responsibilities(dense)), rtol=1e-4, atol=1e-5)
            for key, value in [('expert_logits', logits), ('expert_features', features), ('projected_feature', z),
                               ('router_dense', dense), ('router_topk', topk), ('labels', y),
                               ('domains', torch.full_like(y, local_domain))]:
                columns[key].append(value.cpu())
            ids = subset.indices[offset:offset + len(y)]
            image_ids.extend([f'{DOMAINS[env]}/{Path(subset.dataset.samples[i][0]).parent.name}/{Path(subset.dataset.samples[i][0]).name}' for i in ids])
            physical_domains.extend([env] * len(y))
            offset += len(y)
    result = {key: torch.cat(value) for key, value in columns.items()}
    result.update(image_ids=image_ids, parent_ids=list(image_ids), physical_domains=torch.tensor(physical_domains), metadata=metadata)
    return result


def build_model(hparams):
    return HypothesesSIRM((3, 224, 224), 7, 3, hparams).cuda()


def auxiliary_partition(cache):
    # Bank still trains on the entire DomainBed in-split. This partition only
    # prevents H6 linear probes from being scored on their own fitted images.
    fit_mask = torch.zeros(len(cache['labels']), dtype=torch.bool)
    for d in range(3):
        for c in cache['labels'].unique():
            ids = ((cache['domains'] == d) & (cache['labels'] == c)).nonzero().flatten()
            if len(ids) < 4:
                raise ValueError('H6 needs >=4 source-in samples per domain/class')
            fit_mask[ids[:len(ids)//2]] = True
    return subset_cache(cache, fit_mask), subset_cache(cache, ~fit_mask)


def assignment_statistics(source_in, validation, config, criterion=None):
    fit_cache, probe_cache = auxiliary_partition(source_in)
    losses = expert_losses(probe_cache['expert_logits'], probe_cache['labels'])
    result = {'risk': torch.stack([losses[probe_cache['domains'] == d].mean(0) for d in range(3)]),
              'probe_ids': probe_cache['image_ids'], 'fit_ids': fit_cache['image_ids']}
    if criterion in (None, 'Grad'):
        result['grad'], result['grad_info'] = classifier_gradient_gaps(probe_cache, config['grad_min_count'], config['grad_min_norm'])
        if validation is not None:
            result['validation_grad'], result['validation_grad_info'] = classifier_gradient_gaps(validation, config['grad_min_count'], config['grad_min_norm'])
    if criterion in (None, 'Regret'):
        probes = fit_gap_probes(fit_cache, config['probe_regularization'], config['probe_max_iter'], config['probe_tolerance'])
        result['gap'], result['gap_se'] = evaluate_gaps(probes, probe_cache)
        result['probe_gradient_max'] = max(p['gradient_max'] for p in probes.values())
        if validation is not None:
            result['validation_gap'], result['validation_gap_se'] = evaluate_gaps(probes, validation)
    return result


def train(args, config):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    output = Path(args.output)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError('use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'checkpoints').mkdir(exist_ok=args.resume)
    seed_everything(args.seed)
    source_envs = [e for e in range(4) if e != args.target_env]
    split = inventory(args.data_dir, source_envs, args.seed, config)
    if args.resume and json.loads((output / 'split_manifest.json').read_text()) != split:
        raise ValueError('resume dataset hashes/splits differ')
    if not args.resume:
        write(output / 'split_manifest.json', split)
    hparams = dict(config['hparams'])
    if args.topk:
        hparams['subset_irm_router_topk'] = args.topk
    steps = args.steps or config['steps']
    if args.smoke:
        hparams.update(subset_irm_topk_warmup_steps=2, subset_irm_sirm_anneal_steps=2,
                       subset_irm_route_anneal_steps=2, subset_irm_probe_interval=2,
                       subset_irm_gradient_log_interval=2)
    effective = {'config': config, 'hparams': hparams, 'seed': args.seed, 'target_env': args.target_env,
                 'variant': args.variant or f'{args.cohort}_top{hparams["subset_irm_router_topk"]}_{args.criterion or "nativeQ"}',
                 'cohort': args.cohort, 'criterion': args.criterion, 'assignment_weight': args.assignment_weight, 'steps': steps, 'smoke': args.smoke,
                 'limit': args.limit, 'split_hash': canonical_hash(split)}
    if args.resume and json.loads((output / 'effective_config.json').read_text()) != effective:
        raise ValueError('resume requires identical config, steps, split, cohort, seed and variant')
    if not args.resume:
        write(output / 'effective_config.json', effective)
    model = build_model(hparams)
    # Match DeiTFeaturizer's precedence: local offline weights before hub cache.
    pretrained = list(Path('domainbed/pretrained').glob('deit_small_distilled_patch16_224-649709d9.pth'))
    pretrained += list((Path(torch.hub.get_dir()) / 'checkpoints').glob('deit_small_distilled_patch16_224-649709d9.pth'))
    if not pretrained:
        raise RuntimeError('cannot identify pretrained weights for provenance')
    manifest = dict(effective, git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    git_diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest(),
                    gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                    python=platform.python_version(), pretrained_sha256=digest(pretrained[0]),
                    target_read_during_training=False, selection='mean source-out accuracy; earliest tie')
    initial_digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        initial_digest.update(key.encode()); initial_digest.update(value.detach().cpu().numpy().tobytes())
    manifest['initialization_sha256'] = initial_digest.hexdigest()
    manifest['parameters_total'] = sum(p.numel() for p in model.parameters())
    manifest['parameters_trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not args.resume:
        write(output / 'manifest.json', manifest)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    if args.resume and (output / 'environment.freeze.txt').read_text() != freeze:
        raise ValueError('resume environment changed')
    if not args.resume:
        (output / 'environment.freeze.txt').write_text(freeze)
    # Store the executed source, including untracked implementation files.
    snapshot = output / 'source_snapshot'
    snapshot.mkdir(exist_ok=args.resume)
    for name in ['experiments/sirm_h1_h6/run.py', 'experiments/sirm_h1_h6/statistics.py',
                 'experiments/sirm_h1_h6/controls.py', 'experiments/sirm_h1_h6/model.py', 'domainbed/subset_irm.py']:
        saved_source = snapshot / Path(name).name
        if args.resume and saved_source.read_bytes() != Path(name).read_bytes():
            raise ValueError(f'resume code changed: {name}')
        if not args.resume:
            saved_source.write_bytes(Path(name).read_bytes())
    train_sets = datasets_for(args.data_dir, split, config, train=True, cohort=args.cohort, limit=args.limit)
    val_sets = source_data(args.data_dir, split, config, 'out', cohort=args.cohort, limit=args.limit)
    if args.criterion:
        model.set_assignment(torch.zeros(3, 6))
    best, best_step, start, previous_q = -1., None, time.time(), None
    start_step, resume_state = 0, None
    if args.resume:
        resume_state = load(output / 'resume.pt')
        if resume_state['effective_hash'] != canonical_hash(effective):
            raise ValueError('resume config hash differs')
        model.load_state_dict(resume_state['model_dict'], strict=True)
        model.optimizer.load_state_dict(resume_state['optimizer'])
        start_step, best, best_step = resume_state['next_step'], resume_state['best'], resume_state['best_step']
        previous_q = resume_state['previous_q']
        for name in ['train_log.jsonl', 'assignment_log.jsonl']:
            path = output / name
            if path.exists():
                rows = [line for line in path.read_text().splitlines() if json.loads(line)['step'] < start_step]
                path.write_text(''.join(line + '\n' for line in rows))
        if start_step >= steps and (output / 'summary.json').exists() and (output / 'checkpoints/last.pkl').exists():
            raise ValueError('run already complete; resume is not required')
    loaders = []
    schedules = []
    for env, subset in sorted(train_sets.items()):
        schedule = make_schedule(list(range(len(subset))), steps, hparams['batch_size'], args.seed, env)
        schedules.append(schedule)
        loaders.append(iter(DataLoader(ScheduledImages(subset, args.seed, env),
                         batch_sampler=ScheduledBatches(schedule, start_step), num_workers=config['num_workers'],
                         generator=torch.Generator().manual_seed(misc.seed_hash(args.seed, env, 'loader')), pin_memory=True)))
    if resume_state:
        restore_rng(resume_state['rng'])
    manifest['schedule_sha256'] = hashlib.sha256(b''.join(s.tobytes() for s in schedules)).hexdigest()
    if not args.resume:
        write(output / 'manifest.json', manifest)
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step, steps):
        if args.criterion and step % (2 if args.smoke else config['assignment_interval']) == 0:
            frozen = collect(model, source_data(args.data_dir, split, config, 'in', cohort=args.cohort, limit=args.limit),
                             config, {'kind': 'source', 'split': 'in'})
            statistics = assignment_statistics(frozen, None, config, args.criterion)
            compatible = statistics.get('grad' if args.criterion == 'Grad' else 'gap', torch.zeros(8, 6))
            q = choose_assignment(statistics['risk'], compatible, args.criterion, args.seed, args.assignment_weight)
            model.set_assignment(q)
            row = {'step': step, 'q': q.tolist(), 'risk': statistics['risk'].tolist(),
                   'compatibility': compatible.tolist(), 'fit_ids': statistics['fit_ids'], 'probe_ids': statistics['probe_ids'],
                   'churn': 0 if previous_q is None else (q.gt(0) != previous_q.gt(0)).float().mean().item()}
            if 'grad_info' in statistics:
                row['gradient_support'] = statistics['grad_info']
            if 'probe_gradient_max' in statistics:
                row['probe_gradient_max'] = statistics['probe_gradient_max']
            if 'gap_se' in statistics:
                row['gap_se'] = statistics['gap_se'].tolist()
            with (output / 'assignment_log.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            previous_q = q
        model.train()
        minibatches = [(x.cuda(), y.cuda()) for x, y in [next(loader) for loader in loaders]]
        values = model.update(minibatches)
        if step % config['checkpoint_freq'] != 0 and step != steps - 1:
            continue
        model.eval()
        scores = {}
        with torch.no_grad():
            for env, data in val_sets.items():
                correct, count = 0, 0
                for x, y in DataLoader(data, batch_size=config['eval_batch_size'], num_workers=config['num_workers']):
                    correct += model.predict(x.cuda()).argmax(1).cpu().eq(y).sum().item()
                    count += len(y)
                scores[str(env)] = correct / count
        score = sum(scores.values()) / 3
        row = dict(values, step=step, source_validation=scores, source_mean=score,
                   peak_vram_gib=torch.cuda.max_memory_allocated() / 1024 ** 3, seconds=time.time() - start)
        with (output / 'train_log.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        if score > best:
            best, best_step = score, step
            save_checkpoint(output / 'checkpoints/best.pkl', model, {'step': step, 'manifest': manifest})
        state = {'model_dict': model.state_dict(), 'optimizer': model.optimizer.state_dict(),
                 'next_step': step + 1, 'best': best, 'best_step': best_step, 'previous_q': previous_q,
                 'rng': capture_rng(), 'effective_hash': canonical_hash(effective)}
        torch.save(state, output / 'resume.tmp.pt')
        (output / 'resume.tmp.pt').replace(output / 'resume.pt')
        print(json.dumps({'step': step, 'source_mean': score, 'loss': values['loss'], 'sirm_active': values['sirm_active']}), flush=True)
        if args.stop_after and step + 1 >= args.stop_after:
            if not args.smoke:
                raise ValueError('--stop-after is only for smoke resume verification')
            return
    save_checkpoint(output / 'checkpoints/last.pkl', model, {'step': steps - 1, 'manifest': manifest})
    write(output / 'summary.json', {'selected_step': best_step, 'source_mean': best,
                                   'target_evaluated': False, 'seconds': time.time() - start})


def selected_model(bank):
    checkpoint = Path(bank) / 'checkpoints/best.pkl'
    payload = load(checkpoint)
    effective = json.loads((Path(bank) / 'effective_config.json').read_text())
    seed_everything(effective['seed'])
    model = build_model(effective['hparams'])
    if 'hypotheses_q' in payload['model_dict']:
        model.set_assignment(payload['model_dict']['hypotheses_q'])
    model.load_state_dict(payload['model_dict'], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, effective, digest(checkpoint)


def cache(args, config):
    bank = Path(args.bank)
    if args.target_only and args.source_only:
        raise ValueError('choose exactly one cache stage')
    if args.target_only and not args.routers:
        raise ValueError('--target-only requires source-selected --routers before reading target')
    if not args.source_only and not args.target_only:
        raise ValueError('cache requires --source-only first, then --target-only after router selection')
    model, effective, checkpoint_hash = selected_model(bank)
    config = effective['config']
    if args.target_only:
        locked = load(args.routers)
        if locked['metadata']['checkpoint_sha256'] != checkpoint_hash:
            raise ValueError('router was fitted on a different bank')
    split = json.loads((bank / 'split_manifest.json').read_text())
    if canonical_hash(split) != effective['split_hash']:
        raise ValueError('split manifest changed')
    output = Path(args.output)
    for env, entry in split['sources'].items():
        for sample in entry['images']:
            if digest(Path(args.data_dir) / 'PACS' / sample['id']) != sample['sha256']:
                raise ValueError('source image bytes changed after training')
    if args.target_only:
        for split_name, expected in locked['metadata']['source_cache_sha256'].items():
            if digest(output / f'source_{split_name}_original.pt') != expected:
                raise ValueError('source cache changed after router selection')
    output.mkdir(parents=True, exist_ok=args.target_only)
    views = ['original', 'grayscale', 'blur', 'style']
    if effective['cohort'] == 'border':
        views += ['border_swapped', 'border_removed', 'border_random']
    for kind in (['target'] if args.target_only else ['source', 'target']):
        if kind == 'target':
            if args.source_only:
                continue
            # Target is opened only in cache stage, after source-selected bank is locked.
            data = ImageFolder(root_domain(args.data_dir, effective['target_env']))
            if data.classes != split['class_names']:
                raise ValueError('target class mapping differs')
        for split_name in (['in', 'out'] if kind == 'source' else ['all']):
            for view in views:
                if kind == 'source':
                    data_sets = source_data(args.data_dir, split, config, split_name, view, effective['cohort'], args.limit)
                else:
                    env = effective['target_env']
                    transform = ViewTransform(config, view, False, -1 if effective['cohort'] == 'border' else env, effective['cohort'])
                    # No correct target-domain color exists: original border target always uses independent colors.
                    target = ImageFolder(root_domain(args.data_dir, env), transform)
                    ids = list(range(len(target)))[:args.limit] if args.limit else list(range(len(target)))
                    data_sets = {env: Subset(target, ids)}
                random.seed(effective['seed'])
                metadata = {'kind': kind, 'split': split_name, 'view': view, 'checkpoint_sha256': checkpoint_hash,
                            'split_hash': effective['split_hash'], 'seed': effective['seed'], 'target_env': effective['target_env'],
                            'smoke': effective['smoke'] or bool(args.limit) or (locked['metadata']['smoke'] if args.target_only else False),
                            'routers_sha256': digest(args.routers) if args.target_only else None, 'transform': config['interventions'],
                            'bank_assignment': model._assignment(torch.full((3, 6), 1/6, device='cuda'), torch.arange(3, device='cuda')).cpu().tolist(),
                            'variant': effective['variant'], 'cohort': effective['cohort'], 'torchvision': __import__('torchvision').__version__}
                collected = collect(model, data_sets, config['interventions'] | config, metadata)
                destination = output / f'{kind}_{split_name}_{view}.pt'
                if destination.exists():
                    raise FileExistsError(f'refuse to overwrite cache: {destination}')
                torch.save(collected, destination)
                print(f'cached {kind}/{split_name}/{view}: {len(collected["labels"])}', flush=True)


def fit(args, config):
    directory = Path(args.cache)
    if list(directory.glob('target_*.pt')):
        raise ValueError('target already accessed; fit/select routers before target caching')
    fit_cache, validation = load(directory / 'source_in_original.pt'), load(directory / 'source_out_original.pt')
    for key in ['checkpoint_sha256', 'split_hash', 'seed', 'target_env', 'cohort']:
        if fit_cache['metadata'][key] != validation['metadata'][key]:
            raise ValueError(f'cache provenance mismatch: {key}')
    if Path(args.output).exists():
        raise FileExistsError('refuse to overwrite source-selected routers')
    if set(fit_cache['image_ids']) & set(validation['image_ids']):
        raise ValueError('source train/validation cache overlaps')
    routers = {'metadata': fit_cache['metadata']}
    seed = fit_cache['metadata']['seed']
    for sigma in config['noise_sigmas']:
        candidates = [fit_router(fit_cache, validation, config, t, sigma, seed=seed) for t in config['temperatures']]
        routers[f'input_sigma{sigma:g}'] = min(candidates, key=lambda r: r['validation_nll'])
    routers['global'] = fit_router(fit_cache, validation, config, global_only=True, seed=seed)
    pairs = []
    for a, b in itertools.combinations(range(6), 2):
        allowed = torch.zeros(6)
        allowed[[a, b]] = 1
        candidate = fit_router(fit_cache, validation, config, global_only=True, allowed=allowed, seed=seed)
        candidate['pair'] = [a, b]
        pairs.append(candidate)
    routers['fixed_pair'] = min(pairs, key=lambda r: r['validation_nll'])
    risks = expert_losses(fit_cache['expert_logits'], fit_cache['labels'])
    risk = torch.stack([risks[fit_cache['domains'] == d].mean(0) for d in range(3)]).mean(0)
    allowed = F.one_hot(risk.argmin(), 6).float()
    routers['best_global_expert'] = fit_router(fit_cache, validation, config, global_only=True, allowed=allowed, seed=seed)
    for m in range(6):
        allowed = torch.ones(6)
        allowed[m] = 0
        routers[f'refit_drop_expert{m}'] = fit_router(fit_cache, validation, config,
                    temperature=routers['input_sigma0']['temperature'], allowed=allowed, seed=seed)
    if not args.skip_h6:
        stats = assignment_statistics(fit_cache, validation, config)
        routers['h6_statistics'] = stats
        for criterion in CRITERIA:
            compatible = stats['grad'] if criterion == 'Grad' else stats['gap']
            weights = config['assignment_weights']  # same candidate/epoch budget for all criteria
            candidates = []
            for weight in weights:
                q = choose_assignment(stats['risk'], compatible, criterion, seed, weight)
                candidate = fit_router(fit_cache, validation, config, teacher_q=q, seed=seed)
                candidate.update(q=q, assignment_weight=weight, criterion=criterion)
                candidates.append(candidate)
            routers[f'Q_{criterion}'] = min(candidates, key=lambda r: r['validation_nll'])
    routers['metadata'] = dict(routers['metadata'], fitting_config=config,
                               fitting_source_sha256={name: digest(Path(__file__).parent / name) for name in ['run.py', 'controls.py', 'statistics.py']},
                               source_cache_sha256={name: digest(directory / f'source_{name}_original.pt') for name in ['in', 'out']},
                               target_used=False, selection='source-out NLL; earliest tie',
                               probe_role='source-in, shared with bank fitting as authorized DomainBed split')
    torch.save(routers, args.output)
    print(f'saved {len(routers) - 1} controls to {args.output}', flush=True)


def evaluate(args, config):
    directory, output = Path(args.cache), Path(args.output)
    evaluation_start = time.time()
    output.mkdir(parents=True, exist_ok=False)
    routers = load(args.routers)
    config = routers['metadata']['fitting_config']
    cache = load(directory / 'target_all_original.pt')
    for key in ['checkpoint_sha256', 'split_hash', 'seed', 'target_env', 'cohort']:
        if cache['metadata'][key] != routers['metadata'][key]:
            raise ValueError(f'target cache/router mismatch: {key}')
    if cache['metadata']['routers_sha256'] != digest(args.routers):
        raise ValueError('routers changed after target access')
    for split_name, expected in routers['metadata']['source_cache_sha256'].items():
        if digest(directory / f'source_{split_name}_original.pt') != expected:
            raise ValueError('source cache changed after fitting')
    seed = cache['metadata']['seed']
    controls = frozen_controls(cache, routers, seed)
    results = {name: metrics(cache, pi) for name, pi in controls.items()}
    comparisons = {name: paired_delta(cache, controls[name], controls['global'], seed, config['bootstrap_repeats'])
                   for name in ['input_sigma0', 'fixed_pair', 'native', 'input_shuffle_domain_class']}
    shuffle_deltas = {name: paired_delta(cache, controls['input_sigma0'], controls[name], seed, config['bootstrap_repeats'])
                      for name in ['input_shuffle_all', 'input_shuffle_domain_class', 'input_shuffle_feature', 'fixed_pair']}
    class_metrics = {}
    for c in cache['labels'].unique().tolist():
        mask = cache['labels'] == c
        part = subset_cache(cache, mask)
        class_metrics[str(c)] = {name: metrics(part, controls[name][mask]) for name in
                                ['input_sigma0', 'global', 'fixed_pair', 'input_shuffle_domain_class']}
    # H2: weighted evaluation of the same parents under all three forms.
    views, probabilities = {}, {}
    for view in ['original', 'grayscale', 'blur']:
        v = load(directory / f'target_all_{view}.pt')
        if v['image_ids'] != cache['image_ids'] or v['metadata'] != dict(cache['metadata'], view=view):
            raise ValueError('paired intervention parent IDs mismatch')
        views[view] = v
        probabilities[view] = {'input_sigma0': per_sample_topk(route(routers['input_sigma0'], v), 2),
                               'global': route(routers['global'], v), 'uniform': torch.full_like(v['router_dense'], 1 / 6),
                               'oracle_ce_single': F.one_hot(expert_losses(v['expert_logits'], v['labels']).argmin(1), 6).float()}
    h2 = {}
    for name in probabilities['original']:
        per_view = {view: metrics(views[view], probabilities[view][name]) for view in views}
        mixtures = []
        for rho in config['mixture_weights']:
            mixtures.append({'rho': rho, **{metric: sum(r * per_view[view][metric] for r, view in zip(rho, views))
                                              for metric in ['accuracy', 'nll', 'routing_regret']}})
        h2[name] = {'per_view': per_view, 'mixtures': mixtures,
                    'worst_mixture_accuracy': min(row['accuracy'] for row in mixtures)}
    # Router-only intervention keeps original expert logits, labels and IDs.
    style = load(directory / 'target_all_style.pt')
    if style['image_ids'] != cache['image_ids'] or style['metadata'] != dict(cache['metadata'], view='style'):
        raise ValueError('style parent IDs mismatch')
    original_pi = probabilities['original']['input_sigma0']
    transformed_pi = per_sample_topk(route(routers['input_sigma0'], style), 2)
    middle = (original_pi + transformed_pi) / 2
    js = .5 * ((original_pi * (original_pi.clamp_min(1e-9).log() - middle.clamp_min(1e-9).log())).sum(1)
               + (transformed_pi * (transformed_pi.clamp_min(1e-9).log() - middle.clamp_min(1e-9).log())).sum(1))
    h3 = {'original': metrics(cache, original_pi), 'router_only': metrics(cache, transformed_pi),
          'full': metrics(style, transformed_pi), 'routing_js': js.mean().item(),
          'top2_support_change_rate': (original_pi.gt(0) != transformed_pi.gt(0)).any(1).float().mean().item(),
          'router_only_accuracy_drop': metrics(cache, original_pi)['accuracy'] - metrics(cache, transformed_pi)['accuracy'],
          'modified_PACS_border': {}}
    for path in directory.glob('target_all_border_*.pt'):
        v = load(path)
        h3['modified_PACS_border'][v['metadata']['view']] = metrics(v, per_sample_topk(route(routers['input_sigma0'], v), 2))
    correct = cache['expert_logits'].argmax(-1).eq(cache['labels'].unsqueeze(1))
    disagreement = [[correct[:, a].ne(correct[:, b]).float().mean().item() for b in range(6)] for a in range(6)]
    h6 = {}
    if 'h6_statistics' in routers:
        stats = routers['h6_statistics']
        bank_q = torch.tensor(cache['metadata']['bank_assignment'])
        bank_masks = sum(bank_q[d].gt(0).long() * (1 << d) for d in range(3))
        shared = bank_q.gt(0).sum(0) >= 2
        if bool(shared.any()):
            bank_columns = torch.arange(6)
            h6['trained_bank_assignment'] = {'q': bank_q.tolist(),
                'validation_classifier_gap': stats['validation_gap'][bank_masks[shared], bank_columns[shared]].mean().item(),
                'validation_gradient_conflict': stats['validation_grad'][bank_masks[shared], bank_columns[shared]].mean().item(),
                'shared_experts': int(shared.sum()), 'target_native': results['native']}
        for criterion in CRITERIA:
            r = routers[f'Q_{criterion}']
            q = r['q']
            masks = sum(q[d].gt(0).long() * (1 << d) for d in range(3))
            active = masks > 0
            cols = torch.arange(6)
            h6[criterion] = {'target': results[f'Q_{criterion}'], 'q': q.tolist(),
                'active_experts': int(active.sum()), 'assignment_weight': r['assignment_weight'],
                'validation_classifier_gap': stats['validation_gap'][masks[active], cols[active]].mean().item(),
                'validation_gradient_conflict': stats['validation_grad'][masks[active], cols[active]].mean().item(),
                'structural_branch': 'global' if criterion == 'Global' else 'matched_two_edges'}
    write(output / 'target_metrics.json', {'metadata': cache['metadata'], 'controls': results,
            'paired_vs_global': comparisons, 'input_vs_shuffled_or_fixed': shuffle_deltas, 'per_class': class_metrics, 'H2': h2, 'H3': h3,
            'H5_error_disagreement': disagreement, 'H6': h6,
            'evaluation_seconds': time.time() - evaluation_start,
            'compute_note': 'all six experts computed densely; no theoretical Top-k FLOP saving claimed'})
    validation = load(directory / 'source_out_original.pt')
    source_controls = frozen_controls(validation, routers, seed)
    source_metrics = {'original_controls': {name: metrics(validation, pi) for name, pi in source_controls.items()},
                      'interventions': {}}
    for path in sorted(directory.glob('source_out_*.pt')):
        v = load(path)
        pi = per_sample_topk(route(routers['input_sigma0'], v), 2)
        source_metrics['interventions'][v['metadata']['view']] = {'input': metrics(v, pi), 'native': metrics(v, v['router_topk']),
                                                                'global': metrics(v, route(routers['global'], v))}
    write(output / 'source_metrics.json', source_metrics)
    torch.save({'parent_ids': cache['parent_ids'], 'labels': cache['labels'], 'expert_logits': cache['expert_logits'],
                'routing': controls}, output / 'paired_predictions.pt')
    print(f'evaluation saved to {output}', flush=True)


def report(args, config):
    files = sorted(Path(args.root).glob('**/target_metrics.json'))
    grouped = {}
    for path in files:
        data = json.loads(path.read_text())
        metadata = data['metadata']
        if metadata['smoke']:
            continue
        group = (metadata['variant'],)
        grouped.setdefault(group, []).append(data)
    summary = {}
    for (cohort,), rows in grouped.items():
        seen = set()
        for row in rows:
            key = (row['metadata']['target_env'], row['metadata']['seed'])
            if key in seen:
                raise ValueError(f'duplicate cohort/target/seed: {cohort} {key}')
            seen.add(key)
        seeds = sorted({seed for _, seed in seen})
        complete = [seed for seed in seeds if all((env, seed) in seen for env in range(4))]
        methods = sorted(set.intersection(*(set(row['controls']) for row in rows)))
        tables = {}
        for name in methods:
            per_target = {}
            for env in range(4):
                values = [row['controls'][name]['accuracy'] for row in rows if row['metadata']['target_env'] == env]
                per_target[str(env)] = {'mean': float(np.mean(values)) if values else None,
                                        'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None, 'seeds': len(values)}
            mean4 = [np.mean([row['controls'][name]['accuracy'] for row in rows if row['metadata']['seed'] == seed]) for seed in complete]
            tables[name] = {'per_target': per_target, 'mean4': float(np.mean(mean4)) if mean4 else None,
                            'seed_sd_mean4': float(np.std(mean4, ddof=1)) if len(mean4) > 1 else None}
        summary[cohort] = {'complete_four_target_seeds': complete, 'tables': tables,
                           'pilot_complete': all(seed in complete for seed in [0, 1, 2]),
                           'confirm_complete': all(seed in complete for seed in range(5))}
    write(args.output, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--data-dir', default='domainbed/data'); p.add_argument('--output', required=True)
    p.add_argument('--target-env', type=int, choices=range(4), required=True); p.add_argument('--seed', type=int, choices=range(5), required=True)
    p.add_argument('--variant'); p.add_argument('--resume', action='store_true'); p.add_argument('--stop-after', type=int)
    p.add_argument('--steps', type=int); p.add_argument('--smoke', action='store_true'); p.add_argument('--limit', type=int)
    p.add_argument('--cohort', choices=['standard', 'balanced_views', 'border'], default='standard')
    p.add_argument('--topk', type=int, choices=[1, 2]); p.add_argument('--criterion', choices=CRITERIA)
    p.add_argument('--assignment-weight', type=float, default=1.)
    p = sub.add_parser('cache')
    p.add_argument('--bank', required=True); p.add_argument('--data-dir', default='domainbed/data')
    p.add_argument('--output', required=True); p.add_argument('--source-only', action='store_true'); p.add_argument('--target-only', action='store_true'); p.add_argument('--routers'); p.add_argument('--limit', type=int)
    p = sub.add_parser('fit')
    p.add_argument('--cache', required=True); p.add_argument('--output', required=True); p.add_argument('--skip-h6', action='store_true')
    p = sub.add_parser('eval')
    p.add_argument('--cache', required=True); p.add_argument('--routers', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('report')
    p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    # Flatten locked intervention settings for image transforms.
    config.update(config['interventions'])
    torch.set_num_threads(config.get('cpu_threads', 4))
    if args.command == 'train' and args.stop_after and not args.smoke:
        parser.error('--stop-after requires --smoke')
    if args.command == 'train' and args.steps and args.steps != config['steps'] and not args.smoke:
        parser.error('nonstandard --steps requires --smoke; edit and lock a config for full runs')
    if hasattr(args, 'limit') and args.limit and args.command == 'train' and not args.smoke:
        parser.error('--limit training requires --smoke; truncated runs are not official experiments')
    globals()[{'train': 'train', 'cache': 'cache', 'fit': 'fit', 'eval': 'evaluate', 'report': 'report'}[args.command]](args, config)


if __name__ == '__main__':
    main()
