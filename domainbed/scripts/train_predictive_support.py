"""Source-only structure learning and source-validation checkpoint selection."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from domainbed.predictive_support import PredictiveSupport, predictive_terms
from domainbed.support_eda import PredictionEDA


class SourceSampler:
    """Separate uniform-domain prediction draws and class-conditional MMD draws."""
    def __init__(self, datasets, indices, num_classes, seed, device):
        self.datasets, self.indices = datasets, indices
        self.rng = np.random.RandomState(seed)
        self.device = device
        self.cells = {}
        self.presence = torch.zeros(len(datasets), num_classes, dtype=torch.bool)
        for k, (data, keys) in enumerate(zip(datasets, indices)):
            if not keys:
                raise ValueError('empty source training split')
            for key in keys:
                c = int(data.targets[key])
                self.cells.setdefault((k, c), []).append(key)
                self.presence[k, c] = True
        # A singleton population cell can only produce repeated copies. Fail
        # explicitly instead of treating those as independent MMD evidence.
        for key, values in self.cells.items():
            if len(values) < 2 and self.presence[:, key[1]].sum() >= 2:
                raise ValueError(f'source cell {key} has fewer than two distinct images')

    def draw(self, k, keys, count):
        chosen = self.rng.choice(keys, size=count, replace=True)
        values = [self.datasets[k][int(i)] for i in chosen]
        return (torch.stack([x for x, _ in values]).to(self.device),
                torch.tensor([y for _, y in values], device=self.device))

    def prediction(self, count):
        return [self.draw(k, keys, count) for k, keys in enumerate(self.indices)]

    def structure_cells(self, count):
        if count < 2:
            raise ValueError('structure cell sample count must be >=2')
        return {key: self.draw(key[0], indices, count)[0] for key, indices in self.cells.items()
                if self.presence[:, key[1]].sum() >= 2}

    def alignment(self, support, count, images):
        if count < 1 or images < 2:
            raise ValueError('alignment needs positive pair draws and >=2 images per cell')
        support = support.cpu()
        pairs = [(m, i, j) for m in range(support.shape[1]) for i in range(len(support))
                 for j in range(i + 1, len(support)) if support[i, m] and support[j, m]
                 and bool((self.presence[i] & self.presence[j]).any())]
        result = []
        for _ in range(count):
            m, i, j = pairs[self.rng.randint(len(pairs))]
            shared = torch.where(self.presence[i] & self.presence[j])[0].tolist()
            c = int(self.rng.choice(shared))
            result.append((m, i, j, c, self.draw(i, self.cells[i, c], images)[0],
                           self.draw(j, self.cells[j, c], images)[0]))
        return result


@torch.no_grad()
def evaluate(model, datasets, device, batch_size, source=True, save_prefix=None):
    was_training = model.training
    model.eval()
    reports = []
    for k, data in enumerate(datasets):
        eda = PredictionEDA(model.num_experts, model.num_classes)
        total = correct = 0
        sums = torch.zeros(4, device=device)
        risk = torch.zeros(model.num_experts, device=device)
        expert_correct = torch.zeros_like(risk)
        for x, y in DataLoader(data, batch_size=batch_size):
            x, y = x.to(device), y.to(device)
            mix, logits, router, features = model.components(x)
            eda.add(y, mix, logits, router, features)
            total += len(y)
            correct += int(mix.argmax(1).eq(y).sum())
            expert_correct += logits.argmax(2).eq(y[:, None]).sum(0)
            s = model.support[k].expand(len(y), -1) if source else torch.ones_like(router, dtype=torch.bool)
            loc, adm, ce = predictive_terms(logits, router, y, s)
            risk += ce.sum(0)
            mixture = torch.nn.functional.cross_entropy(mix, y, reduction='sum')
            mass = (1 - (router.softmax(1) * s).sum(1)).sum()
            sums += torch.stack([loc.sum(), adm.sum(), mixture, mass])
        if total == 0:
            raise ValueError('empty evaluation split')
        loc, adm, mixture, mass = (sums / total).tolist()
        report = {'accuracy': correct / total, 'expert_risk': (risk / total).tolist(),
                  'expert_accuracy': (expert_correct / total).tolist(), 'mixture_ce': mixture}
        if source:
            report.update(local_risk=loc, admissibility=adm, inadmissible_mass=mass,
                          prediction_bound=loc + model.lmax * adm,
                          bound_gap=loc + model.lmax * adm - mixture,
                          assigned_experts=model.support[k].cpu().tolist())
        report["eda"] = eda.finalize()
        if save_prefix is not None:
            eda.save(str(save_prefix) + f"_{k}.npz")
        reports.append(report)
    model.train(was_training)
    return reports


def run(args):
    from domainbed import datasets
    config = json.loads(Path(args.config).read_text())
    hp = config['hparams']
    seed = args.seed if args.seed is not None else config.get('seed', 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(config.get('torch_threads', 4))
    device = args.device
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'output is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    cls = getattr(datasets, config['dataset'])
    target = args.target_env
    if target not in range(len(cls.ENVIRONMENTS)):
        raise ValueError('invalid held-out environment')
    train = cls(args.data_dir, [target], dict(hp, data_augmentation=True))
    val = cls(args.data_dir, [target], dict(hp, data_augmentation=False))
    sources = [k for k in range(len(train)) if k != target]
    splits = {}
    for k in sources:
        keys = np.random.RandomState(seed + k).permutation(len(train[k])).tolist()
        n = max(1, int(config.get('holdout_fraction', .2) * len(keys)))
        splits[k] = {'train': keys[n:], 'validation': keys[:n]}
    sampler = SourceSampler([train[k] for k in sources], [splits[k]['train'] for k in sources],
                            train.num_classes, seed, device)
    model = PredictiveSupport(train.input_shape, train.num_classes, len(sources), hp).to(device)
    model.configure_support(sampler.presence)
    mode = args.support_mode
    # The random intervention draws S once and holds it fixed throughout training.
    if mode == 'random':
        from domainbed.predictive_support import feasible_supports
        candidates, _ = feasible_supports(sampler.presence, model.num_experts, model.budget)
        model.support.copy_(candidates[int(torch.randint(len(candidates), ()).item())])
    steps = args.steps if args.steps is not None else config['steps']
    interval = int(config.get('structure_interval', 100))
    if interval < 1 or steps <= model.warmup:
        raise ValueError('need positive structure interval and steps beyond warmup')
    resolved = dict(config, seed=seed, target_env=target, source_envs=sources,
                    environment_names=[Path(train[k].root).name for k in range(len(train))],
                    class_names=train[sources[0]].classes,
                    support_mode=mode, steps=steps, splits=splits)
    (output / 'config.json').write_text(json.dumps(resolved, indent=2))
    source_val = [Subset(val[k], splits[k]['validation']) for k in sources]
    best, start = -1., time.perf_counter()
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        record = {'step': step}
        if step >= model.warmup and (step == model.warmup or (step - model.warmup) % interval == 0):
            structure_start = time.perf_counter()
            record.update(model.update_structure(
                sampler.prediction(config.get('structure_prediction_samples', 128)),
                sampler.structure_cells(config.get('structure_cell_samples', 16)),
                mode='fixed' if mode == 'random' else mode))
            record['structure_total_seconds'] = time.perf_counter() - structure_start
        alignment = None if step < model.warmup else sampler.alignment(
            model.support, config.get('alignment_pair_draws', 2), config.get('alignment_cell_samples', 4))
        neural_start = time.perf_counter()
        record.update(model.update(sampler.prediction(hp['batch_size']), alignment_batches=alignment))
        record['neural_seconds'] = time.perf_counter() - neural_start
        record['elapsed_seconds'] = time.perf_counter() - start
        if str(device).startswith('cuda'):
            record['gpu_peak_memory_mb'] = torch.cuda.max_memory_allocated() / 2**20
        if (step + 1) % config.get('checkpoint_freq', 100) == 0 or step == steps - 1:
            record['source_validation'] = evaluate(model, source_val, device, config.get('eval_batch_size', 64))
            score = np.mean([v['accuracy'] for v in record['source_validation']])
            if step >= model.warmup and score > best:
                best = float(score)
                torch.save({'model': model.state_dict(), 'optimizer': model.optimizer.state_dict(),
                            'step': step, 'source_score': best, 'config': resolved}, output / 'best.pt')
        with (output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps({k: v for k, v in record.items() if k not in ('source_validation', 'structure_eda')}), flush=True)
        if 'source_validation' in record:
            from domainbed.scripts.report_predictive_support import build_report
            build_report(output)
    torch.save({'model': model.state_dict(), 'optimizer': model.optimizer.state_dict(),
                'config': resolved}, output / 'last.pt')
    state = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    final = {'best_source_accuracy': best, 'best_step': state['step'],
             'seconds': time.perf_counter() - start,
             'source': evaluate(model, source_val, device, config.get('eval_batch_size', 64),
                                save_prefix=output / 'source_predictions'),
             'support': model.support.cpu().tolist()}
    # Target is read for scoring only after source-selected checkpoint is frozen.
    final['target'] = evaluate(model, [val[target]], device, config.get('eval_batch_size', 64), source=False,
                               save_prefix=output / 'target_predictions')
    (output / 'results.json').write_text(json.dumps(final, indent=2))
    from domainbed.scripts.report_predictive_support import build_report
    build_report(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/predictive_support_pacs.json')
    parser.add_argument('--data-dir', default='domainbed/data')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--target-env', type=int, default=0)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--support-mode', choices=['learned', 'fixed', 'random', 'no_discrepancy'], default='learned')
    run(parser.parse_args())
