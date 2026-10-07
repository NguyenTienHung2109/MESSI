"""Source-only teacher label dependence: calibrated finite routing controls.

Class prototypes use source-in labels/teachers only. True-class controls on
source-out are label oracles, not usable predictors. No fitting/updates/targets.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .shrinkage import digest, write


def js(a, b):
    mean = (a + b) / 2
    def kl(x):
        return (x * (x.clamp_min(1e-30).log() - mean.clamp_min(1e-30).log())).sum(1)
    return (kl(a) + kl(b)) / 2


def run(dest):
    torch.set_num_threads(4)
    src = dest / 'source_router_diagnosis'
    paths = {split: src / (split + '_original.npz') for split in ['source_in', 'source_out']}
    locks = {split: digest(path) for split, path in paths.items()}
    arrays = {}
    for split, path in paths.items():
        with np.load(path) as a:
            arrays[split] = {k: torch.tensor(a[k]) for k in ['g', 'delta', 'pi', 'teacher', 'y', 'domains']}
    a = arrays['source_in']
    classes = sorted(a['y'].unique().tolist())
    prototypes = []
    domains_by_class = {}
    for cls in classes:
        means = []
        for domain in sorted(a['domains'].unique().tolist()):
            mask = a['y'].eq(cls) & a['domains'].eq(domain)
            if mask.any():
                means.append(a['teacher'][mask].mean(0))
        assert means
        prototypes.append(torch.stack(means).mean(0))
        domains_by_class[str(cls)] = len(means)
    prototypes = torch.stack(prototypes)
    assert classes == list(range(len(prototypes)))
    constant = torch.stack([a['teacher'][a['domains'].eq(d)].mean(0)
                            for d in sorted(a['domains'].unique().tolist())]).mean(0)
    plan = dict(cache_hashes=locks, checkpoint_locks=json.loads((src / 'plan.json').read_text())['checkpoint_hashes'],
                code_sha256=digest(Path(__file__)), calibration='Source-in equal-domain means within class; each available domain weighted equally. Constant teacher also equal-domain. No source-out or target calibration.',
                prototypes=prototypes.tolist(), valid_source_domains_per_class=domains_by_class,
                true_class_control_is_label_oracle=True, target_read=False, fitting=False)
    write(src / 'teacher_prototype_plan.json', plan)
    result = {}
    for split, a in arrays.items():
        g, delta, y, domains = [a[k] for k in ['g', 'delta', 'y', 'domains']]
        pred = g.argmax(1)
        policies = dict(actual=a['pi'], per_image_label_teacher=a['teacher'],
                        true_class_teacher_prototype=prototypes[y],
                        global_prediction_teacher_prototype=prototypes[pred],
                        constant_teacher=constant.expand(len(y), -1))
        correct = pred.eq(y)
        masks = dict(all=torch.ones(len(y), dtype=torch.bool), global_correct=correct, global_wrong=~correct)
        for d in sorted(domains.unique().tolist()):
            masks[f'domain/{d}'] = domains.eq(d)
        summary = {}
        for group, mask in masks.items():
            if not mask.any():
                continue
            row = dict(n=int(mask.sum()),
                       per_image_vs_true_class_prototype_JS=float(js(a['teacher'][mask], policies['true_class_teacher_prototype'][mask]).mean()),
                       per_image_vs_global_prediction_prototype_JS=float(js(a['teacher'][mask], policies['global_prediction_teacher_prototype'][mask]).mean()),
                       per_image_vs_actual_JS=float(js(a['teacher'][mask], a['pi'][mask]).mean()))
            for name, pi in policies.items():
                logits = g + (pi[:, :6, None] * delta).sum(1)
                good = logits.argmax(1).eq(y)
                row[name] = dict(accuracy=float(good[mask].double().mean()),
                                 CE=float(F.cross_entropy(logits[mask], y[mask])),
                                 repairs=int((mask & ~correct & good).sum()),
                                 harms=int((mask & correct & ~good).sum()),
                                 NULL=float(pi[mask, -1].mean()))
            summary[group] = row
        result[split] = summary
    assert all(digest(path) == locks[split] for split, path in paths.items())
    write(src / 'teacher_prototype_results.json', dict(plan=plan, splits=result,
        scope='Finite policies at the identical source-selected bank; descriptive source-out already used for selection. Class-label oracles cannot establish deployable routing or useful factor recovery. No target or training.'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('destination', type=Path)
    run(parser.parse_args().destination.resolve())
