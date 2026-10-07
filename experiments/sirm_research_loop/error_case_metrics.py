"""Locked replay EDA: error-conditioned expert capacity and margin changes.

No fitting, selection, or parameter updates. Label-based expert oracles describe
saved capacity and are not deployable predictors or independent test evidence.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(a, classes, arm):
    y = a['y'].astype(int)
    g = a[arm + '_g']
    delta = a[arm + '_delta']
    pi = a[arm + '_pi'][:, :delta.shape[1]]
    term = delta * pi[:, :, None]
    full = g + term.sum(1)
    pred = g.argmax(1)
    actual = full.argmax(1)
    single = (g[:, None, :] + delta).argmax(2)
    routed_single = (g[:, None, :] + term).argmax(2)
    global_wrong = pred != y
    rows = np.arange(len(y))
    # Fixed original competitor excludes the true class even for global-correct.
    competitors = g.copy()
    competitors[rows, y] = -np.inf
    competitor = competitors.argmax(1)
    change = term[rows, :, y] - term[rows, :, competitor]
    assert np.allclose(change.sum(1),
                       (full[rows, y] - full[rows, competitor])
                       - (g[rows, y] - g[rows, competitor]), atol=2e-6)

    def group(mask):
        n = int(mask.sum())
        if not n:
            return None
        wrong = mask & global_wrong
        return dict(
            n=n, global_correct=int((mask & ~global_wrong).sum()),
            actual_correct=int((mask & (actual == y)).sum()),
            repairs=int((wrong & (actual == y)).sum()),
            harms=int((mask & ~global_wrong & (actual != y)).sum()),
            original_wrong_with_any_unscaled_single_repair=int(
                (wrong & (single == y[:, None]).any(1)).sum()),
            original_wrong_with_any_routed_single_repair=int(
                (wrong & (routed_single == y[:, None]).any(1)).sum()),
            per_expert_unscaled_repairs=[int((wrong & (single[:, m] == y)).sum())
                                         for m in range(delta.shape[1])],
            per_expert_routed_single_repairs=[int((wrong & (routed_single[:, m] == y)).sum())
                                              for m in range(delta.shape[1])],
            mean_fixed_competitor_margin_change=float(change[mask].sum(1).mean()),
            mean_per_expert_fixed_competitor_margin_change=change[mask].mean(0).tolist(),
            mean_router_null=float(a[arm + '_pi'][mask, -1].mean()),
        )

    groups = {'all': group(np.ones(len(y), dtype=bool)),
              'global_wrong': group(global_wrong),
              'global_correct': group(~global_wrong)}
    for env in sorted(set(a['domains'].tolist())):
        mask = a['domains'] == env
        groups[f'env/{env}'] = group(mask)
        for cls in range(len(classes)):
            part = mask & (y == cls)
            if part.any():
                groups[f'env/{env}/true/{classes[cls]}'] = group(part)
    for cls in range(len(classes)):
        for predicted in range(len(classes)):
            mask = (y == cls) & (pred == predicted)
            if mask.any():
                groups[f'true/{classes[cls]}/global_prediction/{classes[predicted]}'] = group(mask)
    return groups


def run(dest):
    manifest = json.loads((dest / 'experiment/subset_manifest.json').read_text())
    locks = json.loads((dest / 'eda/checkpoint_locks.json').read_text())
    result = dict(
        scope='All saved paired original-view cases, exploratory post-hoc; no fitting or selection. '
              'Single-expert label oracles are upper-bound capacity diagnostics, not usable routers. '
              'Routed single retains the original full-model routing weight; no rerouting. '
              'Margin comparator is the strongest original global incorrect class, fixed across interventions. '
              'Counts overlap across experts and groups; do not sum them as unique causal contributions.',
        checkpoint_locks=locks, splits={})
    for split in ['source_out', 'target_in', 'target_out']:
        with np.load(dest / 'eda' / (split + '_replay.npz')) as a:
            result['splits'][split] = {
                arm: summarize(a, manifest['class_names'], arm) for arm in ['OFF', 'ON']}
    out = dest / 'eda/error_conditioned_metrics.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    run(parser.parse_args().destination.resolve())
