"""Source-only class-composition decomposition of fixed-bank utilities.

Descriptive changes of averaging measure, not new training EMA or Q policy.
Missing classes are never imputed. Common-class control requires >=2 images
per class in all three sources, and is distinct from SIRM ESS eligibility.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.sirm_discover.math import residual_utilities, soft_null_q
from .shrinkage import digest, write


def decomposition(g, delta, pi, y, domains):
    _, _, u, _ = residual_utilities(g, delta, pi, y)
    ds = sorted(domains.unique().tolist())
    classes = list(range(g.shape[1]))
    cells = {}
    counts = np.zeros((len(ds), len(classes)), dtype=int)
    for i, d in enumerate(ds):
        for cls in classes:
            mask = domains.eq(d) & y.eq(cls)
            counts[i, cls] = int(mask.sum())
            if mask.any():
                cells[f'{d}/{cls}'] = u[mask].mean(0).tolist()
    pooled = {}
    for cls in classes:
        valid = [torch.tensor(cells[f'{d}/{cls}']) for d in ds if f'{d}/{cls}' in cells]
        if valid:
            pooled[cls] = torch.stack(valid).mean(0)
    observed = torch.stack([u[domains.eq(d)].mean(0) for d in ds])
    composition = torch.stack([sum(float(counts[i, cls] / counts[i].sum()) * pooled[cls]
                                   for cls in classes if counts[i, cls]) for i, d in enumerate(ds)])
    conditional = observed - composition
    def interaction(x):
        return x - x.mean(0, keepdim=True) - x.mean(1, keepdim=True) + x.mean()
    ov, cv, rv = [interaction(x) for x in [observed, composition, conditional]]
    assert torch.allclose(ov, cv + rv, atol=1e-7)
    common = [cls for cls in classes if (counts[:, cls] >= 2).all()]
    controls = {}
    if common:
        balanced = torch.stack([torch.tensor([cells[f'{d}/{cls}'] for cls in common]).mean(0) for d in ds])
        for name, value in [('observed', observed), ('matched_common_class', balanced)]:
            q, _, _, v = soft_null_q(value, torch.ones_like(value, dtype=torch.bool),
                                      temperature=.05, utility_floor=.002, interaction_floor=.002)
            controls[name] = dict(U=value.tolist(), V=v.tolist(), fresh_Q=q.tolist(),
                                  selected_cells=int((q>0).sum()), shared_experts=int(((q>0).sum(0)>=2).sum()))
    return dict(domains=ds, class_counts=counts.tolist(), class_utility=cells,
                common_classes_at_least_two_per_domain=common,
                U_observed=observed.tolist(), U_class_composition=composition.tolist(),
                U_conditional_residual=conditional.tolist(),
                V_observed=ov.tolist(), V_class_composition=cv.tolist(), V_conditional_residual=rv.tolist(),
                V_frobenius_norms=dict(observed=float(ov.norm()), composition=float(cv.norm()), conditional=float(rv.norm())),
                controls=controls,
                limitation='Exact algebraic decomposition, not causal attribution. Composition and conditional residual are not orthogonal; their norm ratios are not explained variance. '
                           'Common-class control discards unmatched classes; it is descriptive, not a gate/threshold/ESS change.')


def run(dest):
    torch.set_num_threads(4)
    out = dest / 'source_router_diagnosis'
    result = dict(scope='Source-in/source-out original-view means at the fixed selected bank. No target, fitting, training, updates or selection. '
                       'Source-out already used for bank selection; class-conditioned means can be noisy.', splits={}, cache_hashes={})
    for split in ['source_in', 'source_out']:
        path = out / (split + '_original.npz')
        result['cache_hashes'][split] = digest(path)
        with np.load(path) as a:
            g, delta, y, domains, pi, teacher = [torch.tensor(a[k]) for k in ['g', 'delta', 'y', 'domains', 'pi', 'teacher']]
        with np.load(dest / 'fixed_teacher_probe' / (split + '_policies.npz')) as a:
            probe = torch.tensor(a['probe_pi'])
        result['splits'][split] = {name:decomposition(g, delta, policy, y, domains)
                                  for name, policy in [('actual', pi), ('diagnostic_probe', probe), ('label_teacher', teacher)]}
    write(out / 'class_utility_results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('destination', type=Path)
    run(parser.parse_args().destination.resolve())
