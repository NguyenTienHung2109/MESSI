"""Controlled fixed-map tests with known Gaussian SSI geometry, seed 0 only.

Sign projections give bounded binary features. Compress actual iid observations
into sufficient counts for convex CE; n remains the original sample count.
These isolate certification, not neural feature discovery or OOD routing.
"""
import argparse
import itertools
import math
from pathlib import Path
import numpy as np
import torch
from domainbed.predictive_compatibility import make_cell, solve_common, audit_head, gaussian_signal
from domainbed.scripts.audit_predictive_compatibility import write_json


def sampled_cell(rng, agreement, n):
    # Joint probabilities for (z,y) = (+,1),(-,1),(-,0),(+,0).
    joint = rng.multinomial(n, [agreement/2, (1-agreement)/2, agreement/2, (1-agreement)/2])
    cell = make_cell([[1.], [-1.], [-1.], [1.]], [1, 1, 0, 0], joint)
    return cell, [int(joint[2]+joint[3]), int(joint[0]+joint[1])]


def run(n=200000):
    rng = np.random.default_rng(0)
    cases = dict(positive=np.array([[1., 0.], [1., 1.], [0., 1.]]),
                 global_only=np.array([[1., 0.], [1., 0.], [1., 0.]]),
                 no_subset=np.array([[-1.], [0.], [1.]]))
    supports = list(itertools.combinations(range(3), 2)) + [(0, 1, 2)]
    alpha = .05/(3 * sum(map(len, supports))*2)
    results = {}
    for name, means in cases.items():
        rows = []
        for support in supports:
            geometry = gaussian_signal(means, np.eye(means.shape[1]), support)
            differences = means[list(support)[1:]] - means[support[0]]
            _, singular, vh = np.linalg.svd(differences, full_matrices=True)
            rank = np.count_nonzero(singular > 1e-10)
            null = vh[rank:]
            direction = null.T @ (null @ means[support[0]])
            if np.linalg.norm(direction) < 1e-10:
                direction = np.eye(means.shape[1])[0]
            direction /= np.linalg.norm(direction)
            agreements = [.5 * math.erfc(-float(means[d]@direction)/math.sqrt(2)) for d in support]
            calibration = [sampled_cell(rng, p, n)[0] for p in agreements]
            audit_pairs = [sampled_cell(rng, p, n) for p in agreements]
            w, solver = solve_common(calibration, 2)
            audit = audit_head(w, [c for c, _ in audit_pairs], [c for _, c in audit_pairs],
                               1., .12, .05, alpha)
            rows.append(dict(support=support, gaussian_ssi_geometry=geometry,
                             sign_map_direction=direction.tolist(), agreements=agreements,
                             calibration=solver, audit=audit))
        results[name] = rows
    assert all(r['audit']['accepted'] for r in results['positive'][:3])
    assert not results['positive'][-1]['audit']['accepted']
    assert all(r['audit']['accepted'] for r in results['global_only'])
    assert not any(r['audit']['accepted'] for r in results['no_subset'])
    return dict(seed=0, n_per_domain_per_stage=n, tau=.12, delta=.05, alpha=.05,
                cases=results,
                conclusions=dict(positive='proper supports certified, global candidate not certified',
                                 global_only='common global mechanism; duplicate maps provide no additional population prediction',
                                 no_subset='no candidate certified; global fallback required'),
                limitation='Controlled fixed-map certificate test; not a test of neural discovery or router recovery.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/predictive_compatibility/synthetic_seed0.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run()
    write_json(Path(args.output), result)
    print(result['conclusions'])


if __name__ == '__main__':
    main()
