"""Fixed-bank direct classification router probe, source-only.

Matches the teacher probe's preset standardization, initialization, solver and
L2. Only the diagnostic fit objective changes from teacher CE to full-model CE.
This is not a new400-step SIRM run or evidence of target generalization.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .fixed_teacher_probe import score
from .shrinkage import digest, write


def run(dest):
    torch.set_num_threads(4)
    src = dest / 'source_router_diagnosis'
    out = dest / 'fixed_classification_probe'; out.mkdir(exist_ok=True)
    paths = {s:src / (s + '_original.npz') for s in ['source_in', 'source_out']}
    locks = {s:digest(path) for s,path in paths.items()}
    bank = dest / 'experiment/runs/L38/OFF/source_best.pkl'
    bank_sha = digest(bank)
    l2 = .01
    plan = dict(checkpoint_sha256=bank_sha, source_cache_hashes=locks,
                code_sha256=digest(Path(__file__)), scoring_code_sha256=digest(Path(__file__).with_name('fixed_teacher_probe.py')),
                fit_split='source_in', evaluation_split='source_out',
                objective='Equal-domain true-label CE of frozen global + routed fixed residuals, plus0.5*0.01*sum standardized-router weights squared',
                comparator='Prior fixed teacher probe: same data/standardization/current-router initialization/solver/L2; objective is the only diagnostic change',
                solver='LBFGS strong_wolfe max_iter1000 tolerance_grad1e-4 tolerance_change1e-10; classification through softmax mixture is nonconvex',
                hypothesis='Teacher-only fitting improved train but harmed validation. Direct predictive CE may reveal a surrogate mismatch; if it also fails, merely stronger router fitting is insufficient at this bank.',
                validation_selection=False, hyperparameter_grid=False, target_read=False, primary_B1_B2_modified=False,
                scope='Diagnostic fixed-bank objective comparison; no residual/global/backbone update or target access. Solver iteration budget is separate from primary400-update Stage2.')
    write(out / 'plan.json', plan)
    arrays = {}
    for split, path in paths.items():
        with np.load(path) as a:
            arrays[split] = {k:torch.tensor(a[k]) for k in ['z', 'g', 'delta', 'pi', 'teacher', 'y', 'domains']}
    a = arrays['source_in']
    mean = a['z'].double().mean(0)
    std = a['z'].double().std(0, unbiased=False).clamp_min(1e-8)
    z = (a['z'].double() - mean) / std
    payload = torch.load(bank, map_location='cpu', weights_only=False)
    weight = payload['model_dict']['model.router.weight'].double()
    bias = payload['model_dict']['model.router.bias'].double()
    w = (weight * std[None]).clone().requires_grad_()
    b = (bias + weight @ mean).clone().requires_grad_()
    assert float(((z @ w.T + b).softmax(1).detach() - a['pi']).abs().max()) < 1e-5
    weights = torch.zeros(len(z), dtype=torch.double)
    ds = a['domains'].unique()
    for d in ds:
        mask = a['domains'].eq(d); weights[mask] = 1 / (len(ds) * int(mask.sum()))
    g, delta, y = a['g'].double(), a['delta'].double(), a['y']
    optimizer = torch.optim.LBFGS([w, b], max_iter=1000, tolerance_grad=1e-4,
                                  tolerance_change=1e-10, line_search_fn='strong_wolfe')
    history = []
    def closure():
        optimizer.zero_grad(set_to_none=True)
        pi = (z @ w.T + b).softmax(1)
        full = g + (pi[:, :6, None] * delta).sum(1)
        data_loss = (F.cross_entropy(full, y, reduction='none') * weights).sum()
        penalty = .5 * l2 * w.square().sum()
        objective = data_loss + penalty
        assert torch.isfinite(objective)
        objective.backward()
        history.append(dict(closure=len(history), data_loss=float(data_loss.detach()),
                            penalty=float(penalty.detach()), objective=float(objective.detach())))
        return objective
    optimizer.step(closure); closure()
    gradient = max(float(w.grad.abs().max()), float(b.grad.abs().max()))
    results = {}
    with torch.no_grad():
        for split, a in arrays.items():
            z = (a['z'].double() - mean) / std
            pi = (z @ w.T + b).softmax(1).float()
            results[split] = dict(actual=score(a, a['pi']), classification_probe=score(a, pi))
            np.savez_compressed(out / (split + '_policies.npz'), probe_pi=pi.numpy(),
                                actual_pi=a['pi'].numpy(), y=a['y'].numpy(), domains=a['domains'].numpy())
    assert digest(bank) == bank_sha and all(digest(path) == locks[s] for s,path in paths.items())
    result = dict(plan=plan, LBFGS_iterations=optimizer.state[w].get('n_iter'),
                  final_objective_max_gradient=gradient, converged_at_requested_gradient_tolerance=gradient<=1e-4,
                  history=history, splits=results,
                  limitations='Nonconvex diagnostic fit from one initialization and one fixed regularization. Source-out already selected the bank; no independently confirmed result or target selection. '
                              'Global fallback can lower loss by suppressing harmful corrections; improved CE alone is not factor recovery.')
    write(out / 'results.json', result)
    torch.save(dict(weight_standardized=w.detach(), bias_standardized=b.detach(), source_mean=mean, source_std=std,
                    plan=plan), out / 'diagnostic_router.pt')
    print(json.dumps(dict(status='complete', iterations=result['LBFGS_iterations'], max_grad=gradient,
                          source_out=results['source_out']['classification_probe']['equal_domain'])))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('destination', type=Path)
    run(parser.parse_args().destination.resolve())
