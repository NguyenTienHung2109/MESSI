"""Source-only linear accessibility probe for a saved bank's detached teacher.

This diagnostic fit is distinct from the completed400-update B1/B2 experiment.
No residual/backbone/global training, target access, grid or checkpoint selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.sirm_discover.math import residual_utilities, soft_null_q
from .shrinkage import digest, write


def score(a, pi):
    g, delta, y, domains, teacher = [a[k] for k in ['g', 'delta', 'y', 'domains', 'teacher']]
    full, _, utility, _ = residual_utilities(g, delta, pi, y)
    global_good = g.argmax(1).eq(y)
    good = full.argmax(1).eq(y)
    rows, means = {}, []
    for d in sorted(domains.unique().tolist()):
        mask = domains.eq(d)
        row = dict(n=int(mask.sum()), accuracy=float(good[mask].double().mean()),
                   global_accuracy=float(global_good[mask].double().mean()),
                   CE=float(F.cross_entropy(full[mask], y[mask])),
                   global_CE=float(F.cross_entropy(g[mask], y[mask])),
                   teacher_KL=float(F.kl_div(pi[mask].clamp_min(1e-30).log(), teacher[mask], reduction='batchmean')),
                   router_NULL=float(pi[mask, -1].mean()),
                   repairs=int((mask & ~global_good & good).sum()),
                   harms=int((mask & global_good & ~good).sum()))
        rows[str(d)] = row; means.append(utility[mask].mean(0))
    u = torch.stack(means)
    q, q_null, confidence, v = soft_null_q(u, torch.ones_like(u, dtype=torch.bool),
                                          temperature=.05, utility_floor=.002, interaction_floor=.002)
    result = dict(per_domain=rows, equal_domain={
        k: float(np.mean([r[k] for r in rows.values()]))
        for k in ['accuracy', 'global_accuracy', 'CE', 'global_CE', 'teacher_KL', 'router_NULL']},
        pooled=dict(n=len(y), accuracy=float(good.double().mean()),
                    repairs=int((~global_good & good).sum()), harms=int((global_good & ~good).sum())),
        U=u.tolist(), V=v.tolist(), fresh_Q=q.tolist(), selected_cells=int((q>0).sum()),
        shared_experts=int(((q>0).sum(0)>=2).sum()),
        Q_scope='Direct source original-view means, not historical training EMA or a trained dynamic-Q trajectory')
    return result


def run(dest):
    torch.set_num_threads(4)
    src = dest / 'source_router_diagnosis'
    out = dest / 'nonlinear_teacher_probe'; out.mkdir(exist_ok=True)
    paths = {split: src / (split + '_original.npz') for split in ['source_in', 'source_out']}
    locks = {split: digest(path) for split, path in paths.items()}
    checkpoint = dest / 'experiment/runs/L38/OFF/source_best.pkl'
    checkpoint_sha = digest(checkpoint)
    l2 = .01
    plan = dict(checkpoint_sha256=checkpoint_sha, source_cache_hashes=locks,
                code_sha256=digest(Path(__file__)), fit_split='source_in', evaluation_split='source_out',
                teacher='Existing detached per-image single-utility teacher at saved selected375 bank',
                solver='LBFGS, strong_wolfe, max_iter1000, tolerance_grad1e-4, tolerance_change1e-10',
                L2=l2, weighting='Equal-source-domain soft-target cross entropy',
                hypothesis='Linear router may miss nonlinear label-free teacher policies on hard source errors; fixed nonlinear basis isolates accessibility without jointly changing bank.',
                nonlinear_features='Append256 fixed tanh random features to standardized CLS, seed0 Gaussian weights std1/sqrt384, source-in feature mean/std normalization; width matches existing residual_hidden256, no grid. Fit output linear weights only; convex teacher objective.',
                comparator_code_sha256=digest(Path(__file__).with_name('fixed_teacher_probe.py')),
                feature_standardization='source-in mean/std only; initialized to the exact current linear router function',
                validation_selection=False, hyperparameter_grid=False, target_read=False,
                primary_B1_B2_modified=False,
                scope='Diagnostic teacher linear accessibility at a fixed bank. Additional probe solver iterations are not the400-update primary Stage2 protocol or a new SIRM treatment. No deployable factor/invariance/OOD claim follows.')
    write(out / 'plan.json', plan)
    arrays = {}
    for split, path in paths.items():
        with np.load(path) as data:
            arrays[split] = {k: torch.tensor(data[k]) for k in ['z', 'g', 'delta', 'pi', 'teacher', 'y', 'domains']}
    a = arrays['source_in']
    mean = a['z'].double().mean(0)
    std = a['z'].double().std(0, unbiased=False).clamp_min(1e-8)
    z = (a['z'].double() - mean) / std
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weight = payload['model_dict']['model.router.weight'].double()
    bias = payload['model_dict']['model.router.bias'].double()
    generator = torch.Generator().manual_seed(0)
    random_weight = torch.randn(384, 256, generator=generator, dtype=torch.double) / (384 ** .5)
    hidden_train = torch.tanh(z @ random_weight)
    hidden_mean = hidden_train.mean(0)
    hidden_std = hidden_train.std(0, unbiased=False).clamp_min(1e-8)
    def expand(features):
        return torch.cat([features, (torch.tanh(features @ random_weight)-hidden_mean)/hidden_std],1)
    z = expand(z)
    w = torch.cat([weight * std[None], torch.zeros(7,256,dtype=torch.double)],1).clone().requires_grad_()
    b = (bias + weight @ mean).clone().requires_grad_()
    initial_pi = (z @ w.T + b).softmax(1)
    max_difference = float((initial_pi.detach() - a['pi']).abs().max())
    assert max_difference < 1e-5
    sample_weights = torch.zeros(len(z), dtype=torch.double)
    ds = a['domains'].unique()
    for d in ds:
        mask = a['domains'].eq(d)
        sample_weights[mask] = 1 / (len(ds) * int(mask.sum()))
    teacher = a['teacher'].double()
    assert torch.allclose(teacher.sum(1), torch.ones(len(z), dtype=torch.double), atol=1e-6)
    optimizer = torch.optim.LBFGS([w, b], max_iter=1000, tolerance_grad=1e-4,
                                  tolerance_change=1e-10, line_search_fn='strong_wolfe')
    history = []
    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = z @ w.T + b
        data_loss = (-(teacher * F.log_softmax(logits, 1)).sum(1) * sample_weights).sum()
        penalty = .5 * l2 * w.square().sum()
        objective = data_loss + penalty
        assert torch.isfinite(objective)
        objective.backward()
        history.append(dict(closure=len(history), data_loss=float(data_loss.detach()),
                            penalty=float(penalty.detach()), objective=float(objective.detach())))
        return objective
    optimizer.step(closure)
    closure()
    max_grad = max(float(w.grad.abs().max()), float(b.grad.abs().max()))
    state = optimizer.state[w]
    results = {}
    with torch.no_grad():
        for split, a in arrays.items():
            features = expand((a['z'].double() - mean) / std)
            pi = (features @ w.T + b).softmax(1).float()
            results[split] = dict(actual=score(a, a['pi']),
                                  teacher_probe=score(a, pi), label_teacher=score(a, a['teacher']))
            np.savez_compressed(out / (split + '_policies.npz'), probe_pi=pi.numpy(),
                                actual_pi=a['pi'].numpy(), y=a['y'].numpy(), domains=a['domains'].numpy())
    assert digest(checkpoint) == checkpoint_sha
    assert all(digest(path) == locks[split] for split, path in paths.items())
    result = dict(plan=plan, initial_router_probability_max_difference=max_difference,
                  LBFGS_iterations=state.get('n_iter'), closure_evaluations=len(history),
                  final_objective_max_gradient=max_grad, converged_at_requested_gradient_tolerance=max_grad<=1e-4,
                  history=history, splits=results,
                  limitations='One preset regularized convex probe on source-normalized fixed tanh-expanded original source-in features. Source-out was already used to select the bank. '
                              'A useful diagnostic fit is not independent validation, not joint H2 training, and not proof that teacher loss or a larger router generalizes to target.')
    write(out / 'results.json', result)
    torch.save(dict(weight_standardized=w.detach(), bias_standardized=b.detach(),
                    source_mean=mean, source_std=std, random_weight=random_weight, hidden_mean=hidden_mean, hidden_std=hidden_std, plan=plan), out / 'diagnostic_router.pt')
    print(json.dumps(dict(status='complete', iterations=state.get('n_iter'),
                          max_grad=max_grad, source_out=results['source_out']['teacher_probe']['equal_domain'])))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('destination', type=Path)
    run(parser.parse_args().destination.resolve())
