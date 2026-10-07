"""Source-only saved-state soft-min responsibility and output-gradient audit.

Reads the locked source-out replay, never target caches. No fitting, updates,
temperature sweep or checkpoint selection. Output-gradient similarity is not
parameter-gradient similarity or evidence of causal factors.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import multiple_choice_loss, expert_ce


def run(destination):
    torch.set_num_threads(4)
    config = json.loads((destination / 'plan.json').read_text())['hparams']['OFF']['sirm_discover_config']
    with np.load(destination / 'eda/source_out_replay.npz') as data:
        g = torch.tensor(data['OFF_g'], dtype=torch.float64)
        delta = torch.tensor(data['OFF_delta'], dtype=torch.float64, requires_grad=True)
        y = torch.tensor(data['y'], dtype=torch.long)
        domains = torch.tensor(data['domains'], dtype=torch.long)
    temperature = config['discovery_temperature']
    loss, responsibility = multiple_choice_loss(g, delta, y, temperature)
    gradient, = torch.autograd.grad(loss, delta)
    losses = expert_ce(g[:, None] + delta, y)
    global_ce = F.cross_entropy(g, y, reduction='none')
    entropy = -(responsibility * responsibility.clamp_min(1e-30).log()).sum(1)
    best_two = losses.sort(1).values[:, :2]
    flat = gradient.transpose(0, 1).flatten(1)
    normalized = F.normalize(flat, dim=1)
    cosine = normalized @ normalized.T
    groups = {}
    for d in domains.unique(sorted=True).tolist():
        for state in ['all', 'global_correct', 'global_wrong']:
            mask = domains.eq(d)
            if state != 'all':
                correct = g.argmax(1).eq(y)
                mask &= correct if state == 'global_correct' else ~correct
            if not mask.any():
                continue
            groups[f'{d}/{state}'] = dict(
                n=int(mask.sum()),
                normalized_entropy=float(entropy[mask].mean() / np.log(delta.shape[1])),
                mean_max_responsibility=float(responsibility[mask].max(1).values.mean()),
                mean_responsibility=responsibility[mask].mean(0).detach().tolist(),
                best_second_gap_over_temperature=float(((best_two[:, 1]-best_two[:, 0])[mask]/temperature).mean()),
                single_expert_CE_gain=(global_ce[:, None]-losses)[mask].mean(0).detach().tolist(),
            )
    result = dict(
        scope='Source-out original cached predictions at the source-selected OFF state only. No target reads, fitting, parameter updates, sweep or new selection. Source-out was previously used for selection.',
        temperature=temperature, images=len(y), groups=groups,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        replay_sha256=hashlib.sha256((destination/'eda/source_out_replay.npz').read_bytes()).hexdigest(),
        selection_lock=json.loads((destination/'eda/checkpoint_locks.json').read_text()),
        normalized_entropy=float(entropy.mean()/np.log(delta.shape[1])),
        mean_max_responsibility=float(responsibility.max(1).values.mean()),
        fraction_max_responsibility_over_half=float((responsibility.max(1).values>.5).double().mean()),
        output_gradient_cosine_matrix=cosine.detach().tolist(),
        mean_off_diagonal_output_gradient_cosine=float(cosine[~torch.eye(delta.shape[1], dtype=torch.bool)].mean()),
        limitation='Gradients are with respect to per-image expert output logits at this saved state, not head parameters or historical training gradients. Near-uniform responsibility supports weak competition on this split, not proof that changing temperature will improve transfer.',
    )
    out = destination / 'discovery_responsibility_diagnosis'
    out.mkdir(exist_ok=True)
    (out / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    result = run(args.destination)
    print(json.dumps({k: result[k] for k in ['normalized_entropy', 'mean_max_responsibility', 'fraction_max_responsibility_over_half', 'mean_off_diagonal_output_gradient_cosine']}))
