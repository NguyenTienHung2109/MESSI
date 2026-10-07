"""Source-only utility/assignment alignment at locked saved states; no fitting."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .hard_router_diagnosis import outputs
from .centered_model import CenteredGateSIRM
from .shrinkage import digest, write
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover.math import soft_null_q

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/sirm_research_loop'
DEST = BASE / 'iteration_09/utility_alignment_diagnosis'


def utility_arrays(a):
    g = torch.tensor(a['g']).double()
    delta = torch.tensor(a['delta']).double()
    pi = torch.tensor(a['pi']).double()
    y = torch.tensor(a['y']).long()
    term = pi[:, :6, None] * delta
    full = g + term.sum(1)
    def ce(x):
        n, m, c = x.shape
        return F.cross_entropy(x.reshape(-1, c), y[:, None].expand(n, m).reshape(-1), reduction='none').reshape(n, m)
    base = F.cross_entropy(g, y, reduction='none')[:, None]
    full_ce = F.cross_entropy(full, y, reduction='none')[:, None]
    u = {'single': (base - ce(g[:, None] + delta)).numpy(),
         'routed_single': (base - ce(g[:, None] + term)).numpy(),
         'marginal': (ce(full[:, None] - term) - full_ce).numpy()}
    assert np.allclose(u['single'], a['single'], atol=3e-6)
    assert np.allclose(u['marginal'], a['marginal'], atol=3e-6)
    return u


def compare(x, y):
    x, y = x.flatten(), y.flatten()
    valid = (abs(x) > 1e-9) & (abs(y) > 1e-9)
    xp = x > 1e-9
    return {'cells': len(x), 'nonzero_cells': int(valid.sum()),
            'sign_agreement_nonzero': float(((x[valid] > 0) == (y[valid] > 0)).mean()) if valid.any() else None,
            'positive_first_cells': int(xp.sum()),
            'positive_first_to_nonpositive_second': int((xp & (y <= 1e-9)).sum()),
            'pearson': float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None,
            'median_abs_first': float(np.median(abs(x))),
            'median_abs_second': float(np.median(abs(y)))}


def aggregate(a, u):
    wrong = a['g'].argmax(1) != a['y']
    masks = {'all': np.ones(len(wrong), bool), 'global_wrong': wrong, 'global_correct': ~wrong}
    for d in np.unique(a['domains']):
        masks[f'domain/{d}'] = a['domains'] == d
        for c in np.unique(a['y']):
            mask = (a['domains'] == d) & (a['y'] == c)
            if mask.any():
                masks[f'domain/{d}/class/{c}'] = mask
    result = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        result[name] = {'images': int(mask.sum()),
                        'mean_utilities': {k: v[mask].mean(0).tolist() for k, v in u.items()},
                        'single_vs_routed': compare(u['single'][mask], u['routed_single'][mask]),
                        'routed_vs_context': compare(u['routed_single'][mask], u['marginal'][mask]),
                        'single_vs_context': compare(u['single'][mask], u['marginal'][mask])}
    return result


def pair_transfer(raw, us):
    rows = []
    for state in ['all', 'global_wrong', 'global_correct']:
        means = {}
        for split, a in raw.items():
            wrong = a['g'].argmax(1) != a['y']
            mask = np.ones(len(wrong), bool) if state == 'all' else (wrong if state == 'global_wrong' else ~wrong)
            means[split] = {(int(c), int(d)): {k: u[mask & (a['y'] == c) & (a['domains'] == d)].mean(0)
                                            for k, u in us[split].items()}
                            for c in np.unique(a['y']) for d in np.unique(a['domains'])
                            if (mask & (a['y'] == c) & (a['domains'] == d)).sum() >= 2}
        for c in np.unique(raw['source_in']['y']):
            for d1, d2 in itertools.combinations(np.unique(raw['source_in']['domains']), 2):
                keys = [(int(c), int(d1)), (int(c), int(d2))]
                if any(k not in means[s] for s in means for k in keys):
                    continue
                worst = np.minimum(*(means['source_in'][k]['single'] for k in keys))
                weights = (worst == worst.max()).astype(float); weights /= weights.sum()
                rows.append({'stratum': state, 'class': int(c), 'domains': [int(d1), int(d2)],
                             'source_in_winner': weights.tolist(),
                             'worst_endpoint_U': {s: {kind: float(np.minimum(*(means[s][k][kind] for k in keys)) @ weights)
                                                         for kind in us[s]} for s in means}})
    summaries = {}
    for state in ['all', 'global_wrong', 'global_correct']:
        rr = [r for r in rows if r['stratum'] == state]
        summaries[state] = {'matched_class_domain_pairs': len(rr),
                            'positive_train_single': sum(r['worst_endpoint_U']['source_in']['single'] > 0 for r in rr),
                            'positive_both': {k: sum(r['worst_endpoint_U']['source_in']['single'] > 0 and
                                                    r['worst_endpoint_U']['source_out'][k] > 0 for r in rr)
                                              for k in ['single', 'routed_single', 'marginal']}}
        summaries[state]['above_002_both'] = {
            k: sum(r['worst_endpoint_U']['source_in']['single'] > .002 and
                   r['worst_endpoint_U']['source_out'][k] > .002 for r in rr)
            for k in ['single', 'routed_single', 'marginal']}
    return {'summary': summaries, 'rows': rows}


def domain_aggregation(raw, us):
    domains = sorted(np.unique(raw['source_in']['domains']).tolist())
    classes = sorted(set.intersection(*(set(np.unique(a['y']).tolist()) for a in raw.values())))
    matched = [c for c in classes if all(((a['y'] == c) & (a['domains'] == d)).sum() >= 2
                                         for a in raw.values() for d in domains)]
    out = {'matched_classes_min2_all_domains_both_splits': matched, 'splits': {}}
    for split, a in raw.items():
        wrong = a['g'].argmax(1) != a['y']; u = us[split]['marginal']; rows = {}
        policies = {'original_image_weight': [], 'matched_class_equal_weight': [],
                    'equal_correct_wrong_within_domain': []}
        for d in domains:
            dm = a['domains'] == d
            conditional = {c: u[dm & (a['y'] == c)].mean(0) for c in classes if (dm & (a['y'] == c)).any()}
            pos = sum((dm & (a['y'] == c)).sum() / dm.sum() * np.maximum(v, 0) for c, v in conditional.items())
            neg = sum((dm & (a['y'] == c)).sum() / dm.sum() * np.maximum(-v, 0) for c, v in conditional.items())
            assert np.allclose(pos - neg, u[dm].mean(0), atol=1e-12)
            policies['original_image_weight'].append(u[dm].mean(0))
            policies['matched_class_equal_weight'].append(np.mean([conditional[c] for c in matched], axis=0))
            assert (dm & wrong).any() and (dm & ~wrong).any()
            policies['equal_correct_wrong_within_domain'].append((u[dm & wrong].mean(0) + u[dm & ~wrong].mean(0)) / 2)
            rows[str(d)] = {'weighted_positive_class_contributions': pos.tolist(),
                            'weighted_negative_class_contributions': neg.tolist(),
                            'class_mean_U': {str(k): v.tolist() for k, v in conditional.items()},
                            'global_wrong_U': u[dm & wrong].mean(0).tolist(),
                            'global_correct_U': u[dm & ~wrong].mean(0).tolist()}
        summaries = {}
        for name, matrix in policies.items():
            U = torch.tensor(np.array(matrix)).double()
            q, null, confidence, V = soft_null_q(U, torch.ones_like(U, dtype=torch.bool))
            summaries[name] = {'U': U.tolist(), 'V': V.tolist(),
                               'U_floor_cells': int((U > .002).sum()), 'V_floor_cells': int((V > .002).sum()),
                               'selected_cells': int((q > 0).sum()), 'shared_experts': int(((q > 0).sum(0) >= 2).sum())}
        out['splits'][split] = {'class_contribution_decomposition': rows, 'fixed_floor_fresh_Q_diagnostics': summaries}
    return out


def conditional_q(aggregation):
    out = {}
    for split, x in aggregation['splits'].items():
        rows = {}
        for c in aggregation['matched_classes_min2_all_domains_both_splits']:
            U = torch.tensor([x['class_contribution_decomposition'][str(d)]['class_mean_U'][str(c)]
                              for d in [0, 2, 3]], dtype=torch.float64)
            q, _, _, V = soft_null_q(U, torch.ones_like(U, dtype=torch.bool))
            rows[str(c)] = {'U': U.tolist(), 'V': V.tolist(), 'selected_mask': (q > 0).tolist(),
                            'selected_cells': int((q > 0).sum()),
                            'shared_experts': int(((q > 0).sum(0) >= 2).sum())}
        out[split] = {'rows': rows, 'total_selected_cells': sum(r['selected_cells'] for r in rows.values()),
                      'class_expert_shared_terms': sum(r['shared_experts'] for r in rows.values())}
    common = {}
    for c in out['source_in']['rows']:
        mask = np.array(out['source_in']['rows'][c]['selected_mask']) & np.array(out['source_out']['rows'][c]['selected_mask'])
        common[c] = {'common_selected_cells': int(mask.sum()), 'common_shared_experts': int((mask.sum(0) >= 2).sum())}
    out['common_support_across_source_splits'] = common
    return out


def main():
    torch.set_num_threads(4)
    paths = {it: {s: BASE / f'iteration_{it}/source_router_diagnosis/{s}_original.npz'
                  for s in ['source_in', 'source_out']} for it in ['08', '09']}
    batches = sorted((BASE / 'iteration_08/pair_shared_parameter_audit').glob('H2_step*.pt'))
    locks = {str(p.relative_to(ROOT)): digest(p) for v in paths.values() for p in v.values()}
    locks.update({str(p.relative_to(ROOT)): digest(p) for p in batches})
    checkpoints = {it: BASE / f'iteration_{it}/experiment/runs/L38/OFF/source_best.pkl' for it in paths}
    locks.update({str(p.relative_to(ROOT)): digest(p) for p in checkpoints.values()})
    dependencies = [Path(__file__).with_name(f) for f in ['hard_router_diagnosis.py', 'centered_model.py', 'shrinkage.py']]
    dependencies += [ROOT / 'experiments/sirm_discover/math.py']
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), *dependencies]}
    anchors = []
    for it in paths:
        p = BASE / f'iteration_{it}/plan.json'
        locks[str(p.relative_to(ROOT))] = digest(p)
        pp = json.loads(p.read_text()); anchors.append((pp['anchor_sha256'], pp['manifest_sha256'], pp['feature_cache_sha256']))
    assert anchors[0] == anchors[1]
    plan = {'code_sha256': digest(Path(__file__)), 'code_hashes': code_hashes, 'input_hashes': locks,
            'scope': 'Source-only selected saved states (08 step375, 09 step250); no fitting, updates, target reads or selection. '
                     'Float64 CE recomputation; nonzero sign comparisons exclude |U|<=1e-9. '
                     'Class/domain/wrong-conditioned winners use labels, not deployable selectors. '
                     'H2 batches at 200/215/250/300/350/395 are saved-state replays, not historical outputs. '
                     'Source-out was used for checkpoint selection and is exploratory.'}
    DEST.mkdir(exist_ok=True); write(DEST / 'plan.json', plan)
    results = {}
    raw_all = {}
    for it, pp in paths.items():
        raw = {s: dict(np.load(p)) for s, p in pp.items()}
        raw_all[it] = raw
        us = {s: utility_arrays(a) for s, a in raw.items()}
        results[it] = {'original': {s: aggregate(a, us[s]) for s, a in raw.items()},
                       'pair_transfer': pair_transfer(raw, us),
                       'domain_aggregation': domain_aggregation(raw, us), 'H2': []}
        results[it]['conditional_Q'] = conditional_q(results[it]['domain_aggregation'])
        payload = torch.load(checkpoints[it], map_location='cpu', weights_only=False)
        c = payload['config']; backbone = DeiTFeaturizer(pretrained=False, model_name=c['backbone'])
        m = CenteredGateSIRM(backbone, backbone.n_outputs, 10, 3, c)
        m.load_state_dict({k.removeprefix('model.'): v for k, v in payload['model_dict'].items()})
        m.eval(); m.requires_grad_(False)
        for path in batches:
            b = torch.load(path, map_location='cpu', weights_only=False)
            for mode in ['batch', 'saved']:
                with torch.no_grad(): f = outputs(m, b['z'], b['y'], mode)
                a = {k: v.numpy() for k, v in f.items()}
                a.update(y=b['y'].numpy(), domains=b['domains'].numpy())
                results[it]['H2'].append({'scheduled_step': b['scheduled_step'], 'mean_mode': mode,
                                         'metrics': aggregate(a, utility_arrays(a))})
        assert all(p.grad is None for p in m.parameters())
    for split in ['source_in', 'source_out']:
        for key in ['ids', 'y', 'domains']:
            assert np.array_equal(raw_all['08'][split][key], raw_all['09'][split][key]), key
        assert np.array_equal(raw_all['08'][split]['g'], raw_all['09'][split]['g'])
    assert all(digest(ROOT / p) == h for p, h in locks.items())
    assert all(digest(ROOT / p) == h for p, h in code_hashes.items())
    write(DEST / 'results.json', {'plan': plan, 'iterations': results})
    write(DEST / 'conditional_Q_diagnostic.json', {
        'scope': 'Source-only eight matched classes min2 per class/domain both splits; unchanged Q formula/floors '
                 'applied separately per label class. Post-hoc assignment diagnostic only, no EMA/training/ESS or target. '
                 'Support need not agree across splits; no claim of invariant factors.',
        'code_sha256': plan['code_sha256'], 'input_hashes': locks,
        'iterations': {it: r['conditional_Q'] for it, r in results.items()}})
    write(DEST / 'completion_audit.json', {'complete': True, 'inputs_unchanged': True,
                                         'same_images_labels_domains_global': True,
                                         'target_read': False, 'parameter_updates': False,
                                         'scientific_goal_achieved': False})
    print({it: results[it]['pair_transfer']['summary'] for it in results})


if __name__ == '__main__': main()
