"""Post-hoc class-conditioned domain/nuisance diagnostics; never train a bank.

New domain probes use source-in/source-out only. Target features are read after
their selection is locked. All inference runs on CPU using saved CLS caches.
"""
import argparse
import itertools
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.deit_transformer import ExplicitMoEHead
from experiments.h2_log_analysis import digest, save_csv
from experiments.sirm_h1_h6.statistics import fit_linear_probe, probe_logits

DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
REGULARIZATION = [.001, .01, .1]
EPS = 1e-12
FILES = ['class_cond_domain_distance', 'domain_probe', 'nuisance_displacement',
         'nuisance_projection', 'classifier_nuisance_reliance']


def load_cache(run, bank, p, kind, split, view):
    path = run / 'cache' / f'{kind}_{split}_{view}.pt'
    if not path.exists():
        path = bank / 'cache' / path.name
    c = torch.load(path, map_location='cpu', weights_only=False)
    assert c['metadata']['checkpoint_sha256'] == p['checkpoint_sha256']
    assert c['metadata']['split_hash'] == p['split_hash']
    assert c['metadata']['kind'] == kind and c['metadata']['view'] == view
    if kind == 'source':
        assert p['target_env'] not in c['physical_domains'].unique().tolist()
    c['z'] = c.get('backbone_feature', c['projected_feature'])
    return c


def paired(a, b):
    assert a['image_ids'] == b['image_ids']
    assert torch.equal(a['labels'], b['labels'])
    assert torch.equal(a['physical_domains'], b['physical_domains'])


def mean(values):
    return statistics.mean(values)


def centroids(cache):
    out, counts = {}, {}
    for domain in cache['physical_domains'].unique().tolist():
        for label in cache['labels'].unique().tolist():
            mask = (cache['physical_domains'] == domain) & (cache['labels'] == label)
            if mask.any():
                out[domain, label] = cache['z'][mask].double().mean(0)
                counts[domain, label] = int(mask.sum())
    return out, counts


def class_macro(values, labels):
    return mean(float(values[labels == c].mean()) for c in labels.unique())


def domain_balanced(values, labels, domains):
    return mean(float(values[(labels == c) & (domains == d)].mean())
                for c in labels.unique() for d in domains.unique()
                if ((labels == c) & (domains == d)).any())


def domain_probes(fit, val, identity, destination):
    assert not (set(fit['image_ids']) & set(val['image_ids']))
    sources = sorted(fit['physical_domains'].unique().tolist())
    assert sources == sorted(val['physical_domains'].unique().tolist())
    rows, selected = [], {}
    for label in sorted(fit['labels'].unique().tolist()):
        mask = fit['labels'] == label
        vm = val['labels'] == label
        x, vx = fit['z'][mask].double(), val['z'][vm].double()
        domains, vd = fit['physical_domains'][mask], val['physical_domains'][vm]
        assert all((domains == d).any() and (vd == d).any() for d in sources)
        y = torch.tensor([sources.index(int(d)) for d in domains])
        vy = torch.tensor([sources.index(int(d)) for d in vd])
        mu, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
        weights = torch.zeros(len(x), dtype=torch.double)
        for d in sources:
            dm = domains == d
            weights[dm] = 1 / (len(sources) * int(dm.sum()))
        candidates = []
        for reg in REGULARIZATION:
            probe = fit_linear_probe(x, y, len(sources), mu, std,
                regularization=reg, max_iter=1000, tolerance=1e-4, sample_weights=weights)
            logits = probe_logits(probe, vx)
            ce = F.cross_entropy(logits, vy, reduction='none')
            correct = logits.argmax(1).eq(vy).double()
            score = mean(float(ce[vd == d].mean()) for d in sources)
            accuracy = mean(float(correct[vd == d].mean()) for d in sources)
            candidates.append((score, reg, probe, accuracy, float(correct.mean())))
        best = min(candidates, key=lambda item: item[0])
        selected[label] = dict(probe=best[2], regularization=best[1])
        for score, reg, probe, acc, image_acc in candidates:
            rows.append(dict(**identity, data_split='source_out', scope='class_candidate',
                label=label, regularization=reg, selected=reg == best[1],
                domain_accuracy=acc, image_domain_accuracy=image_acc,
                balanced_validation_nll=score, chance_accuracy=1/len(sources),
                gradient_max=probe['gradient_max'], source_in_samples=len(x), source_out_samples=len(vx)))
    locked = dict(probes=selected, sources=sources, class_order=sorted(selected),
                  checkpoint_sha256=identity['checkpoint_sha256'], target_used=False,
                  selection='per-class equal-source-domain out NLL, first minimum in fixed grid')
    torch.save(locked, destination)
    picks = [r for r in rows if r['selected']]
    rows.append(dict(**identity, data_split='source_out', scope='checkpoint',
                     domain_accuracy=mean(r['domain_accuracy'] for r in picks),
                     image_domain_accuracy=mean(r['image_domain_accuracy'] for r in picks),
                     balanced_validation_nll=mean(r['balanced_validation_nll'] for r in picks),
                     chance_accuracy=1/len(sources), classes=len(picks), target_used_for_probe=False))
    return rows


class LatentPredictor(nn.Module):
    """Exact selected HypothesesSIRM native expert-logit mixture after CLS."""
    def __init__(self, state, hp):
        super().__init__()
        assert hp['subset_irm_prediction_mode'] == 'expert_logit_mix'
        assert hp['moe_dim'] == 'auto' and hp['subset_irm_use_expert_heads']
        self.moe = ExplicitMoEHead(384, hp['num_experts'], 7, hp['expert_mlp_ratio'])
        self.moe.load_state_dict({k.removeprefix('moe_head.'): v for k, v in state.items() if k.startswith('moe_head.')}, strict=True)
        self.heads = nn.ModuleList([nn.Linear(384, 7) for _ in range(hp['num_experts'])])
        self.heads.load_state_dict({k.removeprefix('expert_heads.heads.'): v for k, v in state.items() if k.startswith('expert_heads.heads.')}, strict=True)
        assert isinstance(self.moe.input_proj, nn.Identity)
        self.topk = hp['subset_irm_router_topk']
        assert int(state['subset_step']) >= hp['subset_irm_topk_warmup_steps']
        self.eval().requires_grad_(False)

    def forward(self, z):
        dense = self.moe.router(z).softmax(1)
        indices = dense.topk(self.topk, dim=1).indices
        pi = dense * torch.zeros_like(dense).scatter_(1, indices, 1.)
        pi = pi / pi.sum(1, keepdim=True).clamp_min(1e-8)
        logits = torch.stack([head(expert(z)) for head, expert in zip(self.heads, self.moe.experts)], 1)
        return (pi.unsqueeze(-1) * logits).sum(1)


@torch.no_grad()
def infer(model, z):
    return torch.cat([model(part) for part in z.split(256)], 0)


def distances(identity, mu, counts, pairs, scope, scale):
    rows = []
    for label in range(7):
        for first, second in pairs:
            if (first, label) not in mu or (second, label) not in mu:
                continue
            a, b = mu[first, label], mu[second, label]
            v = a-b
            rows.append(dict(**identity, data_split=scope, scope='class_pair', label=label,
                first_domain=first, second_domain=second, first_samples=counts[first, label],
                second_samples=counts[second, label], euclidean_distance=float(v.norm()),
                cosine_distance=float(1-F.cosine_similarity(a, b, dim=0)),
                scale_normalized_distance=float(v.norm()/scale), source_feature_RMS_norm=scale))
    aggregate = aggregate_class_pair(rows, ['euclidean_distance', 'cosine_distance', 'scale_normalized_distance'])
    rows.append(dict(**identity, data_split=scope, scope='checkpoint', valid_pair_classes=len(rows), **aggregate))
    return rows


def aggregate_class_pair(rows, keys):
    # Class means first: unequal valid pair counts must not reweight classes.
    labels = sorted(set(r['label'] for r in rows))
    return {k: mean(mean(r[k] for r in rows if r['label'] == c and r[k] is not None)
                    for c in labels if any(r[k] is not None for r in rows if r['label'] == c))
            if any(r[k] is not None for r in rows) else None for k in keys}


def evaluate(file, out):
    result = json.loads(file.read_text())
    p = result['provenance']
    assert not p['target_used_for_model_or_hyperparameter_selection'] and not p['smoke']
    identity = {k: p[k] for k in ['condition', 'target_env', 'target_domain', 'seed', 'selected_step', 'checkpoint_sha256', 'split_hash']}
    identity.update(method='SIRM_ON' if p['sirm_enabled'] else 'SIRM_OFF', augmentation=p['augmentation'])
    bank, run = Path(p['bank']), file.parent
    assert digest(bank / 'checkpoints/best.pkl') == p['checkpoint_sha256']
    fit = load_cache(run, bank, p, 'source', 'in', 'original')
    source = load_cache(run, bank, p, 'source', 'out', 'original')
    records = {name: [] for name in FILES}
    probe_lock = out / 'domain_probes' / f"{p['condition']}_env{p['target_env']}_seed{p['seed']}.pt"
    records['domain_probe'] = domain_probes(fit, source, identity, probe_lock)
    probe_lock_hash = digest(probe_lock)
    del fit
    # Nuisance axes are estimated on source-out only, before target access.
    axes = {}
    source_views = {}
    for view in ['grayscale', 'blur']:
        cache = load_cache(run, bank, p, 'source', 'out', view)
        paired(source, cache)
        axes[view] = torch.stack([(source['z'][source['labels'] == c].double() -
                                  cache['z'][source['labels'] == c].double()).mean(0)
                                 for c in range(7)])
        source_views[view] = cache
    mu, counts = centroids(source)
    sources = sorted(source['physical_domains'].unique().tolist())
    pairs = list(itertools.combinations(sources, 2))
    feature_scale = float(source['z'].double().square().sum(1).mean().sqrt())
    records['class_cond_domain_distance'] += distances(identity, mu, counts, pairs, 'source_out', feature_scale)
    checkpoint = torch.load(bank / 'checkpoints/best.pkl', map_location='cpu', weights_only=False)
    model = LatentPredictor(checkpoint['model_dict'], p['hparams'])
    del checkpoint
    label_probe = torch.load(run / 'source_selected.pt', map_location='cpu', weights_only=False)['probe']
    verification = dict(**identity, probe_lock_sha256=probe_lock_hash, max_native_logit_replay_error=0.,
                        native_prediction_disagreements=0, target_used_for_probe=False,
                        nuisance_axis_split='source_out', zero_axes=0)
    for split, cache in [('source_out', source), ('target_all', None)]:
        if cache is None:
            cache = load_cache(run, bank, p, 'target', 'all', 'original')
            target_mu, target_counts = centroids(cache)
            assert list(cache['physical_domains'].unique()) == [p['target_env']]
            all_mu, all_counts = {**mu, **target_mu}, {**counts, **target_counts}
            target_pairs = [(d, p['target_env']) for d in sources]
            records['class_cond_domain_distance'] += distances(identity, all_mu, all_counts,
                target_pairs, 'target_vs_source_out', feature_scale)
        else:
            all_mu, all_counts, target_pairs = mu, counts, pairs
        original_logits = infer(model, cache['z'])
        pi = cache.get('router_native', cache['router_topk'])
        cached_logits = (pi.unsqueeze(-1)*cache['expert_logits']).sum(1)
        error = float((original_logits-cached_logits).abs().max())
        disagreement = int((original_logits.argmax(1) != cached_logits.argmax(1)).sum())
        verification['max_native_logit_replay_error'] = max(verification['max_native_logit_replay_error'], error)
        verification['native_prediction_disagreements'] += disagreement
        assert error < 1e-4 and disagreement == 0
        label_predictions = probe_logits(label_probe, cache['z']).argmax(1)
        label_correct = label_predictions.eq(cache['labels']).float()
        label_accuracy = (mean(float(label_correct[cache['physical_domains'] == d].mean())
                               for d in cache['physical_domains'].unique())
                          if split == 'source_out' else float(label_correct.mean()))
        records['domain_probe'].append(dict(**identity, data_split=split, scope='existing_label_probe',
            label_probe_accuracy=label_accuracy, label_probe_macro_class_accuracy=class_macro(label_correct, cache['labels']),
            label_probe_refitted=False, target_clean_accuracy=result['views']['original']['native']['accuracy']))
        if split == 'target_all':
            assert abs(label_accuracy - result['linear_probe']['original']['accuracy']) < 1e-7
        for view in ['grayscale', 'blur']:
            transformed = source_views[view] if split == 'source_out' else load_cache(run, bank, p, 'target', 'all', view)
            paired(cache, transformed)
            displacement = cache['z'].double()-transformed['z'].double()
            norms = displacement.norm(dim=1)
            cos_disp = 1-F.cosine_similarity(cache['z'].double(), transformed['z'].double(), dim=1)
            nuisance = axes[view]
            magnitudes = nuisance.norm(dim=1)
            valid_axis = magnitudes > EPS
            verification['zero_axes'] += int((~valid_axis).sum())
            unit = nuisance / magnitudes[:, None].clamp_min(EPS)
            raw_shifted = infer(model, cache['z']-nuisance[cache['labels']].float())
            unit_shifted = infer(model, cache['z']-unit[cache['labels']].float())
            raw_diff, unit_diff = original_logits-raw_shifted, original_logits-unit_shifted
            raw_norm, unit_norm = raw_diff.norm(dim=1), unit_diff.norm(dim=1)
            centered = raw_diff-raw_diff.mean(1, keepdim=True)
            centered_unit = unit_diff-unit_diff.mean(1, keepdim=True)
            for label in range(7):
                mask = cache['labels'] == label
                records['nuisance_displacement'].append(dict(**identity, data_split=split, view=view,
                    scope='class', label=label, samples=int(mask.sum()),
                    euclidean_displacement=float(norms[mask].mean()),
                    cosine_displacement=float(cos_disp[mask].mean()),
                    feature_RMS_norm=float(cache['z'][mask].double().square().sum(1).mean().sqrt()),
                    source_axis_magnitude=float(magnitudes[label])))
                records['classifier_nuisance_reliance'].append(dict(**identity, data_split=split, view=view,
                    scope='class', label=label, samples=int(mask.sum()), axis_valid=bool(valid_axis[label]),
                    logit_L2_change=float(raw_norm[mask].mean()), centered_logit_L2_change=float(centered[mask].norm(dim=1).mean()),
                    correct_class_abs_logit_change=float(raw_diff[mask, label].abs().mean()),
                    unit_axis_logit_L2_change=float(unit_norm[mask].mean()) if valid_axis[label] else None,
                    unit_axis_centered_logit_L2_change=float(centered_unit[mask].norm(dim=1).mean()) if valid_axis[label] else None,
                    prediction_change_fraction=float(original_logits[mask].argmax(1).ne(raw_shifted[mask].argmax(1)).float().mean()),
                    source_axis_magnitude=float(magnitudes[label]), nuisance_axis_split='source_out'))
                for first, second in target_pairs:
                    if (first, label) not in all_mu or (second, label) not in all_mu:
                        continue
                    v = all_mu[first, label]-all_mu[second, label]
                    distance = float(v.norm())
                    valid = bool(valid_axis[label]) and distance > EPS
                    projection = abs(float(v@unit[label])) if valid else None
                    records['nuisance_projection'].append(dict(**identity,
                        data_split='source_out' if split == 'source_out' else 'target_vs_source_out',
                        view=view, scope='class_pair', label=label, first_domain=first, second_domain=second,
                        domain_distance=distance, source_axis_magnitude=float(magnitudes[label]),
                        projection_magnitude=projection, projection_fraction=projection/(distance+EPS) if valid else None,
                        axis_valid=bool(valid_axis[label]), gap_valid=distance > EPS, nuisance_axis_split='source_out'))
            for name, keys in [
                    ('nuisance_displacement', ['euclidean_displacement', 'cosine_displacement', 'feature_RMS_norm', 'source_axis_magnitude']),
                    ('classifier_nuisance_reliance', ['logit_L2_change', 'centered_logit_L2_change', 'correct_class_abs_logit_change',
                        'unit_axis_logit_L2_change', 'unit_axis_centered_logit_L2_change', 'prediction_change_fraction', 'source_axis_magnitude']),
                    ('nuisance_projection', ['projection_magnitude', 'projection_fraction', 'source_axis_magnitude'])]:
                domain_split = 'target_vs_source_out' if name == 'nuisance_projection' and split == 'target_all' else split
                part = [r for r in records[name] if r['data_split'] == domain_split and r.get('view') == view and r['scope'] != 'checkpoint']
                records[name].append(dict(**identity, data_split=domain_split, view=view, scope='checkpoint',
                    **aggregate_class_pair(part, keys), valid_class_terms=len(part)))
            if split == 'target_all':
                del transformed
        if split == 'target_all':
            del cache
    assert digest(probe_lock) == probe_lock_hash
    return dict(records=records, verification=verification)


METRICS = {
    'class_cond_domain_distance': ['euclidean_distance', 'cosine_distance', 'scale_normalized_distance'],
    'domain_probe': ['domain_accuracy', 'image_domain_accuracy', 'label_probe_accuracy'],
    'nuisance_displacement': ['euclidean_displacement', 'cosine_displacement', 'feature_RMS_norm'],
    'nuisance_projection': ['projection_magnitude', 'projection_fraction'],
    'classifier_nuisance_reliance': ['logit_L2_change', 'centered_logit_L2_change', 'correct_class_abs_logit_change',
                                    'unit_axis_logit_L2_change', 'unit_axis_centered_logit_L2_change', 'prediction_change_fraction']}


def summarize(output, cells):
    records = {name: [row for cell in cells for row in cell['records'][name]] for name in FILES}
    for name, rows in records.items():
        save_csv(output / f'{name}.csv', rows)
    effects, aggregates = [], []
    for name, rows in records.items():
        part = [r for r in rows if r['scope'] in ['checkpoint', 'existing_label_probe']]
        for metric in METRICS[name]:
            measured = [r for r in part if r.get(metric) is not None]
            groups = sorted(set((r['data_split'], r.get('view', 'original')) for r in measured))
            for split, view in groups:
                select = [r for r in measured if (r['data_split'], r.get('view', 'original')) == (split, view)]
                for condition in 'ABCD':
                    values = [r[metric] for r in select if r['condition'] == condition]
                    if values:
                        aggregates.append(dict(file=name, metric=metric, data_split=split, view=view, condition=condition,
                            target_env='all', scope='cell_mean', n=len(values), mean=mean(values),
                            sample_SD=statistics.stdev(values) if len(values)>1 else None))
                    for target in range(4):
                        vals = [r[metric] for r in select if r['condition'] == condition and r['target_env'] == target]
                        if vals:
                            aggregates.append(dict(file=name, metric=metric, data_split=split, view=view, condition=condition,
                                target_env=target, scope='per_target_mean', n=len(vals), mean=mean(vals),
                                sample_SD=statistics.stdev(vals) if len(vals)>1 else None))
                lookup = {(r['condition'], r['target_env'], r['seed']): r for r in select}
                for first, second, effect in [('A', 'B', 'OFF'), ('C', 'D', 'ON')]:
                    deltas = []
                    for target in range(4):
                        for seed in range(3):
                            a, b = lookup.get((first, target, seed)), lookup.get((second, target, seed))
                            if a is None or b is None:
                                continue
                            assert a['split_hash'] == b['split_hash']
                            delta = b[metric]-a[metric]
                            deltas.append((target, seed, delta))
                            effects.append(dict(file=name, metric=metric, data_split=split, view=view, effect=effect,
                                scope='paired_cell', target_env=target, seed=seed, baseline=a[metric], H2=b[metric], delta=delta))
                    for target in list(range(4))+['all']:
                        values = [d for t,s,d in deltas if target == 'all' or t == target]
                        if values:
                            effects.append(dict(file=name, metric=metric, data_split=split, view=view, effect=effect,
                                scope='overall_paired_mean' if target == 'all' else 'per_target_paired_mean', target_env=target,
                                n=len(values), mean=mean(values), sample_SD=statistics.stdev(values) if len(values)>1 else None,
                                decreased_cells=sum(v<0 for v in values), increased_cells=sum(v>0 for v in values)))
                    blocks = [mean(d for t,s,d in deltas if s == seed) for seed in range(3)
                              if sum(s == seed for t,s,d in deltas) == 4]
                    if blocks:
                        effects.append(dict(file=name, metric=metric, data_split=split, view=view, effect=effect,
                            scope='equal_target_seed_block', target_env='all', n=len(blocks), mean=mean(blocks),
                            sample_SD=statistics.stdev(blocks) if len(blocks)>1 else None))
    save_csv(output / 'paired_effects.csv', effects)
    save_csv(output / 'aggregate_metrics.csv', aggregates)
    (output / 'verification.json').write_text(json.dumps([c['verification'] for c in cells], indent=2)+'\n')
    lines = ['# H2 latent nuisance diagnosis', '', f'Completed {len(cells)}/48 checkpoint analyses. Existing selected PACS checkpoints; no bank retraining, model selection changes, or hyperparameter tuning. New diagnostic domain probes are fitted on source-in only and locked before target access. Classifier replay uses CPU. CSV accuracy/fraction units are fractions.', '',
        '## Protocol', '',
        '- True pre-expert/router backbone CLS (384 dimensions, Identity input projection). Original deterministic view is primary. Source-in and source-out image IDs are disjoint; all gray/blur views are exactly image/label/domain paired.',
        '- Centroid distances average valid source-domain pairs within each class, then average classes equally. Euclidean distances are in raw feature units; cosine distance is 1−cos(centroids). Scale-normalized distance divides by source-out feature RMS norm to diagnose mere global feature rescaling.',
        '- A target-only cache has one domain: within-target cross-domain distance is undefined. Supplementary target_vs_source_out compares each target class centroid against the three source-out class centroids. It is a descriptive target-source comparison, not target-only separation.',
        '- Domain probe: seven separate 3-way linear softmax classifiers, one per label class. Each uses its source-in mean/std, equal-domain source-in CE weights, and L2 grid [0.001, 0.01, 0.1]. Select L2 separately per class by equal-domain source-out NLL (earliest grid tie). LBFGS max_iter=1000, gradient tolerance=1e-4. Primary accuracy averages domains within class and then classes; chance=1/3. Secondary image-weighted within-class accuracy is also saved. No target class/domain is fitted or selected.',
        '- Nuisance displacement averages image norms within class, then classes equally. Source-only nuisance axes are pooled class means of z(original)−z(view) from source-out; pooling within class follows the requested conditional expectation (sample weighting within class). These fixed source axes also define target-source projections and target classifier interventions. Zero axes/gaps are excluded from normalized projection, with validity recorded, never replaced by a spurious zero.',
        '- Projection P=|v·unit(n)|, F=P/(||v||+1e-12), averaged pairs within class then classes. Gray/blur are separate axes, not orthogonal components; do not sum their fractions or call F explained variance.',
        '- No single linear W maps CLS to native logits: experts are nonlinear and routing is input dependent. Classifier reliance replays the exact selected expert-logit mixture with recomputed Top-2 router at z and z−n_class. Primary S is mean logit-vector L2 change, equal-class averaged. Correct-class absolute change and class-centered logit L2 remove ambiguities about common-logit offsets. Unit-axis one-feature-unit interventions are supplementary and separate direction sensitivity from shortened displacement. These finite shifts can switch router support; they are not infinitesimal Jacobian derivatives.',
        '- Existing frozen label probes are replayed without refitting. Source score is equal-domain mean; target score is image mean, matching the previous report. Their L2 was already selected on source-out.',
        '- A/B=baseline/H2 SIRM OFF; C/D=baseline/H2 SIRM ON. This is the legacy six-expert architecture; OFF still retains expert/router auxiliary objectives. All reported means use the 12 paired target×seed cells. Cell SD is descriptive; shared-source overlap means 12 cells are not independent replicates. Per-target means/SD and paired raw differences are preserved; equal-target means within each of three seed blocks supply an additional SD.', '',
        '## Metrics and paired effects', '',
        '| Metric / split / view | A mean ± cell SD | B mean ± cell SD | B−A mean ± paired SD | C mean ± cell SD | D mean ± cell SD | D−C mean ± paired SD |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, metrics in METRICS.items():
        primary = {'class_cond_domain_distance': ['euclidean_distance','cosine_distance','scale_normalized_distance'],
                   'domain_probe': ['domain_accuracy','label_probe_accuracy'],
                   'nuisance_displacement': ['euclidean_displacement'],
                   'nuisance_projection': ['projection_magnitude','projection_fraction'],
                   'classifier_nuisance_reliance': ['logit_L2_change','unit_axis_centered_logit_L2_change']}[name]
        for metric in primary:
            groups = sorted(set((r['data_split'],r['view']) for r in aggregates if r['file']==name and r['metric']==metric))
            for split, view in groups:
                def fmt(condition):
                    match = [r for r in aggregates if r['file']==name and r['metric']==metric and r['data_split']==split and r['view']==view and r['condition']==condition and r['target_env']=='all']
                    return f"{match[0]['mean']:.6f} ± {match[0]['sample_SD']:.6f}" if len(match)==1 and match[0]['sample_SD'] is not None else 'pending'
                def eff(effect):
                    match = [r for r in effects if r['file']==name and r['metric']==metric and r['data_split']==split and r['view']==view and r['effect']==effect and r['scope']=='overall_paired_mean']
                    return f"{match[0]['mean']:+.6f} ± {match[0]['sample_SD']:.6f}" if len(match)==1 and match[0]['sample_SD'] is not None else 'pending'
                lines.append(f'| {metric} / {split} / {view} | {fmt("A")} | {fmt("B")} | {eff("OFF")} | {fmt("C")} | {fmt("D")} | {eff("ON")} |')
    lines += ['', '## Six direct questions', '']
    questions = [
        ('1. Does class-conditional domain separation decrease?', 'class_cond_domain_distance', 'euclidean_distance', 'source_out', 'original'),
        ('2. Does class-conditioned domain predictability decrease?', 'domain_probe', 'domain_accuracy', 'source_out', 'original'),
        ('3a. Does grayscale displacement shorten?', 'nuisance_displacement', 'euclidean_displacement', 'source_out', 'grayscale'),
        ('3b. Does blur displacement shorten?', 'nuisance_displacement', 'euclidean_displacement', 'source_out', 'blur'),
        ('4a. Does gray-aligned domain-shift fraction decrease?', 'nuisance_projection', 'projection_fraction', 'source_out', 'grayscale'),
        ('4b. Does blur-aligned domain-shift fraction decrease?', 'nuisance_projection', 'projection_fraction', 'source_out', 'blur'),
        ('5a. Does classifier gray-axis reliance decrease?', 'classifier_nuisance_reliance', 'logit_L2_change', 'source_out', 'grayscale'),
        ('5b. Does classifier blur-axis reliance decrease?', 'classifier_nuisance_reliance', 'logit_L2_change', 'source_out', 'blur'),
        ('6. Is label-probe target accuracy preserved/improved?', 'domain_probe', 'label_probe_accuracy', 'target_all', 'original')]
    for title,name,metric,split,view in questions:
        parts=[]
        for effect in ['OFF','ON']:
            match=[r for r in effects if r['file']==name and r['metric']==metric and r['data_split']==split and r['view']==view and r['effect']==effect and r['scope']=='overall_paired_mean']
            if match:
                r=match[0];parts.append(f"{effect}: mean H2−baseline {r['mean']:+.6f}; decrease in {r['decreased_cells']}/{r['n']} paired cells, increase in {r['increased_cells']}/{r['n']}")
        lines += [f'**{title}** '+('; '.join(parts) if parts else 'Pending.')+'.', '']
    lines += ['Interpretation is descriptive until all 48 cells complete. Lower linear domain-probe accuracy is a proxy for less linearly accessible conditional domain information, not a measurement of mutual information or proof of information removal. Reduced displacement demonstrates reduced variation under the tested transforms; it does not establish disappearance of spurious features or shape learning. Lower finite classifier sensitivity can reflect both shorter axes and predictor response: consult unit-axis/class-centered measures before attributing it to changed reliance.', '',
              'Raw per-class/pair values and per-checkpoint aggregates: '+', '.join(f'[{name}.csv]({name}.csv)' for name in FILES)+'.', '',
              '[Per-target/condition means and sample SD](aggregate_metrics.csv), [raw paired effects, per-target means/SD and seed-block SD](paired_effects.csv), [replay/probe locks](verification.json).']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


def main(root, output, limit=None):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    (output/'cells').mkdir(exist_ok=True)
    (output/'domain_probes').mkdir(exist_ok=True)
    files = sorted((root/'runs').glob('*/env*/seed*/metrics.json'))
    if limit:
        files=files[:limit]
    started=time.time()
    plan=dict(input_root=str(root.resolve()), code_sha256=digest(Path(__file__)),
              domain_probe_grid=REGULARIZATION, domain_probe_max_iter=1000, domain_probe_tolerance=1e-4,
              model_retraining=False, checkpoint_selection_changed=False, GPU_inference=False,
              nuisance_axis_split='source_out', expected_cells=len(files))
    if (output/'plan.json').exists():
        assert json.loads((output/'plan.json').read_text()) == plan, 'analysis protocol changed'
    else:
        (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    cells=[]
    for file in files:
        p=json.loads(file.read_text())['provenance']
        name=f"{p['condition']}_env{p['target_env']}_seed{p['seed']}"
        saved=output/'cells'/f'{name}.json'
        if saved.exists():
            cell=json.loads(saved.read_text())
            assert cell['verification']['checkpoint_sha256'] == p['checkpoint_sha256']
        else:
            print(f'start {name}',flush=True)
            cell=evaluate(file,output)
            saved.write_text(json.dumps(cell,indent=2)+'\n')
        cells.append(cell)
        summarize(output,cells)
        (output/'progress.json').write_text(json.dumps(dict(completed=len(cells),expected=len(files),
            status='completed' if len(cells)==len(files) else 'running', elapsed_seconds=time.time()-started),indent=2)+'\n')
        print(f'completed {name}, {len(cells)}/{len(files)}, elapsed={time.time()-started:.1f}s',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('results/sirm_representation_h2/full'))
    parser.add_argument('--output',type=Path,default=Path('results/h2_latent_diagnosis'))
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    main(args.root,args.output,args.limit)
