"""Read-only PACS baseline/H2 package from locked checkpoints and CPU caches.

No training, probe fitting, selection changes, or GPU inference. CSV accuracies
are fractions; summary accuracies/drops are percent/percentage points.
"""
import argparse
import ast
import csv
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

VIEWS = ('original', 'grayscale', 'blur', 'gray_blur')
METHODS = {'A': ('baseline', 'SIRM_OFF'), 'B': ('H2', 'SIRM_OFF'),
           'C': ('baseline', 'SIRM_ON'), 'D': ('H2', 'SIRM_ON')}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def separation(x, y):
    # Equal-class within cosine, equal-class-pair between cosine, no self pairs.
    z = F.normalize(x.float(), dim=1)
    within, centers = [], []
    for label in y.unique():
        part = z[y == label]
        centers.append(part.mean(0))
        n = len(part)
        if n > 1:
            within.append(float((part.sum(0).square().sum() - part.square().sum()) / (n * (n - 1))))
    between = [float(a @ b) for i, a in enumerate(centers) for b in centers[i + 1:]]
    w, b = statistics.mean(within), statistics.mean(between)
    return dict(within_class_cosine=w, between_class_cosine=b, cosine_margin=w-b,
                feature_variance=float(x.float().var(0, unbiased=False).mean()),
                mean_feature_norm=float(x.float().norm(dim=1).mean()))


def main(root, output):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    clean, view_rows, feature_rows, curves, source_domains, probes = [], [], [], [], [], []
    audit = []
    paired_ids = {}
    for file in sorted((root / 'runs').glob('*/env*/seed*/metrics.json')):
        result = json.loads(file.read_text())
        p = result['provenance']
        assert not p['target_used_for_model_or_hyperparameter_selection']
        assert not p['smoke']
        aug, method = METHODS[p['condition']]
        identity = dict(dataset=p['dataset'], method=method, augmentation=aug,
                        condition=p['condition'], target_env=p['target_env'],
                        target_domain=p['target_domain'], seed=p['seed'],
                        selected_step=p['selected_step'], backbone=p['hparams']['model'])
        bank = Path(p['bank'])
        saved_code = ast.parse((bank / 'source_snapshot/run.py').read_text())
        current_code = ast.parse(Path('experiments/sirm_h1_h6/run.py').read_text())
        saved_transform = next(n for n in saved_code.body if isinstance(n, ast.ClassDef) and n.name == 'ViewTransform')
        current_transform = next(n for n in current_code.body if isinstance(n, ast.ClassDef) and n.name == 'ViewTransform')
        assert ast.dump(saved_transform) == ast.dump(current_transform)
        effective = json.loads((bank / 'effective_config.json').read_text())
        assert effective['config']['blur_kernel'] == 5 and effective['config']['blur_sigma'] == 1
        checkpoint = bank / 'checkpoints/best.pkl'
        assert digest(checkpoint) == p['checkpoint_sha256'], checkpoint
        logs = [json.loads(line) for line in (bank / 'train_log.jsonl').read_text().splitlines()]
        best = max(logs, key=lambda row: row['source_mean'])
        assert best['step'] == p['selected_step']
        assert best['source_mean'] == p['selected_source_accuracy']
        assert [row['step'] for row in logs] == list(range(0, 5001, 500))
        clean.append(dict(**identity, target_clean_accuracy=result['views']['original']['native']['accuracy'],
                          source_validation_accuracy=p['selected_source_accuracy'],
                          checkpoint_sha256=p['checkpoint_sha256'], split_hash=p['split_hash'],
                          checkpoint_path=str(checkpoint.resolve()), metric_path=str(file.resolve())))
        for row in logs:
            curves.append(dict(**identity, step=row['step'], train_CE=row['loss_cls'],
                               train_accuracy=row['train_acc'], source_validation_accuracy=row['source_mean'],
                               target_accuracy_posthoc=(result['views']['original']['native']['accuracy']
                                                        if row['step'] == p['selected_step'] else None),
                               selected_checkpoint=row['step'] == p['selected_step'],
                               train_metric_scope='single augmented minibatch at logged step',
                               seconds=row['seconds']))
            for env, acc in row['source_validation'].items():
                source_domains.append(dict(**identity, step=row['step'], source_env=int(env),
                                           source_validation_accuracy=acc))
        for view in VIEWS:
            probes.append(dict(**identity, view=view, **result['linear_probe'][view],
                               regularization=p['probe']['regularization'], fit_split='source_in',
                               selection_split='source_out'))
        checks = dict(**identity, checkpoint_sha256_verified=True, source_selection_verified=True,
                      saved_augmentation_code_verified=True,
                      paired_views_verified=True, same_images_across_conditions=True,
                      max_target_accuracy_replay_error=0., max_source_feature_replay_error=0.,
                      selected_source_accuracy_replay_error=0.)
        for kind, split in [('target', 'all'), ('source', 'out')]:
            original = None
            for view in VIEWS:
                name = f'{kind}_{split}_{view}.pt'
                path = file.parent / 'cache' / name
                if not path.exists():
                    path = bank / 'cache' / name
                c = torch.load(path, map_location='cpu', weights_only=False)
                assert c['metadata']['checkpoint_sha256'] == p['checkpoint_sha256']
                assert c['metadata']['split_hash'] == p['split_hash']
                assert c['metadata']['kind'] == kind and c['metadata']['view'] == view
                if kind == 'source':
                    assert p['target_env'] not in c['physical_domains'].unique().tolist()
                if original is None:
                    original = c
                    key = (p['target_env'], p['seed'], kind)
                    signature = (c['image_ids'], c['labels'], c['physical_domains'])
                    if key in paired_ids:
                        ids, labels, domains = paired_ids[key]
                        assert ids == signature[0] and torch.equal(labels, signature[1])
                        assert torch.equal(domains, signature[2])
                    else:
                        paired_ids[key] = signature
                assert c['image_ids'] == original['image_ids']
                assert torch.equal(c['labels'], original['labels'])
                assert torch.equal(c['physical_domains'], original['physical_domains'])
                features = c.get('backbone_feature', c['projected_feature'])
                original_features = original.get('backbone_feature', original['projected_feature'])
                feature = dict(**identity, data_split=f'{kind}_{split}', view=view,
                               paired_cosine=float(F.cosine_similarity(original_features, features, dim=1).mean()),
                               samples=len(c['labels']), **separation(features, c['labels']))
                feature_rows.append(feature)
                if kind == 'source':
                    saved = json.loads((file.parent / 'source_diagnostics.json').read_text())
                    prev = next(x for x in saved['representation_stability']
                                if x['representation'] == 'backbone' and x['view'] == view)
                    err = max(abs(feature[k] - prev[k]) for k in separation(features, c['labels']))
                    checks['max_source_feature_replay_error'] = max(err, checks['max_source_feature_replay_error'])
                    assert err < 1e-6
                pi = c.get('router_native')
                if pi is None:
                    pi = c['router_topk']  # All selected checkpoints are past warmup.
                    assert p['selected_step'] >= p['hparams']['subset_irm_topk_warmup_steps']
                for mode, weights in [('native', pi), ('global_weights', torch.tensor(p['global_weights']).expand(len(pi), -1))]:
                    logits = (c['expert_logits'] * weights.unsqueeze(-1)).sum(1)
                    correct = logits.argmax(1).eq(c['labels']).float()
                    ce = F.cross_entropy(logits, c['labels'], reduction='none')
                    domains = c['physical_domains'].unique()
                    acc = float(correct.mean()) if kind == 'target' else statistics.mean(
                        float(correct[c['physical_domains'] == d].mean()) for d in domains)
                    nll = float(ce.mean()) if kind == 'target' else statistics.mean(
                        float(ce[c['physical_domains'] == d].mean()) for d in domains)
                    view_rows.append(dict(**identity, data_split=f'{kind}_{split}', inference_mode=mode,
                                          view=view, accuracy=acc, nll=nll, samples=len(correct),
                                          averaging='images' if kind == 'target' else 'equal source domains'))
                    if kind == 'target':
                        err = abs(acc - result['views'][view][mode]['accuracy'])
                        checks['max_target_accuracy_replay_error'] = max(err, checks['max_target_accuracy_replay_error'])
                        assert err < 1e-7
                    elif view == 'original' and mode == 'native':
                        err = abs(acc - p['selected_source_accuracy'])
                        checks['selected_source_accuracy_replay_error'] = err
                        assert err < 1e-7
                del c
            del original
        audit.append(checks)
        print(f"verified {p['condition']}/env{p['target_env']}/seed{p['seed']}", flush=True)
    assert len(clean) == 48 and len(curves) == 528
    assert len(view_rows) == 768 and len(feature_rows) == 384
    lookup = {(r['condition'], r['target_env'], r['seed']): r for r in clean}
    gains = []
    for first, second in [('A', 'B'), ('C', 'D')]:
        for target in range(4):
            for seed in range(3):
                a, b = lookup[first, target, seed], lookup[second, target, seed]
                assert a['split_hash'] == b['split_hash']
                gains.append(dict(dataset='PACS', method=a['method'], target_env=target,
                                  target_domain=a['target_domain'], seed=seed,
                                  baseline_target_accuracy=a['target_clean_accuracy'], H2_target_accuracy=b['target_clean_accuracy'],
                                  delta_target_pp=100*(b['target_clean_accuracy']-a['target_clean_accuracy']),
                                  baseline_source_accuracy=a['source_validation_accuracy'], H2_source_accuracy=b['source_validation_accuracy'],
                                  delta_source_pp=100*(b['source_validation_accuracy']-a['source_validation_accuracy'])))
    drops = []
    for row in view_rows:
        if row['view'] != 'original':
            continue
        matches = [x for x in view_rows if all(x[k] == row[k] for k in
                   ['condition', 'target_env', 'seed', 'data_split', 'inference_mode'])]
        by_view = {x['view']: x['accuracy'] for x in matches}
        drops.append({**{k: v for k, v in row.items() if k not in ['view', 'accuracy', 'nll']},
                      **{f'accuracy_{v}': by_view[v] for v in VIEWS},
                      **{f'drop_{v}_pp': 100*(by_view['original']-by_view[v]) for v in VIEWS[1:]}})
    for name, rows in [('clean_accuracy', clean), ('paired_H2_gains', gains), ('four_view_accuracy', view_rows),
                       ('color_texture_sensitivity', drops), ('training_curves', curves),
                       ('source_domain_curves', source_domains), ('backbone_diagnostics', feature_rows), ('frozen_probe', probes)]:
        save_csv(output / f'{name}.csv', rows)
    (output / 'verification.json').write_text(json.dumps(dict(runs=48, checks=audit, retraining=False,
        GPU_inference=False, new_checkpoint_selection=False, views=list(VIEWS), csv_accuracy_unit='fraction',
        csv_drop_unit='percentage points'), indent=2) + '\n')
    report(output, clean, gains, view_rows, drops, feature_rows, probes, curves)


def report(output, clean, gains, views, drops, features, probes, curves):
    def mean(rows, key):
        return statistics.mean(r[key] for r in rows)
    def group(rows, **where):
        return [r for r in rows if all(r[k] == v for k, v in where.items())]
    lines = ['# H2 augmentation: existing-log diagnostic package', '',
             '48 completed PACS runs: 4 targets × 3 seeds × baseline/H2 × SIRM OFF/ON. CPU cache replay only; no new training, fitting, checkpoint selection or hyperparameter changes. All selected checkpoint hashes and paired image identities verified. CSV accuracies are fractions; tables use percent and accuracy deltas/drops use percentage points.', '',
             'These runs use the legacy six-expert SIRM architecture. OFF retains expert CE and router distillation; it is not a plain single-head ERM baseline. Native routing, source-fitted global weights and frozen linear probes are three readings of the same bank, not three independently trained algorithms. They are distinct from the new frozen-global/residual SIRM-v2 architecture.', '',
             '## Augmentation confound', '',
             'Saved training code applies crop/flip/color jitter/RandomGrayscale(p=0.1), then H2 uniformly chooses identity/grayscale/Gaussian blur before normalization. Kernel=5, sigma=1. Thus grayscale-operation probability is 0.1 + 0.9/3 = **0.40**, blur-operation probability is **1/3**, and combined gray+blur operation probability is **0.1/3 ≈ 3.33%**. These are sampling probabilities, not empirical pixel-level color proportions (some images already lack color). A dedicated gray+blur branch is evaluation-only. Existing runs do not isolate increased gray exposure from blur.', '',
             '## 1. Clean accuracy by method, target and seed', '',
             '[48 checkpoint scores/source validation](clean_accuracy.csv), [24 paired H2 gains](paired_H2_gains.csv). Checkpoints use equal-source-domain out accuracy, earliest best tie; target scores never select them.', '',
             '| Target | OFF baseline | OFF H2 | Δ OFF | ON baseline | ON H2 | Δ ON |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for target in range(4):
        rows = group(clean, target_env=target)
        vals = [100*mean(group(rows, condition=c), 'target_clean_accuracy') for c in METHODS]
        lines.append(f"| {rows[0]['target_domain']} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[1]-vals[0]:+.3f} | {vals[2]:.3f} | {vals[3]:.3f} | {vals[3]-vals[2]:+.3f} |")
    for method in ['SIRM_OFF', 'SIRM_ON']:
        rows = group(gains, method=method)
        lines += ['', f"{method}: mean clean gain **{mean(rows, 'delta_target_pp'):+.3f} pp**, positive in **{sum(r['delta_target_pp'] > 0 for r in rows)}/12** target×seed cells; mean selected source-validation gain {mean(rows, 'delta_source_pp'):+.3f} pp. Three seeds only; overlapping sources limit cell independence."]
    lines += ['', '## 2. Same checkpoint, four views and color/texture sensitivity', '',
              '[All views, target and source-out, native/global](four_view_accuracy.csv); [per-run drops](color_texture_sensitivity.csv). Source-out scores are equal-domain means. Negative drop means the transformed view scores higher. No clipping.', '',
              '| Condition | Split | Original | Gray | Blur | Gray+blur | Gray drop | Blur drop |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in METHODS:
        for split in ['target_all', 'source_out']:
            rows = group(drops, condition=c, data_split=split, inference_mode='native')
            vals = [100*mean(rows, f'accuracy_{v}') for v in VIEWS]
            lines.append(f"| {c}: {METHODS[c][0]}/{METHODS[c][1]} | {split} | " + ' | '.join(f'{v:.3f}' for v in vals) + f" | {mean(rows, 'drop_grayscale_pp'):.3f} | {mean(rows, 'drop_blur_pp'):.3f} |")
    lines += ['', '## 3. Training curves', '',
              '[CE/train accuracy/source-validation curves](training_curves.csv), [per-source-domain curves](source_domain_curves.csv), [curve figure](training_curves.png), [zoom after step 1000](training_curves_late.png). Original logs have 11 checkpoints per run at 0, 500, …, 5000; CE and train accuracy are the single augmented minibatch at that step, not running epoch means. Validation follows the update. The figures average 12 target×seed cells per condition; the late figure explicitly crops the vertical axes.', '',
              '**Target curves are unavailable**: training deliberately did not read target images, and only best/last/resume states were retained. Stored selected-checkpoint target scores appear only at their selected step; missing target steps are blank, never interpolated or reconstructed from the final checkpoint. No new 250-step rows are invented.', '',
              '| Condition | Final train CE | Final train acc | Final source-val | Mean CE, steps 1000–5000 | Mean train acc, steps 1000–5000 |',
              '|---|---:|---:|---:|---:|---:|']
    for c in METHODS:
        final = group(curves, condition=c, step=5000)
        late = [r for r in curves if r['condition'] == c and r['step'] >= 1000]
        lines.append(f"| {c} | {mean(final, 'train_CE'):.5f} | {100*mean(final, 'train_accuracy'):.3f} | {100*mean(final, 'source_validation_accuracy'):.3f} | {mean(late, 'train_CE'):.5f} | {100*mean(late, 'train_accuracy'):.3f} |")
    lines += ['', 'These sparse minibatch snapshots can suggest training difficulty, but cannot reliably identify generic regularization versus optimization from the train-accuracy trend alone.', '',
              '## 4. Backbone invariance and class separation', '',
              '[Paired backbone diagnostics](backbone_diagnostics.csv), [frozen linear probes](frozen_probe.csv). Features are the true backbone CLS output (Identity input projection). Within cosine averages off-diagonal same-class pairs equally across classes; between cosine averages different-class pairs equally across class pairs. Margin = within − between. Original↔view cosine uses the same image. A higher cosine alone is not sufficient evidence against collapse.', '',
              '| Condition | Split | Orig↔gray | Orig↔blur | Orig within | Orig between | Orig margin | Gray margin | Blur margin |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in METHODS:
        for split in ['source_out', 'target_all']:
            by = {v: group(features, condition=c, data_split=split, view=v) for v in VIEWS}
            vals = [mean(by['grayscale'], 'paired_cosine'), mean(by['blur'], 'paired_cosine'),
                    mean(by['original'], 'within_class_cosine'), mean(by['original'], 'between_class_cosine'),
                    mean(by['original'], 'cosine_margin'), mean(by['grayscale'], 'cosine_margin'), mean(by['blur'], 'cosine_margin')]
            lines.append(f'| {c} | {split} | ' + ' | '.join(f'{v:.5f}' for v in vals) + ' |')
    lines += ['', '| Condition | Probe original | Probe gray | Probe blur | Probe gray+blur |', '|---|---:|---:|---:|---:|']
    for c in METHODS:
        vals = [100*mean(group(probes, condition=c, view=v), 'accuracy') for v in VIEWS]
        lines.append(f'| {c} | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
    lines += ['', 'Original-view cosine margin decreases modestly with H2: source-out A→B **0.64567→0.63592**, C→D **0.64336→0.63731**; target A→B **0.52696→0.52543**, C→D **0.51936→0.51377**. Thus the strict claim “invariance improves with no separation decrease” is not supported by this metric. Frozen original-view probe accuracy nevertheless increases **+0.407 pp OFF / +0.697 pp ON**; the joint evidence does not indicate wholesale feature collapse.']
    lines += ['', 'Probes were already fitted on source-in with shared feature standardization, source-out NLL selecting L2 from 0.001/0.01/0.1. This package reuses their locked results and does not refit them. Pooled class statistics are descriptive and can mix source-domain structure with class structure.', '',
              '## 5. ResNet50 versus DeiT-S', '',
              '**N/A: no matched ResNet50 baseline/H2 pair is present in the local experiment outputs.** All 48 factorial banks use DeiT-S/16 with the same distilled pretrained checkpoint. A ResNet score from another architecture/protocol without matched H2 cannot provide an architecture-specific H2 delta. No new ablation or backbone run was launched, as requested.', '',
              '## What these logs distinguish', '',
              'H2 improves original accuracy as well as transformed-view accuracy, so its effect is broader than robustness restricted to corruptions. Gains concentrate on Sketch/Cartoon while Photo is nearly unchanged, consistent with reduced sensitivity to style cues. This is an interpretation of observed behavior, not a causal identification of texture/color shortcuts. Gray/blur drops can be negative on some targets, and gains need not be positive in every seed.', '',
              'Paired invariance and frozen probe quality provide additional evidence about representation; inspect the margins alongside cosine rather than assuming all separation metrics improve. The package cannot identify gray versus blur as the driver, establish a DeiT-specific effect without matched ResNet runs, or independently establish gains across unrelated algorithms.', '',
              '[Replay/provenance verification](verification.json). Training source code/configuration and all original results remain unchanged.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, constrained_layout=True)
    for col, pair in enumerate([('A', 'B'), ('C', 'D')]):
        for c in pair:
            for row, (key, label, scale) in enumerate([
                    ('train_CE', 'Train CE (sampled minibatch)', 1),
                    ('train_accuracy', 'Train accuracy (%)', 100),
                    ('source_validation_accuracy', 'Source validation (%)', 100)]):
                steps = sorted(set(r['step'] for r in curves))
                vals = [scale*mean(group(curves, condition=c, step=s), key) for s in steps]
                axes[row, col].plot(steps, vals, marker='o', markersize=3, label=METHODS[c][0])
                axes[row, col].set_ylabel(label)
                axes[row, col].grid(alpha=.2)
        axes[0, col].set_title(f'Legacy expert bank: {METHODS[pair[0]][1]}')
        axes[0, col].legend()
        axes[2, col].set_xlabel('Optimizer step')
    fig.suptitle('PACS: 12 target × seed cells per condition; 500-step snapshots')
    fig.savefig(output / 'training_curves.png', dpi=170)
    fig.savefig(output / 'training_curves.svg')
    for row in range(3):
        for col in range(2):
            axes[row, col].set_xlim(950, 5050)
            key, scale = [('train_CE', 1), ('train_accuracy', 100), ('source_validation_accuracy', 100)][row]
            pair = [('A', 'B'), ('C', 'D')][col]
            vals = [scale*mean(group(curves, condition=c, step=s), key)
                    for c in pair for s in range(1000, 5001, 500)]
            span = max(max(vals)-min(vals), .001 if row == 0 else .05)
            axes[row, col].set_ylim(min(vals)-.15*span, max(vals)+.15*span)
    fig.suptitle('PACS: zoom on steps 1000–5000 (vertical axes are cropped)')
    fig.savefig(output / 'training_curves_late.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('results/sirm_representation_h2/full'))
    parser.add_argument('--output', type=Path, default=Path('results/h2_augmentation_diagnosis'))
    args = parser.parse_args()
    main(args.root, args.output)
