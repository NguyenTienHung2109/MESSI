"""One-seed frozen-PACS mechanism audit. No target-driven model selection."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from domainbed.predictive_compatibility import (
    CandidateMap, make_cell, project, solve_common, audit_head)
from domainbed.compatibility_diagnostics import log_event, prediction_metrics, bank_behavior


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def partition(n, seed):
    indices = np.random.default_rng(seed).permutation(n)
    chunks = np.split(indices, [int(.5*n), int(.7*n), int(.9*n)])
    if min(map(len, chunks)) < 1:
        raise ValueError('Too few samples for four disjoint stages')
    return dict(zip(['train', 'calibration', 'audit', 'selection'], chunks))


def candidate_supports(sources):
    # Baseline plus hypotheses. A duplicate global candidate is allowed.
    proper = list(itertools.combinations(sources, 2)) if len(sources) > 2 else []
    return [tuple(sources)] + proper + [tuple(sources)]


def load_features(args):
    from domainbed.datasets import PACS
    from domainbed.deit_transformer import DeiTFeaturizer
    from torch.utils.data import DataLoader
    dataset = PACS(args.data_dir, [], {'data_augmentation': False})
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    backbone, bank, manifest = None, {}, {}
    for domain, ds in enumerate(dataset):
        # Cache identity includes transform, ordered dataset metadata and code.
        samples = [(str(Path(p).relative_to(ds.root)), y, Path(p).stat().st_size,
                    Path(p).stat().st_mtime_ns) for p, y in ds.samples]
        source_files = ['domainbed/deit_transformer.py', 'domainbed/vision_transformer.py',
                        'domainbed/vit_helpers.py', 'domainbed/datasets.py']
        identity = dict(model='deit_small_patch16_224', pretrained=True,
                        transform='resize224/ImageNet-normalization/eval/float32',
                        samples=samples, code={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                              for p in source_files})
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = cache / f'env{domain}_{fingerprint[:16]}.pt'
        if path.exists():
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload['fingerprint'] != fingerprint:
                raise ValueError('Feature-cache fingerprint mismatch')
        else:
            if backbone is None:
                backbone = DeiTFeaturizer(pretrained=True).to(args.device).eval()
                backbone.requires_grad_(False)
            xs, ys = [], []
            with torch.inference_mode():
                for x, y in DataLoader(ds, batch_size=64, shuffle=False, num_workers=args.workers):
                    xs.append(backbone(x.to(args.device)).cpu())
                    ys.append(y)
            payload = dict(x=torch.cat(xs), y=torch.cat(ys), fingerprint=fingerprint)
            torch.save(payload, path)
        bank[domain] = (payload['x'], payload['y'])
        manifest[str(domain)] = dict(name=Path(ds.root).name, n=len(ds), fingerprint=fingerprint,
                                    classes=ds.classes)
        print(f'cached domain {domain}: {len(ds)} fixed features', flush=True)
    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return bank, manifest, dataset.num_classes


def train_map(bank, splits, support, args, classes, seed, log_path=None, expert=None):
    torch.manual_seed(seed)
    model = CandidateMap(bank[support[0]][0].shape[1], args.dimension, args.hidden)
    head = nn.Parameter(torch.zeros(classes, args.dimension))
    optimizer = torch.optim.Adam(list(model.parameters()) + [head], lr=args.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    trace = []
    for step in range(args.map_steps):
        losses = []
        for domain in support:
            indices = torch.as_tensor(splits[domain]['train'])
            selected = indices[torch.randint(len(indices), (args.batch_size,), generator=generator)]
            x, y = bank[domain]
            losses.append(F.cross_entropy(model(x[selected]) @ head.T, y[selected]))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        grad_norm = math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None))
        optimizer.step()
        with torch.no_grad():
            head.copy_(project(head, args.radius))
        if step % 100 == 0 or step == args.map_steps-1:
            event = dict(stage='map', expert=expert, step=step, support=list(support),
                         mean_domain_ce=float(loss.detach()), map_gradient_norm=grad_norm,
                         head_row_norm_mean=float(head.detach().norm(dim=1).mean()),
                         domain_ce={str(d): float(v.detach()) for d, v in zip(support, losses)})
            trace.append(event)
            log_event(log_path, event)
    model.eval().requires_grad_(False)
    return model, trace


def train_router(x, logits, y, args, seed, log_path=None, name='all_bank'):
    torch.manual_seed(seed)
    router = nn.Linear(x.shape[1], logits.shape[1])
    nn.init.zeros_(router.weight)
    nn.init.zeros_(router.bias)
    optimizer = torch.optim.Adam(router.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    for step in range(args.router_steps):
        ix = torch.randint(len(y), (args.batch_size,), generator=generator)
        mixture = (router(x[ix]).softmax(1)[:, :, None] * logits[ix]).sum(1)
        loss = F.cross_entropy(mixture, y[ix])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == args.router_steps-1:
            with torch.no_grad():
                weights = router(x[ix]).softmax(1)
                log_event(log_path, dict(stage='router', name=name, step=step,
                    mixture_ce=float(loss.detach()), mass=weights.mean(0).tolist(),
                    entropy=float(-(weights*weights.clamp_min(1e-30).log()).sum(1).mean())))
    return router.eval().requires_grad_(False)


def metrics(logits, y):
    return prediction_metrics(logits, y)


def fit_fold(bank, sources, classes, args, out, family_events):
    """Receives adaptation domains only: held domains cannot enter any stage."""
    if set(bank) != set(sources):
        raise ValueError('Fold bank must contain exactly the adaptation domains')
    out.mkdir(parents=True, exist_ok=True)
    splits = {d: partition(len(bank[d][1]), args.seed + 1009*d) for d in sources}
    write_json(out / 'split_manifest.json',
               {str(d): {k: v.tolist() for k, v in s.items()} for d, s in splits.items()})
    alpha_event = args.alpha / family_events
    supports = candidate_supports(sources)
    maps, heads, reports = [], [], []
    for expert, support in enumerate(supports):
        print(f'{out.name}: map {expert}, support={support}', flush=True)
        model, trace = train_map(bank, splits, support, args, classes, args.seed + expert,
                                 out/'training.jsonl', expert)
        with torch.no_grad():
            z = {d: model(bank[d][0]) for d in support}
        def cells(stage):
            return [make_cell(z[d][splits[d][stage]], bank[d][1][splits[d][stage]]) for d in support]
        w, solver = solve_common(cells('calibration'), classes, args.radius,
                                 args.solver_steps, args.solver_tolerance)
        counts = [torch.bincount(bank[d][1][splits[d]['audit']], minlength=classes).tolist()
                  for d in support]
        audit = audit_head(w, cells('audit'), counts, args.radius, args.tau, args.delta,
                           alpha_event, args.solver_steps, args.solver_tolerance)
        report = dict(expert=expert, support=list(support), baseline=expert == 0,
                      training_trace=trace, calibration=solver, audit=audit)
        reports.append(report)
        maps.append(model)
        heads.append(w.float())
        write_json(out / 'compatibility.json', reports)
        print(f'  {audit["status"]}; empirical J upper={solver["upper"]:.6f}; '
              f'solve gap={solver["gap"]:.6f}', flush=True)

    accepted = [0] + [i for i, r in enumerate(reports) if i and r['audit']['accepted']]
    with torch.no_grad():
        def bank_logits(x):
            return torch.stack([m(x) @ w.T for m, w in zip(maps, heads)], dim=1)
        # Equal-domain sampling: deterministically cycle shorter domains.
        max_n = max(len(splits[d]['train']) for d in sources)
        train_ids = {d: np.resize(splits[d]['train'], max_n) for d in sources}
        train_x = torch.cat([bank[d][0][train_ids[d]] for d in sources])
        train_y = torch.cat([bank[d][1][train_ids[d]] for d in sources])
        train_logits = bank_logits(train_x)
    all_router = train_router(train_x, train_logits, train_y, args, args.seed, out/'training.jsonl')
    certified_router = (train_router(train_x, train_logits[:, accepted], train_y, args, args.seed,
                                     out/'training.jsonl', 'certified')
                        if len(accepted) > 1 else None)
    selection, lower_gains = {}, []
    for d in sources:
        ix = splits[d]['selection']
        x, y = bank[d][0][ix], bank[d][1][ix]
        with torch.no_grad():
            logits = bank_logits(x)
            baseline = logits[:, 0]
            composition = baseline if certified_router is None else (
                certified_router(x).softmax(1)[:, :, None]*logits[:, accepted]).sum(1)
            gains = F.cross_entropy(baseline, y, reduction='none') - F.cross_entropy(composition, y, reduction='none')
            # Paired gain in [-L,L], fixed models independent of selection split.
            penalty = (math.log(classes)+2*args.radius)*math.sqrt(2*math.log(1/alpha_event)/len(y))
            lower = float(gains.mean())-penalty
            lower_gains.append(lower)
            selection[str(d)] = dict(global_metrics=metrics(baseline, y),
                                     certified_metrics=metrics(composition, y),
                                     mean_ce_gain=float(gains.mean()), ce_gain_lower=lower)
    use_composition = len(accepted) > 1 and np.mean(lower_gains) > 0
    summary = dict(adaptation_domains=sources, candidate_supports=supports,
                   accepted_experts=accepted[1:], selected='certified' if use_composition else 'global',
                   fallback_reason=None if use_composition else (
                       'no_certified_candidate' if len(accepted) == 1 else 'insufficient_contribution_evidence'),
                   selection=selection, mean_ce_gain_lower=float(np.mean(lower_gains)),
                   alpha_per_event=alpha_event, family_events=family_events,
                   baseline_is_always_available_not_assumed_certified=True)
    write_json(out / 'selection.json', summary)
    torch.save(dict(maps=[m.state_dict() for m in maps], heads=heads,
                    all_router=all_router.state_dict(),
                    certified_router=None if certified_router is None else certified_router.state_dict(),
                    accepted=accepted, selected=summary['selected'], config=vars(args)), out / 'model.pt')

    @torch.no_grad()
    def evaluate(x, y, artifact_path=None):
        logits = bank_logits(x)
        features = torch.stack([m(x) for m in maps], 1)
        weights = all_router(x).softmax(1)
        if artifact_path is not None:
            np.savez_compressed(artifact_path, labels=y.numpy(), expert_logits=logits.numpy(),
                                router_weights=weights.numpy(), features=features.numpy())
        certified = logits[:, 0] if certified_router is None else (
            certified_router(x).softmax(1)[:, :, None]*logits[:, accepted]).sum(1)
        controls = dict(global_predictor=logits[:, 0], uniform=logits.mean(1),
                        learned=(weights[:, :, None]*logits).sum(1), certified=certified,
                        selected=certified if use_composition else logits[:, 0])
        predictions = logits.argmax(2)
        losses = torch.stack([F.cross_entropy(logits[:, i], y, reduction='none')
                              for i in range(len(maps))], dim=1)
        return dict(behavior=bank_behavior(features, logits, weights, y),
                    controls={k: metrics(v, y) for k, v in controls.items()},
                    experts=[metrics(logits[:, i], y) for i in range(len(maps))],
                    router_mass=weights.mean(0).tolist(),
                    disagreement=[[float((predictions[:, i] != predictions[:, j]).float().mean())
                                   for j in range(len(maps))] for i in range(len(maps))],
                    oracle_single_expert_accuracy=float((predictions == y[:, None]).any(1).float().mean()),
                    leave_one_expert_out_uniform_ce=[float(F.cross_entropy(
                        logits[:, [j for j in range(len(maps)) if j != i]].mean(1), y))
                        for i in range(len(maps))],
                    expert_risk_correlation=torch.nan_to_num(torch.corrcoef(losses.T)).tolist())
    # Descriptive stage comparisons only after the deployment decision is frozen.
    source_diagnostics = {}
    for d in sources:
        source_diagnostics[str(d)] = {}
        for stage, ix in splits[d].items():
            source_diagnostics[str(d)][stage] = evaluate(bank[d][0][ix], bank[d][1][ix])
    write_json(out/'source_behavior.json', source_diagnostics)
    return summary, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='domainbed/data')
    parser.add_argument('--cache-dir', default='predictive_support_outputs/compatibility_cache')
    parser.add_argument('--output-dir', default='predictive_support_outputs/pacs_compatibility_seed0')
    parser.add_argument('--target', type=int, default=3, choices=range(4))
    parser.add_argument('--seed', type=int, choices=[0], default=0)
    parser.add_argument('--loso', action='store_true')
    parser.add_argument('--family-runs', type=int, default=1,
                        help='Prespecified number of target runs sharing the alpha budget')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dimension', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--radius', type=float, default=1.)
    parser.add_argument('--tau', type=float, default=.12)
    parser.add_argument('--delta', type=float, default=.05)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--map-steps', type=int, default=500)
    parser.add_argument('--router-steps', type=int, default=300)
    parser.add_argument('--solver-steps', type=int, default=1000)
    parser.add_argument('--solver-tolerance', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=.001)
    args = parser.parse_args()
    if min(args.map_steps, args.router_steps, args.solver_steps, args.dimension, args.hidden) < 1:
        parser.error('Budgets and dimensions must be positive')
    if not (0 < args.alpha < 1 and args.radius > 0 and args.tau >= 0 and args.delta > 0):
        parser.error('Invalid bounds/thresholds')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    out = Path(args.output_dir)
    if (out / 'config.json').exists():
        raise FileExistsError('Use a fresh output directory; audit data must not drive iterative tuning')
    write_json(out / 'config.json', vars(args))
    code_files = ['domainbed/predictive_compatibility.py', 'domainbed/compatibility_diagnostics.py', __file__]
    for name in code_files:
        destination = out / 'source_snapshot' / Path(name).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(name).read_bytes())
    bank, manifest, classes = load_features(args)
    write_json(out / 'features.json', manifest)
    sources = [d for d in bank if d != args.target]
    folds = [sources] + ([[d for d in sources if d != held] for held in sources] if args.loso else [])
    # Uniform-risk and entropy events per expert/domain, plus paired-gain events.
    if args.family_runs < 1:
        parser.error('family-runs must be positive')
    events = args.family_runs * sum(2*sum(map(len, candidate_supports(f))) + len(f) for f in folds)
    start = time.time()
    summary, evaluate = fit_fold({d: bank[d] for d in sources}, sources, classes, args, out/'main', events)
    transfer = {}
    if args.loso:
        for held in sources:
            adaptation = [d for d in sources if d != held]
            fold_summary, fold_evaluate = fit_fold({d: bank[d] for d in adaptation}, adaptation,
                                                   classes, args, out/f'loso_env{held}', events)
            transfer[str(held)] = dict(selection=fold_summary, evaluation=fold_evaluate(
                *bank[held], artifact_path=out/f'loso_env{held}'/'held_predictions.npz'))
            write_json(out/'transfer.json', transfer)
    # Target is evaluated only after every source-only fit and choice is frozen.
    target = evaluate(*bank[args.target], artifact_path=out/'target_predictions.npz')
    result = dict(selection=summary, target_domain=args.target, target_evaluation=target,
                  transfer=transfer, elapsed_fit_seconds=time.time()-start,
                  interpretation='Fixed-map, bounded-linear-head source audit; no unconditional DG guarantee.')
    write_json(out/'results.json', result)
    lines = ['# Frozen PACS predictive compatibility audit', '',
             f'Seed: 0. Target domain: {manifest[str(args.target)]["name"]}.',
             f'Selected predictor: **{summary["selected"]}**. Reason: {summary["fallback_reason"]}.',
             f'Accepted extra experts: {summary["accepted_experts"]}.', '',
             '| Matched control | Target accuracy | Target CE |', '|---|---:|---:|']
    for name, m in target['controls'].items():
        lines.append(f'| {name} | {100*m["accuracy"]:.2f}% | {m["cross_entropy"]:.4f} |')
    lines += ['', 'Target scores are diagnostics only, never used to select supports or thresholds.',
              'Failure to certify is insufficient evidence in this candidate/head class, not proof that subsets do not exist.',
              'Independent audit assumptions and conservative uniform bounds are documented in docs/predictive_compatibility.md.',
              'These frozen-feature/split results are not a matched comparison against earlier fine-tuned V1/V2 runs.', '',
              'See main/compatibility.json for per-domain risk, optimum bounds, numerical gaps,',
              'entropy lower bounds and regret confidence bounds; transfer.json contains strict domain-exclusion diagnostics.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(selected=summary['selected'], accepted=summary['accepted_experts'],
                          target_accuracy=target['controls']['selected']['accuracy'])), flush=True)


if __name__ == '__main__':
    main()
