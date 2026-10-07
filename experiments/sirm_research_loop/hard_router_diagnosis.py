"""Source-only saved-bank router, utility and paired-view diagnosis.

No fitting, optimizer calls, parameter updates, target access or selection.
Exact historical H2 image schedules are replayed at saved states, not historical
intermediate weights. Original source-in cache labels are joined from manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from domainbed.deit_transformer import DeiTFeaturizer
from domainbed.scripts.train_subset_irm_pacs import make_schedule
from experiments.sirm_discover.math import center_logits, residual_utilities, router_teacher
from experiments.sirm_discover_gate.data import Adapter
from experiments.sirm_h1_h6.run import ScheduledImages
from experiments.terra_10pct_factor_diagnosis.report import data_for
from .centered_model import CenteredGateSIRM
from .shrinkage import ROOT, digest, write

STEPS = [200, 215, 250, 300, 350, 395]


def outputs(model, z, y, mean_mode='saved'):
    g = model.global_head(z)
    raw = center_logits(torch.stack([e(z) for e in model.residuals], 1))
    delta = raw - (raw.mean(0) if mean_mode == 'batch' else model.expert_source_mean)[None]
    pi = model.router(z).softmax(1)
    full, _, marginal, single = residual_utilities(g, delta, pi, y)
    teacher = router_teacher(single, model.config['router_temperature'], model.config['utility_floor'])
    oracle = g + (teacher[:, :6, None] * delta).sum(1)
    return dict(g=g, delta=delta, pi=pi, full=full, teacher=teacher,
                oracle=oracle, marginal=marginal, single=single,
                winner=F.cross_entropy((g[:, None] + delta).reshape(-1, g.shape[1]),
                                      y[:, None].expand(-1, 6).reshape(-1), reduction='none')
                .reshape(-1, 6).argmin(1))


def metrics(f, y, domains):
    global_correct = f['g'].argmax(1).eq(y)
    results = {}
    masks = {'all': torch.ones(len(y), dtype=torch.bool),
             'global_correct': global_correct, 'global_wrong': ~global_correct}
    for d in sorted(domains.unique().tolist()):
        masks[f'domain/{d}'] = domains.eq(d)
        for cls in sorted(y.unique().tolist()):
            mask = domains.eq(d) & y.eq(cls)
            if mask.any():
                masks[f'domain/{d}/class/{cls}'] = mask
    for name, mask in masks.items():
        if not mask.any():
            continue
        row = dict(n=int(mask.sum()),
                   actual_NULL=float(f['pi'][mask, -1].mean()),
                   teacher_NULL=float(f['teacher'][mask, -1].mean()),
                   teacher_KL=float(F.kl_div(f['pi'][mask].clamp_min(1e-30).log(),
                                            f['teacher'][mask], reduction='batchmean')),
                   actual_negative_context_utility_mass=float(
                       (f['pi'][mask, :6] * (f['marginal'][mask] < 0)).sum(1).mean()),
                   mean_U=f['marginal'][mask].mean(0).tolist(),
                   winner_load=F.one_hot(f['winner'][mask], 6).double().mean(0).tolist())
        for label, key in [('global', 'g'), ('actual', 'full'), ('label_teacher', 'oracle')]:
            logits = f[key]
            correct = logits.argmax(1).eq(y)
            row[label] = dict(accuracy=float(correct[mask].double().mean()),
                              CE=float(F.cross_entropy(logits[mask], y[mask])),
                              repairs=int((mask & ~global_correct & correct).sum()),
                              harms=int((mask & global_correct & ~correct).sum()))
        results[name] = row
    # Equal-domain aggregate complements pooled and conditional groups.
    domain_rows = [results[f'domain/{d}'] for d in domains.unique().tolist()]
    results['equal_domain'] = {
        name: {key: float(np.mean([r[name][key] for r in domain_rows]))
               for key in ['accuracy', 'CE']}
        for name in ['global', 'actual', 'label_teacher']}
    return results


def gradient_metrics(model, z, y, mode):
    f = outputs(model, z, y, mode)
    parameters = list(model.router.parameters())
    cls = F.cross_entropy(f['full'], y)
    route = model.config['lambda_router'] * F.kl_div(
        F.log_softmax(model.router(z.detach()), 1), f['teacher'], reduction='batchmean')
    cg = torch.autograd.grad(cls, parameters, retain_graph=True)
    rg = torch.autograd.grad(route, parameters)
    a = torch.cat([g.detach().double().flatten() for g in cg])
    b = torch.cat([g.detach().double().flatten() for g in rg])
    an, bn = float(a.norm()), float(b.norm())
    assert all(p.grad is None for p in model.parameters())
    return dict(classification_router_grad_norm=an,
                weighted_distillation_router_grad_norm=bn,
                weighted_distillation_to_classification_ratio=bn / an if an else None,
                classification_distillation_cosine=float(F.cosine_similarity(a, b, dim=0)) if an and bn else None,
                norm_scope='router parameters only; autograd.grad, no .grad assignment or optimizer update')


def main(dest):
    torch.set_num_threads(4)
    out = dest / 'source_router_diagnosis'
    out.mkdir(exist_ok=True)
    bank = dest / 'experiment/runs/L38/OFF'
    manifest_path = dest / 'experiment/subset_manifest.json'
    cache_path = dest / 'source_feature_cache.pt'
    manifest = json.loads(manifest_path.read_text())
    sources = [0, 2, 3]
    paths = {'selected': bank / 'source_best.pkl', 'last': bank / 'last.pkl'}
    locks = {k: digest(p) for k, p in paths.items()}
    assert locks['selected'] == json.loads((bank / 'posthoc/complete.json').read_text())['checkpoint_sha256']
    plan = dict(checkpoint_hashes=locks, manifest_sha256=digest(manifest_path),
                source_cache_sha256=digest(cache_path), code_sha256=digest(Path(__file__)),
                source_envs=sources, scheduled_steps=STEPS,
                scope='Source-in/source-out original and exact paired H2 scheduled source batches at selected/final saved OFF states. '
                      'No target, fitting, updates, tuning or checkpoint selection. Label teacher is diagnostic only. '
                      'Train batch versus saved mean comparisons are saved-state replays, not historical intermediate states.')
    write(out / 'plan.json', plan)
    models = {}
    for name, path in paths.items():
        payload = torch.load(path, map_location='cpu', weights_only=False)
        c = payload['config']
        backbone = DeiTFeaturizer(pretrained=False, model_name=c['backbone'])
        model = CenteredGateSIRM(backbone, backbone.n_outputs, 10, 3, c)
        model.load_state_dict({k.removeprefix('model.'): v for k, v in payload['model_dict'].items()})
        model.configure_stage(); model.eval()
        assert all(not p.requires_grad for mod in [model.featurizer, model.global_head] for p in mod.parameters())
        models[name] = model
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    lookup = {item['id']: int(item['class']) for d in sources
              for item in manifest['sources'][str(d)]['images']}
    # The manifest schema uses label rather than class in some historic files.
    y_in = torch.tensor([lookup[i] for i in cache['ids']])
    assert set(cache['domains'].tolist()) == set(sources)
    source_features = {'source_in': (cache['features'], y_in, cache['domains'], cache['ids'])}
    phi = models['selected'].featurizer.cuda().eval()
    zs, ys, ds, ids = [], [], [], []
    with torch.no_grad():
        for d, dataset in sorted(data_for(manifest, sources, c, 'out').items()):
            ids.extend(manifest['sources'][str(d)]['images'][i]['id'] for i in dataset.indices)
            for x, y in DataLoader(dataset, batch_size=64, num_workers=0):
                zs.append(phi(x.cuda()).cpu()); ys.append(y); ds.append(torch.full_like(y, d))
    source_features['source_out'] = (torch.cat(zs), torch.cat(ys), torch.cat(ds), ids)
    assert not set(cache['ids']) & set(ids)
    source_results = {}
    with torch.no_grad():
        for state, model in models.items():
            source_results[state] = {}
            for split, (z, y, domains, image_ids) in source_features.items():
                f = outputs(model, z, y)
                source_results[state][split] = metrics(f, y, domains)
                if state == 'selected':
                    np.savez_compressed(out / (split + '_original.npz'), z=z.numpy(), y=y.numpy(),
                                        domains=domains.numpy(), ids=np.array(image_ids),
                                        **{k: v.numpy() for k, v in f.items()})
    with np.load(dest / 'eda/source_out_replay.npz') as saved:
        replay = outputs(models['selected'], source_features['source_out'][0], source_features['source_out'][1])
        assert np.array_equal(saved['ids'], np.array(ids)) and np.array_equal(saved['y'], source_features['source_out'][1].numpy())
        assert np.allclose(replay['g'].detach().numpy(), saved['OFF_g'], atol=1e-4)
        assert np.array_equal(replay['full'].detach().argmax(1).numpy(), saved['B1'].argmax(1))
    adapter = Adapter('TerraIncognita'); adapter.classes = manifest['class_names']
    source_manifest = dict(manifest, sources={str(d): manifest['sources'][str(d)] for d in sources})
    original = adapter.source_data(ROOT / 'domainbed/data', source_manifest, c, 'in')
    augmented = adapter.source_data(ROOT / 'domainbed/data', source_manifest, c, 'in', train=True)
    schedules = {d: make_schedule(list(range(len(dataset))), 400, 32, 0, d)
                 for d, dataset in augmented.items()}
    meta = json.loads((bank / 'source_schedules.json').read_text())
    for d in sources:
        assert list(augmented[d].indices) == meta[str(d)]['split_keys']
        assert hashlib.sha256(schedules[d].tobytes()).hexdigest() == meta[str(d)]['schedule_sha256']
    records = []
    for step in STEPS:
        paired = {'original': [], 'H2': []}; labels = []; domains = []; image_ids = []
        for d in sources:
            scheduled = ScheduledImages(augmented[d], 0, d)
            for position, index in enumerate(schedules[d][step]):
                image, label = scheduled[(int(index), step, position)]
                clean, clean_label = original[d][int(index)]
                assert label == clean_label
                paired['H2'].append(image); paired['original'].append(clean)
                labels.append(label); domains.append(d)
                image_ids.append(manifest['sources'][str(d)]['images'][original[d].indices[int(index)]]['id'])
        y, domain = torch.tensor(labels), torch.tensor(domains)
        with torch.no_grad():
            features = {view: phi(torch.stack(images).cuda()).cpu() for view, images in paired.items()}
        for state, model in models.items():
            for mode in ['batch', 'saved']:
                with torch.no_grad():
                    fs = {view: outputs(model, z, y, mode) for view, z in features.items()}
                    stability = dict(
                        winner_agreement=float(fs['original']['winner'].eq(fs['H2']['winner']).double().mean()),
                        actual_prediction_agreement=float(fs['original']['full'].argmax(1).eq(fs['H2']['full'].argmax(1)).double().mean()),
                        mean_router_L1_view_difference=float((fs['original']['pi'] - fs['H2']['pi']).abs().sum(1).mean()),
                        mean_teacher_L1_view_difference=float((fs['original']['teacher'] - fs['H2']['teacher']).abs().sum(1).mean()))
                for view, z in features.items():
                    record = dict(state=state, scheduled_step=step, view=view, mean_mode=mode,
                                  image_ids=image_ids, metrics=metrics(fs[view], y, domain),
                                  paired_view_stability=stability,
                                  gradients=gradient_metrics(model, z, y, mode))
                    records.append(record)
        write(out / 'progress.json', dict(status='paired_batch_replay', completed_batch=step, records=len(records)))
    assert all(digest(path) == locks[name] for name, path in paths.items())
    assert digest(cache_path) == plan['source_cache_sha256']
    assert all(p.grad is None for model in models.values() for p in model.parameters())
    result = dict(plan=plan, original_source=source_results, paired_records=records,
                  limitations='Source-out previously used for model selection; exact H2 batches overlap. '
                              'Saved-state gradients are not historical gradients or independent confirmation. '
                              'Conditional utility/class groups are descriptive; no tuning or target reads.')
    write(out / 'results.json', result)
    write(out / 'progress.json', dict(status='complete', records=len(records), target_read=False, parameter_updates=False))
    print(json.dumps(dict(status='complete', records=len(records), target_read=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('destination', type=Path)
    main(parser.parse_args().destination.resolve())
