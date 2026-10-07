"""Read-only common-endpoint source comparison; never loads target data."""
import json
from pathlib import Path

import numpy as np
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from .centered_model import CenteredGateSIRM
from .hard_router_diagnosis import outputs, metrics, gradient_metrics
from .shrinkage import ROOT, digest, write


def main():
    torch.set_num_threads(4)
    dirs = {str(i): ROOT / f'results/sirm_research_loop/iteration_{i:02d}'
            for i in [8, 9, 10]}
    out = dirs['10'] / 'common_step_source_comparison'
    out.mkdir(exist_ok=True)
    models, locks = {}, {}
    for key, dest in dirs.items():
        path = dest / 'experiment/runs/L38/OFF/last.pkl'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['step'] == 399
        config = payload['config']
        backbone = DeiTFeaturizer(pretrained=False, model_name=config['backbone'])
        model = CenteredGateSIRM(backbone, backbone.n_outputs, 10, 3, config)
        model.load_state_dict({k.removeprefix('model.'): v
                               for k, v in payload['model_dict'].items()})
        model.configure_stage(); model.eval()
        assert all(not p.requires_grad for mod in [model.featurizer, model.global_head]
                   for p in mod.parameters())
        models[key] = model
        locks[key] = dict(checkpoint_sha256=digest(path), step=399,
                          source_cache_sha256=digest(dest / 'source_feature_cache.pt'))
    assert len({v['source_cache_sha256'] for v in locks.values()}) == 1
    a, b = models['9'].config, models['10'].config
    differences = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys()
                   if a.get(k) != b.get(k)}
    assert set(differences) <= {'protocol', 'center_feature_cache', 'router_strength_control'}
    hard_pair_differences = {k: [models['8'].config.get(k), a.get(k)]
        for k in models['8'].config.keys() | a.keys()
        if models['8'].config.get(k) != a.get(k)}
    assert set(hard_pair_differences) <= {'protocol', 'center_feature_cache',
        'discovery_strength_control', 'discovery_assignment'}
    assert models['8'].config['discovery_assignment'] == 'hard_winner'
    assert a['discovery_assignment'] == 'pair_shared'
    for module in ['featurizer', 'global_head']:
        aa, bb = getattr(models['9'], module).state_dict(), getattr(models['10'], module).state_dict()
        assert aa.keys() == bb.keys() and all(torch.equal(aa[k], bb[k]) for k in aa)
    cache_dir = dirs['9'] / 'source_router_diagnosis'
    caches = {split: cache_dir / f'{split}_original.npz'
              for split in ['source_in', 'source_out']}
    results = dict(scope='Source original views only; common final step399. No fitting, '
                   'updates, target reads or selection. Saved-state gradients are not historical '
                   'training gradients. Label teacher is an oracle diagnostic.',
                   code_sha256=digest(Path(__file__)), checkpoint_locks=locks,
                   config_differences=differences,
                   hard_vs_pair_config_differences=hard_pair_differences,
                   cache_hashes={k: digest(p) for k, p in caches.items()}, splits={})
    identities = {}
    for split, path in caches.items():
        with np.load(path) as npz:
            z, y, domains = [torch.from_numpy(npz[k]) for k in ['z', 'y', 'domains']]
            ids = npz['ids'].tolist()
        assert set(domains.tolist()) == {0, 2, 3}
        identities[split] = set(ids)
        split_results, replay = {}, {}
        for key, model in models.items():
            with torch.no_grad():
                f = outputs(model, z, y)
                m = metrics(f, y, domains)
                wrong = f['g'].argmax(1).ne(y)
                single = (f['g'][:, None] + f['delta']).argmax(2).eq(y[:, None]).any(1)
                routed = (f['g'][:, None] + f['pi'][:, :6, None] * f['delta']).argmax(2).eq(y[:, None]).any(1)
                m['capacity'] = dict(global_errors=int(wrong.sum()),
                    any_unscaled_repair=int((wrong & single).sum()),
                    any_routed_single_repair=int((wrong & routed).sum()))
                replay[key] = f
            split_results[key] = dict(metrics=m,
                raw_router_gradient_reference=gradient_metrics(model, z, y, 'saved'))
        assert torch.equal(replay['9']['g'], replay['10']['g'])
        assert torch.equal(replay['8']['g'], replay['9']['g'])
        c9, c10 = [replay[k]['full'].argmax(1).eq(y) for k in ['9', '10']]
        changed = c9.ne(c10)
        split_results['paired_correctness'] = dict(
            new_correct=int((~c9 & c10).sum()), new_wrong=int((c9 & ~c10).sum()),
            cases=[dict(id=ids[i], label=int(y[i]), domain=int(domains[i]),
                        prediction9=int(replay['9']['full'][i].argmax()),
                        prediction10=int(replay['10']['full'][i].argmax()))
                   for i in changed.nonzero().flatten().tolist()])
        results['splits'][split] = split_results
    assert not identities['source_in'] & identities['source_out']
    for key, dest in dirs.items():
        assert digest(dest / 'experiment/runs/L38/OFF/last.pkl') == locks[key]['checkpoint_sha256']
    write(out / 'results.json', results)
    print(json.dumps({s: {k: v['metrics']['equal_domain'] for k, v in r.items()
                          if k in models} for s, r in results['splits'].items()}, indent=2))


if __name__ == '__main__':
    main()
