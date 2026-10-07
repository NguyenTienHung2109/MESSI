"""Source-only matched-state comparison of centering versus iteration04."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.sirm_discover.math import multiple_choice_loss
from experiments.terra_10pct_factor_diagnosis.report import data_for
from .centered_model import CenteredGateSIRM, centered_head

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'results/sirm_research_loop/iteration_07'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    torch.set_num_threads(4)
    paths = {name: ROOT/f'results/sirm_research_loop/iteration_{number}/experiment/runs/L38/OFF/last.pkl'
             for name, number in [('reference', '04'), ('centered', '07')]}
    models = {}; payloads = {}; locks = {}
    for name, path in paths.items():
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['step'] == 399
        config = payload['config']
        backbone = DeiTFeaturizer(pretrained=False, model_name=config['backbone'])
        cls = CenteredGateSIRM if name == 'centered' else GateSIRM
        model = cls(backbone, backbone.n_outputs, 10, 3, config)
        model.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()})
        model.configure_stage(); models[name] = model.eval(); payloads[name] = payload
        locks[name] = dict(path=str(path), sha256=digest(path), step=399)
    for module in ['featurizer', 'global_head']:
        a = getattr(models['reference'], module).state_dict(); b = getattr(models['centered'], module).state_dict()
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    old = payloads['reference']['config']; new = payloads['centered']['config']
    numeric_differences = {k: [old[k], new[k]] for k in old if isinstance(old[k], (float, int, bool)) and old[k] != new[k]}
    assert not numeric_differences
    manifest = json.loads((DEST/'experiment/subset_manifest.json').read_text())
    out = DEST/'centering_equal_step'; out.mkdir(exist_ok=True)
    (out/'locks.json').write_text(json.dumps(dict(checkpoints=locks, code_sha256=digest(Path(__file__)), scope='Source-out original views, matched last state399; no target reads, fitting or new checkpoint selection.'), indent=2)+'\n')
    features = []; labels = []; domains = []
    backbone = models['centered'].featurizer.cuda()
    with torch.no_grad():
        for domain, ds in sorted(data_for(manifest, [0, 2, 3], new, 'out').items()):
            for x, y in DataLoader(ds, batch_size=64, num_workers=2):
                features.append(backbone(x.cuda()).cpu()); labels.append(y); domains.append(torch.full_like(y, domain))
    backbone.cpu(); torch.cuda.empty_cache()
    z = torch.cat(features); y = torch.cat(labels); domain = torch.cat(domains)
    results = {}
    with torch.no_grad():
        for name, model in models.items():
            g, delta, pi = centered_head(model, z)
            logits = g + (pi[:, :6, None] * delta).sum(1)
            ce = F.cross_entropy(logits, y, reduction='none')
            global_ce = F.cross_entropy(g, y, reduction='none')
            _, responsibility = multiple_choice_loss(g, delta, y, old['discovery_temperature'])
            entropy = -(responsibility * responsibility.clamp_min(1e-30).log()).sum(1)
            results[name] = dict(
                source_accuracy=float(torch.stack([logits[domain.eq(d)].argmax(1).eq(y[domain.eq(d)]).float().mean() for d in [0, 2, 3]]).mean()),
                source_CE=float(torch.stack([ce[domain.eq(d)].mean() for d in [0, 2, 3]]).mean()),
                global_source_CE=float(torch.stack([global_ce[domain.eq(d)].mean() for d in [0, 2, 3]]).mean()),
                normalized_responsibility_entropy=float(entropy.mean()/np.log(6)),
                mean_max_responsibility=float(responsibility.max(1).values.mean()),
                mean_expert_norm=float(delta.norm(dim=2).mean()),
            )
    assert all(digest(paths[k]) == locks[k]['sha256'] for k in paths)
    result = dict(checkpoints=locks, global_exact_equal=True, matched_step=399, results=results,
                  limitation='Matched-state source diagnostic, one seed; source-out was used for original selection. Means restored from each saved model. Train H2/batch centering versus original-view inference remains a distinct normalization issue. Lower entropy or CE alone does not establish complementary factors or transfer.')
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    run()
