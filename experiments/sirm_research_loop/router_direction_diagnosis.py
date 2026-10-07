"""Source-only saved-router gradient alignment; derivatives, not update trials."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .centered_model import CenteredGateSIRM
from .hard_router_diagnosis import outputs
from .shrinkage import digest, write
from domainbed.deit_transformer import DeiTFeaturizer

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/sirm_research_loop'


def main():
    torch.set_num_threads(4); out = BASE / 'iteration_09/router_direction_diagnosis'; out.mkdir(exist_ok=True)
    result = {}; locks = {}
    for it in ['08', '09']:
        bank = BASE / f'iteration_{it}/experiment/runs/L38/OFF/source_best.pkl'
        pp = {s: BASE / f'iteration_{it}/source_router_diagnosis/{s}_original.npz' for s in ['source_in', 'source_out']}
        locks.update({str(p.relative_to(ROOT)): digest(p) for p in [bank, *pp.values()]})
        a = {s: dict(np.load(p)) for s, p in pp.items()}
        payload = torch.load(bank, map_location='cpu', weights_only=False); c = payload['config']
        bb = DeiTFeaturizer(pretrained=False, model_name=c['backbone']); m = CenteredGateSIRM(bb, bb.n_outputs, 10, 3, c)
        m.load_state_dict({k.removeprefix('model.'): v for k, v in payload['model_dict'].items()}); m.eval()
        m.requires_grad_(False); m.router.requires_grad_(True); params = list(m.router.parameters())
        def grad(loss):
            return torch.cat([g.detach().double().flatten() for g in torch.autograd.grad(loss, params, retain_graph=True)])
        def equal_domain(v, domains, mask):
            return torch.stack([v[mask & domains.eq(d)].mean() for d in domains.unique() if (mask & domains.eq(d)).any()]).mean()
        train = a['source_in']; z = torch.tensor(train['z']); y = torch.tensor(train['y']); dom = torch.tensor(train['domains'])
        f = outputs(m, z, y); all_mask = torch.ones(len(y), dtype=torch.bool)
        cls = equal_domain(F.cross_entropy(f['full'], y, reduction='none'), dom, all_mask)
        dist = equal_domain(F.kl_div(f['pi'].clamp_min(1e-30).log(), f['teacher'], reduction='none').sum(1), dom, all_mask)
        gg = {'classification': grad(cls), 'weighted_distillation': grad(c['lambda_router'] * dist)}
        val = a['source_out']; yv = torch.tensor(val['y']); dv = torch.tensor(val['domains']); fv = outputs(m, torch.tensor(val['z']), yv)
        loss = F.cross_entropy(fv['full'], yv, reduction='none'); wrong = fv['g'].argmax(1).ne(yv)
        masks = {'all': torch.ones(len(yv), dtype=torch.bool), 'global_wrong': wrong, 'global_correct': ~wrong}
        for d in dv.unique(): masks[f'domain/{int(d)}'] = dv.eq(d)
        rows = {}
        for name, mask in masks.items():
            vg = grad(equal_domain(loss, dv, mask)); rows[name] = {'n': int(mask.sum()), 'val_grad_norm': float(vg.norm()), 'train_direction_effects': {}}
            for key, tg in gg.items():
                rows[name]['train_direction_effects'][key] = {
                    'cosine': float(F.cosine_similarity(vg, tg, dim=0)),
                    'validation_CE_derivative_for_unit_descent_direction': float(-torch.dot(vg, tg) / tg.norm()),
                    'validation_CE_derivative_for_unnormalized_descent_direction': float(-torch.dot(vg, tg))}
        result[it] = {'train_gradient_norms': {k: float(v.norm()) for k, v in gg.items()},
                      'train_aux_to_classification_norm_ratio': float(gg['weighted_distillation'].norm() / gg['classification'].norm()),
                      'train_gradient_cosine': float(F.cosine_similarity(*gg.values(), dim=0)), 'validation_groups': rows}
        assert all(p.grad is None for p in m.parameters())
    assert all(digest(ROOT / p) == h for p, h in locks.items())
    write(out / 'results.json', {'code_sha256': digest(Path(__file__)), 'input_hashes': locks,
                                'scope': 'Exact source-in full original-view equal-domain gradients versus source-out CE gradients, router parameters only. '
                                         'Saved selected08/09 states; no optimizer, update, fitting, target read or selection. '
                                         'Negative derivative predicts infinitesimal CE decrease; this is not an AdamW or finite-step/accuracy prediction. '
                                         'Source-out selected these checkpoints and is exploratory, not an independent evaluation.',
                                'iterations': result, 'inputs_unchanged': True, 'target_read': False, 'parameter_updates': False})
    print(result)


if __name__ == '__main__': main()
