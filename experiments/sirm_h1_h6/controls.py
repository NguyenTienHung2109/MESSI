"""Reusable frozen-bank controls, metrics and source-only router fitting."""
import copy
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.subset_irm import per_sample_topk
from experiments.sirm_h1_h6.statistics import expert_losses


def mixture(logits, probabilities):
    return (logits * probabilities.unsqueeze(-1)).sum(1)


def metrics(cache, pi):
    logits, y = cache['expert_logits'], cache['labels']
    if pi.shape != logits.shape[:2] or not torch.isfinite(pi).all():
        raise ValueError('invalid routing shape or values')
    torch.testing.assert_close(pi.sum(1), torch.ones(len(y)), atol=1e-5, rtol=1e-5)
    if bool((pi < 0).any()):
        raise ValueError('negative routing')
    losses = expert_losses(logits, y)
    mixed = mixture(logits, pi)
    selected = (pi * losses).sum(1)
    correct = mixed.argmax(1).eq(y).float()
    return {
        'accuracy': correct.mean().item(), 'nll': F.cross_entropy(mixed, y).item(),
        'selection_loss': selected.mean().item(),
        'routing_regret': (selected - losses.min(1).values).mean().item(),
        'top1_expert_hit_rate': pi.argmax(1).eq(losses.argmin(1)).float().mean().item(),
        'expert_loss_margin': (losses.sort(1).values[:, 1] - losses.min(1).values).mean().item(),
        'loads': pi.mean(0).tolist(), 'entropy': -(pi * pi.clamp_min(1e-9).log()).sum(1).mean().item(),
        'ess': (1 / pi.square().sum(1)).mean().item(), 'samples': len(y),
    }


def subset_cache(cache, mask):
    return {k: v[mask] if torch.is_tensor(v) and len(v.shape) and len(v) == len(mask)
            else [x for x, keep in zip(v, mask.tolist()) if keep]
            if isinstance(v, list) and len(v) == len(mask) else v for k, v in cache.items()}


class InputRouter(nn.Module):
    def __init__(self, mean, std, experts, hidden=128):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.network = nn.Sequential(nn.Linear(len(mean), hidden), nn.GELU(), nn.Linear(hidden, experts))

    def forward(self, x, noise=0.):
        z = (x - self.mean) / self.std
        if noise:
            z = z + torch.randn_like(z) * noise
        return self.network(z).softmax(-1)


def fit_router(fit, validation, config, temperature=1., sigma=0., teacher_q=None,
               global_only=False, allowed=None, seed=0):
    if fit['metadata']['kind'] != 'source' or validation['metadata']['kind'] != 'source':
        raise ValueError('target caches cannot fit or select routers')
    if fit['metadata']['split'] != 'in' or validation['metadata']['split'] != 'out':
        raise ValueError('fit on source-in; select on source-out')
    torch.manual_seed(seed)
    x, y = fit['projected_feature'], fit['labels']
    val_x = validation['projected_feature']
    experts = fit['expert_logits'].shape[1]
    if global_only:
        model = nn.Parameter(torch.zeros(experts))
        parameters = [model]
    else:
        model = InputRouter(x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6), experts, config['router_hidden'])
        parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=config['router_lr'])
    domain_weights = torch.zeros(len(y))
    for domain in fit['domains'].unique():
        mask = fit['domains'] == domain
        domain_weights[mask] = 1 / (len(fit['domains'].unique()) * int(mask.sum()))
    teacher = ((-expert_losses(fit['expert_logits'], y) / temperature).softmax(1)
               if teacher_q is None else teacher_q[fit['domains']]).detach()
    def probabilities(features, training=False):
        pi = model.softmax(0).expand(len(features), -1) if global_only else model(features, sigma if training else 0.)
        if allowed is not None:
            pi = pi * allowed
            pi = pi / pi.sum(1, keepdim=True).clamp_min(1e-9)
        return pi
    best = float('inf')
    best_state, best_epoch = None, None
    for epoch in range(config['router_epochs']):
        optimizer.zero_grad()
        pi = probabilities(x, True)
        if global_only:
            # A strong control: optimize mixture CE, rather than collapsing the linear selection objective to one expert.
            loss = (F.cross_entropy(mixture(fit['expert_logits'], pi), y, reduction='none') * domain_weights).sum()
        else:
            loss = (F.kl_div(pi.clamp_min(1e-9).log(), teacher, reduction='none').sum(1) * domain_weights).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_pi = probabilities(val_x)
            per_image = F.cross_entropy(mixture(validation['expert_logits'], val_pi if global_only else per_sample_topk(val_pi, 2)),
                                        validation['labels'], reduction='none')
            score = torch.stack([per_image[validation['domains'] == d].mean() for d in validation['domains'].unique()]).mean().item()
        if score < best:
            best, best_epoch = score, epoch
            best_state = model.detach().clone() if global_only else copy.deepcopy(model.state_dict())
    return {'kind': 'global' if global_only else 'input', 'state': best_state,
            'allowed': allowed, 'temperature': temperature, 'sigma': sigma,
            'validation_nll': best, 'selected_epoch': best_epoch, 'hidden': config['router_hidden']}


def route(router, cache, shuffle_features=False, seed=0):
    if router['kind'] == 'global':
        pi = router['state'].softmax(0).expand(len(cache['labels']), -1)
    else:
        state = router['state']
        model = InputRouter(state['mean'], state['std'], state['network.2.bias'].numel(), router['hidden'])
        model.load_state_dict(state)
        x = cache['projected_feature']
        if shuffle_features:
            x = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))]
        with torch.no_grad():
            pi = model(x)
    if router['allowed'] is not None:
        pi = pi * router['allowed']
        pi = pi / pi.sum(1, keepdim=True).clamp_min(1e-9)
    return pi


def shuffle_routes(pi, cache, within_class=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shuffled = pi.clone()
    if within_class:
        for d in cache['domains'].unique():
            for c in cache['labels'].unique():
                indices = ((cache['domains'] == d) & (cache['labels'] == c)).nonzero().flatten()
                shuffled[indices] = pi[indices[torch.randperm(len(indices), generator=generator)]]
    else:
        shuffled = pi[torch.randperm(len(pi), generator=generator)]
    return shuffled


def frozen_controls(cache, routers, seed=0):
    n, m, _ = cache['expert_logits'].shape
    losses = expert_losses(cache['expert_logits'], cache['labels'])
    result = {'uniform': torch.full((n, m), 1 / m), 'native': cache['router_topk'],
              'oracle_ce_single': F.one_hot(losses.argmin(1), m).float()}
    for name, router in routers.items():
        if name in ['metadata', 'h6_statistics']:
            continue
        pi = route(router, cache)
        result[name] = per_sample_topk(pi, 2) if router['kind'] == 'input' else pi
        if name == 'input_sigma0':
            result['input_dense'] = pi
            result['input_top1'] = per_sample_topk(pi, 1)
            result['input_shuffle_feature'] = per_sample_topk(route(router, cache, True, seed), 2)
            result['input_shuffle_all'] = shuffle_routes(result[name], cache, False, seed)
            result['input_shuffle_domain_class'] = shuffle_routes(result[name], cache, True, seed)
            for m in range(pi.shape[1]):
                ablated = result[name].clone()
                ablated[:, m] = 0
                mass = ablated.sum(1, keepdim=True)
                fallback = pi.clone(); fallback[:, m] = 0
                fallback = fallback / fallback.sum(1, keepdim=True).clamp_min(1e-9)
                result[f'drop_expert{m}'] = torch.where(mass > 0, ablated / mass.clamp_min(1e-9), fallback)
    # CE oracle and correctness oracle intentionally have different names.
    correctness = cache['expert_logits'].argmax(-1).eq(cache['labels'].unsqueeze(1))
    result['oracle_correct_single'] = F.one_hot(correctness.long().argmax(1), cache['expert_logits'].shape[1]).float()
    best_loss = torch.full((n,), float('inf'))
    best_pi = torch.zeros(n, cache['expert_logits'].shape[1])
    for a, b in itertools.combinations(range(cache['expert_logits'].shape[1]), 2):
        for weight in torch.linspace(0, 1, 11):
            pi = torch.zeros_like(best_pi)
            pi[:, a], pi[:, b] = weight, 1 - weight
            ce = F.cross_entropy(mixture(cache['expert_logits'], pi), cache['labels'], reduction='none')
            improve = ce < best_loss
            best_loss[improve], best_pi[improve] = ce[improve], pi[improve]
    result['oracle_grid_pair'] = best_pi
    return result


def paired_delta(cache, first, second, seed=0, repeats=500):
    """Paired image bootstrap, never treating transforms as independent samples."""
    y, logits = cache['labels'], cache['expert_logits']
    a, b = mixture(logits, first), mixture(logits, second)
    losses = expert_losses(logits, y)
    deltas = torch.stack([
        a.argmax(1).eq(y).float() - b.argmax(1).eq(y).float(),
        F.cross_entropy(a, y, reduction='none') - F.cross_entropy(b, y, reduction='none'),
        ((first - second) * losses).sum(1)], 1)
    generator = torch.Generator().manual_seed(seed)
    samples = torch.stack([deltas[torch.randint(len(y), (len(y),), generator=generator)].mean(0)
                           for _ in range(repeats)])
    return {name: {'delta': deltas[:, index].mean().item(),
                   'image_bootstrap_ci95': samples[:, index].quantile(torch.tensor([.025, .975])).tolist()}
            for index, name in enumerate(['accuracy', 'nll', 'routing_regret'])} | {
            'uncertainty_unit': 'parent image; training uncertainty requires multiple seeds'}
