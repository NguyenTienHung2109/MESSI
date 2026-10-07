"""Per-sample expert loss for H1-H6 suitability metrics."""
import torch
import torch.nn.functional as F


def expert_losses(logits, labels):
    return torch.stack([F.cross_entropy(logits[:, m], labels, reduction="none")
                        for m in range(logits.shape[1])], 1)

import itertools


def feasible_supports(experts=6):
    supports = []
    for pairs in itertools.product(list(itertools.combinations(range(experts), 2)), repeat=3):
        s = torch.zeros(3, experts, dtype=torch.bool)
        for d, pair in enumerate(pairs):
            s[d, list(pair)] = True
        count = s.sum(0)
        if bool(((count == 0) | (count >= 2)).all()):
            supports.append(s)
    return torch.stack(supports)


def choose_assignment(risk, compatible, criterion, seed=0, weight=1.):
    if risk.shape != (3, 6):
        raise ValueError('H6 requires three source domains and six experts')
    if criterion == 'Global':
        return torch.full_like(risk, 1/6).detach()
    if criterion == 'FixedPairs':
        q = torch.zeros_like(risk)
        q[0, [0, 1]], q[1, [0, 2]], q[2, [1, 2]] = .5, .5, .5
        return q.detach()
    supports = feasible_supports().to(risk.device)
    if criterion == 'Random':
        idx = torch.randint(len(supports), (1,), generator=torch.Generator().manual_seed(seed)).item()
    else:
        scores = (supports * risk.unsqueeze(0)).sum((1, 2)) / 6
        if criterion in ['Grad', 'Regret']:
            masks = sum(supports[:, d].long() * (1 << d) for d in range(3))
            active = masks > 0
            columns = torch.arange(6, device=risk.device).expand_as(masks)
            scores += weight * (compatible[masks, columns] * active).sum(1) / active.sum(1)
        elif criterion != 'Risk':
            raise ValueError(criterion)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError('nonfinite assignment objective')
        idx = scores.argmin().item()
    return supports[idx].to(risk.dtype).div(2).detach()


def classifier_gradient_gaps(cache, min_count=2, min_norm=1e-8):
    z, logits, y, domains = [cache[k] for k in ['expert_features','expert_logits','labels','domains']]
    classes = logits.shape[-1]
    gradients, support = {}, {}
    for d in range(3):
        for c in range(classes):
            mask = (domains == d) & (y == c)
            support[f'{d}:{c}'] = int(mask.sum())
            if int(mask.sum()) < min_count:
                continue
            residual = logits[mask].softmax(-1) - F.one_hot(y[mask], classes).float().unsqueeze(1)
            features = torch.cat([z[mask], torch.ones_like(z[mask, :, :1])], -1)
            gradients[d,c] = (torch.einsum('nmc,nmf->mcf',residual,features)/int(mask.sum())).flatten(1)
    gap, counts = torch.zeros(8,6), torch.zeros(8,6)
    for subset in [3,5,6,7]:
        for d,e in itertools.combinations([d for d in range(3) if subset & (1 << d)],2):
            for c in range(classes):
                if (d,c) not in gradients or (e,c) not in gradients:
                    continue
                a,b=gradients[d,c],gradients[e,c]
                valid=(a.norm(dim=1)>min_norm)&(b.norm(dim=1)>min_norm)
                gap[subset]+=F.relu(-F.cosine_similarity(a,b,dim=1))*valid
                counts[subset]+=valid
    if bool((counts[[3,5,6,7]]==0).any()):
        raise ValueError('insufficient nonzero class/domain gradients for H6')
    return gap/counts.clamp_min(1),dict(class_counts=support,gradient_pair_counts=counts.tolist(),min_count=min_count,min_norm=min_norm)


def fit_linear_probe(x,y,classes,mean,std,regularization=.01,max_iter=250,tolerance=1e-4,sample_weights=None):
    x=(x.double()-mean)/std
    w=torch.zeros(x.shape[1],classes,dtype=torch.double,requires_grad=True)
    b=torch.zeros(classes,dtype=torch.double,requires_grad=True)
    optimizer=torch.optim.LBFGS([w,b],max_iter=max_iter,tolerance_grad=tolerance/10,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        ce=F.cross_entropy(x@w+b,y,reduction='none')
        loss=(ce.mean() if sample_weights is None else (ce*sample_weights).sum())+regularization*w.square().sum()/2
        loss.backward()
        return loss
    optimizer.step(closure)
    closure()
    norm=max(w.grad.abs().max().item(),b.grad.abs().max().item())
    if not norm <= tolerance:
        raise RuntimeError(f'linear probe not converged: max gradient={norm:g}; increase probe_max_iter')
    return dict(w=w.detach(),b=b.detach(),mean=mean,std=std,gradient_max=norm)


def probe_logits(probe,x):
    return (x.double()-probe['mean'])/probe['std']@probe['w']+probe['b']


def fit_gap_probes(fit,regularization=.01,max_iter=250,tolerance=1e-4):
    probes={}
    for m in range(6):
        # Shared scale keeps regularization identical across pooled/separate probes.
        all_x=fit['expert_features'][:,m].double()
        mean,std=all_x.mean(0),all_x.std(0,unbiased=False).clamp_min(1e-6)
        for subset in range(1,8):
            mask=torch.zeros_like(fit['labels'],dtype=torch.bool)
            for d in range(3):
                if subset & (1 << d):
                    mask|=fit['domains']==d
            weights=torch.zeros(int(mask.sum()),dtype=torch.double)
            domains=fit['domains'][mask]
            for d in domains.unique():
                weights[domains==d]=1/(len(domains.unique())*int((domains==d).sum()))
            probes[m,subset]=fit_linear_probe(all_x[mask],fit['labels'][mask],fit['expert_logits'].shape[-1],
                                              mean,std,regularization,max_iter,tolerance,sample_weights=weights)
    return probes


def evaluate_gaps(probes,evaluation):
    gap,se=torch.zeros(8,6),torch.zeros(8,6)
    for m in range(6):
        for subset in [3,5,6,7]:
            deltas,variances=[],[]
            for d in range(3):
                if not subset & (1 << d):
                    continue
                mask=evaluation['domains']==d
                x,y=evaluation['expert_features'][mask,m],evaluation['labels'][mask]
                if len(y)<2:
                    raise ValueError('gap evaluation requires >=2 images/domain')
                delta=F.cross_entropy(probe_logits(probes[m,subset],x),y,reduction='none')-F.cross_entropy(probe_logits(probes[m,1<<d],x),y,reduction='none')
                deltas.append(delta.mean());variances.append(delta.var(unbiased=True)/len(y))
            gap[subset,m]=torch.stack(deltas).mean()
            se[subset,m]=torch.stack(variances).sum().sqrt()/len(deltas)
    return gap,se
