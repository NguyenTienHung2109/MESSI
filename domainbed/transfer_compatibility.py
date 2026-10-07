"""Reference-hull compatibility and episodic predictive feature learning.

The certificate is relative to a calibration-fixed convex hull, NOT the entire
row-bounded head class. See docs/transfer_compatibility.md for the derivation.
"""
import itertools
import math
import numpy as np
from scipy.stats import binomtest
import torch
from torch.nn import functional as F
from domainbed.predictive_compatibility import CandidateMap, project
from domainbed.compatibility_diagnostics import log_event


def bernstein_upper(values, lower, upper, alpha):
    """One-sided Maurer–Pontil Thm 4, rescaled from [0,1]."""
    values=np.asarray(values,dtype=float)
    if len(values)<2 or not 0<alpha<1 or lower>upper:
        raise ValueError('Invalid sample size, interval or confidence')
    if not np.isfinite(values).all() or values.min()<lower-1e-8 or values.max()>upper+1e-8:
        raise ValueError('Samples violate the declared deterministic interval')
    n=len(values);variance=float(values.var(ddof=1));log=math.log(2/alpha)
    correction=math.sqrt(2*variance*log/n)+7*(upper-lower)*log/(3*(n-1))
    return dict(mean=float(values.mean()),variance=variance,correction=correction,
                upper=min(upper,float(values.mean())+correction),range=[float(lower),float(upper)],n=n)


def entropy_box_lower(counts,alpha):
    """Simultaneous exact binomial intervals; minimize H on box∩simplex.

    A concave function attains its minimum at a vertex. At each vertex at most
    one coordinate is not at a box endpoint. Enumeration is exact for small C.
    """
    counts=np.asarray(counts,dtype=int);classes=len(counts);n=int(counts.sum())
    if classes>10 or classes<2 or n<2 or (counts<0).any() or not 0<alpha<1:
        raise ValueError('Requires 2..10 classes, n>=2, valid counts/alpha')
    intervals=[binomtest(int(k),n).proportion_ci(confidence_level=1-alpha/classes,method='exact') for k in counts]
    lo=np.array([i.low for i in intervals]);hi=np.array([i.high for i in intervals])
    best=math.log(classes);witness=None
    for free in range(classes):
        fixed=[i for i in range(classes) if i!=free]
        for mask in itertools.product([0,1],repeat=classes-1):
            p=lo.copy();p[fixed]=np.where(mask,hi[fixed],lo[fixed])
            p[free]=1-p[fixed].sum()
            if lo[free]-1e-12<=p[free]<=hi[free]+1e-12:
                q=p[p>0];entropy=float(-(q*np.log(q)).sum())
                if entropy<=best:best,witness=entropy,p.tolist()
    if witness is None:raise RuntimeError('No entropy-box vertex found')
    empirical=counts/n;q=empirical[empirical>0]
    return dict(empirical=float(-(q*np.log(q)).sum()),lower=max(0.,best),
                class_lower=lo.tolist(),class_upper=hi.tolist(),minimizer=witness)


def ce_interval(w):
    # CE_y = log sum_c exp((w_c-w_y)z), ||z||<=1.
    distances=torch.cdist(w,w)
    lower=float(torch.logsumexp(-distances,dim=1).min())
    upper=float(torch.logsumexp(distances,dim=1).max())
    return lower,upper


def hull_audit(w,vertices,z,y,tau,delta,alpha_event):
    """Fixed deployed w; vertices must include w and all calibration references.

    Population convexity bounds regret on conv(vertices) by the largest
    expected directional derivative <∇CE(w), w-v>. Each is a fixed scalar
    observable and receives its own confidence event.
    """
    w=w.double();vertices=vertices.double();z=z.double();y=y.long()
    if not torch.isfinite(z).all() or (z.norm(dim=1)>1+1e-6).any():
        raise ValueError('Audit features must have norm <=1')
    if not torch.isfinite(w).all() or not torch.isfinite(vertices).all():
        raise ValueError('Heads must be finite')
    if not any(torch.equal(w,v) for v in vertices):
        raise ValueError('Deployed head must be included in fixed hull')
    probability=(z@w.T).softmax(1)
    residual=probability-F.one_hot(y,len(w)).double()
    directional=[]
    for vertex in vertices:
        difference=w-vertex
        # Invariant to adding a common row vector to all logits.
        bound=float(torch.cdist(difference,difference).max())
        value=(residual*(z@difference.T)).sum(1).numpy()
        estimate=bernstein_upper(value,-bound,bound,alpha_event)
        directional.append(dict(estimate,head_difference_bound=bound))
    regret_upper=max(0.,max(d['upper'] for d in directional))
    risk=F.cross_entropy(z@w.T,y,reduction='none').numpy()
    risk_bound=bernstein_upper(risk,*ce_interval(w),alpha_event)
    entropy=entropy_box_lower(torch.bincount(y,minlength=len(w)).numpy(),alpha_event)
    compatible=regret_upper<=tau
    predictive=risk_bound['upper']<=entropy['lower']-delta
    return dict(scope='calibration_fixed_reference_hull',n=len(y),regret_upper=regret_upper,
                empirical_tangent_gap=max(0.,max(d['mean'] for d in directional)),
                directional=directional,risk=risk_bound,entropy=entropy,
                compatible=compatible,predictive=predictive,accepted=compatible and predictive,
                tau=tau,delta=delta,alpha_per_event=alpha_event)


def ridge_head(z,targets,regularization=.01):
    if regularization<=0:raise ValueError('Strictly positive ridge penalty required')
    n=len(z)
    return torch.linalg.solve(z.T@z/n+regularization*torch.eye(z.shape[1],dtype=z.dtype,device=z.device),z.T@targets/n)


def train_transfer_map(bank,splits,support,args,classes,seed,log_path=None,expert=None,
                       mode='cross_domain'):
    """ERM + ridge query error. Query uses TRAIN samples only, never audit.

    within_domain is a compute-matched auxiliary-loss control: the query and
    inner samples come from the same domain (independent draws).
    """
    if mode not in ['cross_domain','within_domain']:raise ValueError('Unknown episode mode')
    torch.manual_seed(seed)
    model=CandidateMap(bank[support[0]][0].shape[1],args.dimension,args.hidden)
    head=torch.nn.Parameter(torch.zeros(classes,args.dimension))
    optimizer=torch.optim.Adam(list(model.parameters())+[head],lr=args.learning_rate)
    generator=torch.Generator().manual_seed(seed)
    inner_generator=torch.Generator().manual_seed(seed+7919)
    trace=[]
    for step in range(args.map_steps):
        minibatches={};batch_indices={}
        for d in support:
            ids=torch.as_tensor(splits[d]['train'])
            selected=ids[torch.randint(len(ids),(args.batch_size,),generator=generator)]
            batch_indices[d]=selected
            minibatches[d]=(model(bank[d][0][selected]),bank[d][1][selected])
        erm=torch.stack([F.cross_entropy(z@head.T,y) for z,y in minibatches.values()]).mean()
        query_domain=support[step%len(support)]
        query_z,query_y=minibatches[query_domain]
        inner_domains=[d for d in support if d!=query_domain] if mode=='cross_domain' else [query_domain]
        # Same inner sample count for cross/within controls. Separate RNG keeps
        # original ERM sample sequence exactly aligned across interventions.
        count=args.batch_size*(len(support)-1)
        per_domain=count//len(inner_domains)
        inner_z=[];inner_y=[]
        for d in inner_domains:
            ids=torch.as_tensor(splits[d]['train'])
            if d==query_domain:
                ids=ids[~torch.isin(ids,batch_indices[query_domain])]
                if len(ids)==0:raise ValueError('Need disjoint query/inner training samples')
            selected=ids[torch.randint(len(ids),(per_domain,),generator=inner_generator)]
            inner_z.append(model(bank[d][0][selected]));inner_y.append(bank[d][1][selected])
        z=torch.cat(inner_z);y=torch.cat(inner_y)
        target=F.one_hot(y,classes).to(z.dtype)
        a=ridge_head(z,target,args.ridge_lambda)
        query_target=F.one_hot(query_y,classes).to(z.dtype)
        transfer=(query_z@a-query_target).square().sum(1).mean()
        loss=erm+args.transfer_weight*transfer
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():head.copy_(project(head,args.radius))
        if step%100==0 or step==args.map_steps-1:
            event=dict(stage='map',mode=mode,expert=expert,step=step,query_domain=query_domain,
                       inner_domains=inner_domains,inner_n=len(y),erm_ce=float(erm.detach()),
                       query_squared_error=float(transfer.detach()),total=float(loss.detach()),
                       ridge_lambda=args.ridge_lambda,transfer_weight=args.transfer_weight)
            trace.append(event);log_event(log_path,event)
    return model.eval().requires_grad_(False),trace
