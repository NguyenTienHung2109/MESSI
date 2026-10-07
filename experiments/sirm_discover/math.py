"""Residual utility, detached soft/null assignment and component IRMv1."""
import itertools
import torch
import torch.nn.functional as F


def center_logits(delta):
    return delta-delta.mean(-1,keepdim=True)


def entmax15(scores):
    """Exact full-sort alpha=1.5 transform on the last axis, no dependency.

    Threshold equation follows Peters et al., https://github.com/deep-spin/entmax.
    Used only for detached teachers/Q; the prediction router uses softmax.
    """
    x=(scores-scores.max(-1,keepdim=True).values)/2
    sorted_x=x.sort(-1,descending=True).values
    rho=torch.arange(1,x.shape[-1]+1,device=x.device,dtype=x.dtype)
    mean=sorted_x.cumsum(-1)/rho
    variance_sum=rho*(sorted_x.square().cumsum(-1)/rho-mean.square())
    tau=mean-((1-variance_sum)/rho).clamp_min(0).sqrt()
    support=(tau<=sorted_x).sum(-1,keepdim=True).clamp_min(1)
    threshold=tau.gather(-1,support-1)
    return (x-threshold).clamp_min(0).square()


def expert_ce(logits,y):
    n,m,c=logits.shape
    return F.cross_entropy(logits.reshape(-1,c),y[:,None].expand(n,m).reshape(-1),reduction='none').reshape(n,m)


def residual_utilities(global_logits,delta,pi,y):
    contribution=pi[:,:delta.shape[1],None]*delta
    full=global_logits+contribution.sum(1)
    full_ce=F.cross_entropy(full,y,reduction='none')
    # Ablation removes exactly one term; no routing renormalization.
    marginal=expert_ce(full[:,None,:]-contribution,y)-full_ce[:,None]
    single=F.cross_entropy(global_logits,y,reduction='none')[:,None]-expert_ce(global_logits[:,None,:]+delta,y)
    return full,contribution,marginal,single


def multiple_choice_loss(global_logits,delta,y,temperature):
    if temperature<=0:raise ValueError('discovery temperature must be positive')
    losses=expert_ce(global_logits[:,None,:]+delta,y)
    logits=-losses/temperature
    # log-mean-exp differs from the requested soft-min by a constant only.
    loss=-temperature*(torch.logsumexp(logits,1)-torch.log(losses.new_tensor(float(losses.shape[1])))).mean()
    responsibility=logits.softmax(1)
    return loss,responsibility


def router_teacher(single_utility,temperature,utility_floor=0.):
    if temperature<=0:raise ValueError('router temperature must be positive')
    u=single_utility.detach()
    scores=u/temperature
    scores=torch.cat([scores,torch.zeros_like(scores[:,:1])],1)
    probabilities=entmax15(scores)
    evidence=((u-utility_floor)/temperature).clamp(0,1)
    residual=probabilities[:,:u.shape[1]]*evidence
    return torch.cat([residual,(1-residual.sum(1,keepdim=True)).clamp(0,1)],1).detach()


def soft_null_q(utility,valid,temperature=.05,utility_floor=.002,interaction_floor=.002,eps=1e-8):
    if temperature<=0:raise ValueError('Q temperature must be positive')
    u=utility.detach()
    if u.ndim!=2 or valid.shape!=u.shape:raise ValueError('Q requires complete domain/expert matrices')
    # Do not impute missing cells into a two-way decomposition.
    if not valid.all():
        return torch.zeros_like(u),torch.ones_like(u[:,0]),torch.zeros_like(u[:,0]),torch.zeros_like(u)
    interaction=u-u.mean(1,keepdim=True)-u.mean(0,keepdim=True)+u.mean()
    raw_std=u.std(1,unbiased=False);interaction_std=interaction.std(1,unbiased=False)
    confidence=(interaction_std/(raw_std+eps)).clamp(0,1)
    confidence=confidence*(raw_std/temperature).clamp(0,1)
    scores=interaction/temperature
    scores=torch.cat([scores,torch.zeros_like(scores[:,:1])],1)
    probabilities=entmax15(scores)
    # Continuous gates at the effect floors: suppress nonpositive usefulness
    # without a hard eligible-logit mask or row renormalization jump.
    evidence=((u-utility_floor)/temperature).clamp(0,1)*((interaction-interaction_floor)/temperature).clamp(0,1)
    q=(probabilities[:,:u.shape[1]]*confidence[:,None]*evidence).detach()
    null=(1-q.sum(1)).clamp(0,1).detach()
    return q,null,confidence.detach(),interaction.detach()


def component_eligibility(pi,domains,q,min_ess=2.):
    """Original routing/count/ESS eligibility, also inspectable when lambda=0."""
    routing=pi[:,:q.shape[1]].detach();q=q.detach()
    valid=torch.zeros_like(q,dtype=torch.bool);ess=torch.zeros_like(q)
    for d in range(q.shape[0]):
        mask=domains==d
        for m in range(q.shape[1]):
            w=routing[mask,m];total=w.sum()
            ess[d,m]=total.square()/w.square().sum().clamp_min(1e-12)
            valid[d,m]=mask.sum()>=2 and total>1e-8 and ess[d,m]>=min_ess
    eligible=sum(int(((q[i]*q[j]>0)&valid[i]&valid[j]).sum()) for i,j in itertools.combinations(range(q.shape[0]),2))
    return valid,ess,eligible


def component_irm(global_logits,delta,pi,y,domains,q,min_ess=2.,weighting='confidence'):
    """Scale one residual contribution with all other logits held fixed.

    Q, router probabilities and other experts' context are detached. Domain CE
    is unweighted: routing enters *inside* the marginal contribution. Pair
    weights are q_i*m q_j*m; a fixed pair-count divisor preserves NULL/confidence.
    """
    if weighting not in ('confidence','binary'):raise ValueError('unknown SIRM weighting')
    q=q.detach();routing=pi[:,:delta.shape[1]].detach()
    contribution=routing[:,:,None]*delta
    context=global_logits.detach()+contribution.detach().sum(1)
    num_domains,num_experts=q.shape
    valid,ess,_=component_eligibility(pi,domains,q,min_ess);gradients={}
    pairs=[]
    for i,j in itertools.combinations(range(num_domains),2):
        for m in range(num_experts):
            weight=q[i,m]*q[j,m]
            if weight<=0 or not valid[i,m] or not valid[j,m]:continue
            for d in [i,j]:
                if (d,m) in gradients:continue
                mask=domains==d;term=contribution[mask,m]
                scale=delta.new_ones((),requires_grad=True)
                logits=context[mask]-term.detach()+scale*term
                risk=F.cross_entropy(logits,y[mask])
                gradients[d,m]=torch.autograd.grad(risk,scale,create_graph=True,retain_graph=True)[0]
            raw=gradients[i,m].square()+gradients[j,m].square()
            pairs.append(raw if weighting=='binary' else weight*raw)
    zero=delta.sum()*0
    divisor=max(len(pairs),1) if weighting=='binary' else max(num_domains*(num_domains-1)//2,1)
    loss=sum(pairs,zero)/divisor
    return loss,dict(active=bool(pairs),pair_count=len(pairs),valid=valid.detach(),ess=ess.detach(),G={f'{d},{m}':float(g.detach()) for (d,m),g in gradients.items()})
