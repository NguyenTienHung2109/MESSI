"""Candidate source-class/domain-pair discovery. Not enabled in primary training."""
import itertools
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import expert_ce

def paired_shared_loss(g,delta,y,domains):
    # Frozen-global CE is a difficulty baseline, never differentiated.
    base=F.cross_entropy(g.detach(),y,reduction='none')
    excess=expert_ce(g.detach()[:,None]+delta,y)-base[:,None]
    class_terms=[];records=[]
    for cls in y.unique(sorted=True):
        means={int(d):excess[(y==cls)&(domains==d)].mean(0) for d in domains[y==cls].unique(sorted=True)}
        pair_terms=[]
        for i,j in itertools.combinations(sorted(means),2):
            worst=torch.stack([means[i],means[j]]).amax(0)
            pair_terms.append(worst.amin(0))
            records.append({'class':int(cls),'domains':[i,j],'worst_excess':worst.detach().tolist(),'best_worst_excess':float(worst.amin().detach())})
        if pair_terms:class_terms.append(torch.stack(pair_terms).mean())
    if not class_terms:
        return excess.amin(1).mean(),{'fallback':True,'pairs':[],'reason':'No label occurs in two domains; per-image relative hard loss.'}
    return torch.stack(class_terms).mean(),{'fallback':False,'pairs':records,'paired_classes':len(class_terms),'scope':'Equal classes, then present domain pairs; minimize worst-domain excess CE of the same expert. No router/Q/threshold change.'}
