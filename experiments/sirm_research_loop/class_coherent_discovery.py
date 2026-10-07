"""Label-class coherent hard discovery; original image weighting and balance retained."""
import torch
from experiments.sirm_discover.math import expert_ce,multiple_choice_loss


def class_coherent_cost(losses,y):
    if losses.ndim!=2 or len(losses)!=len(y) or not len(y):raise ValueError('nonempty paired image/expert losses and labels required')
    total=losses.sum()*0
    for label in y.unique(sorted=True):
        mask=y==label
        # One selected expert for all occurrences of this label in this minibatch.
        # count/N preserves original image weighting, without class balancing.
        total=total+losses[mask].mean(0).amin()*mask.sum()/len(y)
    return total


def class_coherent_loss(global_logits,delta,y,temperature):
    _,original_balance=multiple_choice_loss(global_logits,delta,y,temperature)
    losses=expert_ce(global_logits[:,None,:]+delta,y)
    return class_coherent_cost(losses,y),original_balance
