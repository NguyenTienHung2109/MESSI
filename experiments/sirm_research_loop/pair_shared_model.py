"""Pair-coupled discovery with detached per-batch gradient-norm control."""
from unittest.mock import patch
import torch
from .centered_model import CenteredGateSIRM
from .hard_discovery import hard_winner_loss
from .pair_shared_discovery import paired_shared_loss

def norm(gs):return sum(g.detach().double().square().sum() for g in gs).sqrt()
class PairSharedGateSIRM(CenteredGateSIRM):
    def update(self,minibatches):
        assert self.config['discovery_assignment']=='pair_shared'
        assert self.config['discovery_strength_control']=='match_hard_parameter_gradient'
        assert self.config['balance_assignment']=='original_soft'
        domains=torch.cat([torch.full_like(y,i) for i,(_,y) in enumerate(minibatches)])
        observed={}
        def discover(g,delta,y,temperature):
            hard,balance=hard_winner_loss(g,delta,y,temperature)
            pair,details=paired_shared_loss(g,delta,y,domains)
            params=list(self.residuals.parameters())
            hg=torch.autograd.grad(hard,params,retain_graph=True)
            pg=torch.autograd.grad(pair,params,retain_graph=True)
            hn,pn=norm(hg),norm(pg)
            if pn==0:
                # Reference fallback retains a nonzero hard gradient if pair cancels.
                loss=hard;scale=hn.new_tensor(1.);fallback=True;matched_norm=hn
            else:
                scale=(hn/pn).detach().to(pair.dtype);loss=scale*pair;fallback=False
                matched_norm=norm(torch.autograd.grad(loss,params,retain_graph=True))
            assert torch.isfinite(scale) and torch.isfinite(loss)
            assert torch.allclose(matched_norm,hn,rtol=1e-5,atol=1e-8)
            observed.update(discovery_assignment='pair_shared',discovery_strength_control='match_hard_parameter_gradient',discovery_hard_parameter_grad_norm=float(hn),discovery_pair_parameter_grad_norm=float(pn),discovery_matched_parameter_grad_norm=float(matched_norm),discovery_detached_scale=float(scale),discovery_pair_loss_unscaled=float(pair.detach()),discovery_hard_loss=float(hard.detach()),discovery_pair_count=len(details['pairs']),discovery_paired_classes=details.get('paired_classes',0),discovery_no_pair_fallback=details['fallback'],discovery_zero_pair_gradient_fallback=fallback,discovery_pair_best_worst_excess=[r['best_worst_excess'] for r in details['pairs']])
            return loss,balance
        with patch('experiments.sirm_discover.model.multiple_choice_loss',discover):
            result=super().update(minibatches)
        self.last_diagnostics.update(observed)
        return result
