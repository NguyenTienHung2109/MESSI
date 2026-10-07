"""Candidate: class-coherent discovery on detached routed individual corrections."""
from unittest.mock import patch
import torch
from experiments.sirm_discover.math import expert_ce,multiple_choice_loss
from .class_balanced_model import equal_class_cost,gradient_norm
from .class_coherent_discovery import class_coherent_cost
from .class_coherent_model import ClassCoherentGateSIRM
from .spatial_bank_model import SpatialBankGateSIRM

class RoutedDiscovery:
    def update(self,minibatches):
        assert self.config['discovery_class_weighting']=='equal_class_routed_original_gradient_norm'
        params=list(self.residuals.parameters());observed={};routing=None;reference_blocks=[]
        def loss(g,delta,y,temperature):
            nonlocal routing
            routing=self._cap_forward['pi'][:,:self.num_experts].detach()
            reference=expert_ce(g[:,None]+delta,y)
            original=class_coherent_cost(reference,y)
            for label in y.unique(sorted=True):
                mask=y.eq(label)
                reference_blocks.append({'label':int(label),'count':int(mask.sum()),'mean_expert_CE':reference[mask].mean(0).detach().tolist()})
            routed=equal_class_cost(expert_ce(g[:,None]+routing[:,:,None]*delta,y),y)
            oldnorm=gradient_norm(original,params);newnorm=gradient_norm(routed,params)
            fallback=bool(newnorm==0 and oldnorm>0)
            scale=(oldnorm/newnorm).detach().to(routed.dtype) if newnorm>0 else routed.new_tensor(0.)
            chosen=original if fallback else routed*scale
            matched=gradient_norm(chosen,params);assert torch.allclose(matched,oldnorm,rtol=2e-5,atol=1e-9)
            router_grads=torch.autograd.grad(chosen,list(self.router.parameters()),retain_graph=True,allow_unused=True)
            assert all(v is None for v in router_grads)
            _,balance=multiple_choice_loss(g,delta,y,temperature)
            observed.update(scale=float(scale),fallback=fallback,original_loss=float(original.detach()),routed_loss=float(routed.detach()),original_gradient_norm=float(oldnorm),routed_gradient_norm=float(newnorm),matched_gradient_norm=float(matched))
            return chosen,balance
        def log_losses(logits,y):
            assert routing is not None
            g,delta=self._class_outputs
            return expert_ce(g[:,None]+(delta if observed['fallback'] else routing[:,:,None]*delta),y)
        def replay(per,y):
            return class_coherent_cost(per,y) if observed['fallback'] else equal_class_cost(per,y)*observed['scale']
        with patch('experiments.sirm_research_loop.class_coherent_model.class_coherent_loss',loss),patch('experiments.sirm_research_loop.class_coherent_model.expert_ce',log_losses),patch('experiments.sirm_research_loop.class_coherent_model.class_coherent_cost',replay):
            row=super().update(minibatches)
        self.last_diagnostics.update(discovery_class_weighting=self.config['discovery_class_weighting'],discovery_gradient_norm_match=observed,discovery_reference_class_blocks=reference_blocks,discovery_routing_detached=True,discovery_router_gradient_absent=True,discovery_scope='label_class_batch_routed_individual',discovery_balance_unchanged=True)
        return row

class RoutedDiscoveryGateSIRM(RoutedDiscovery,ClassCoherentGateSIRM):pass
class RoutedDiscoverySpatialGateSIRM(RoutedDiscovery,SpatialBankGateSIRM):pass
