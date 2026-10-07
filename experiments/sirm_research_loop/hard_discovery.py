"""Explicit hard-winner discovery candidate; centering and balance unchanged."""
from unittest.mock import patch
import torch
from experiments.sirm_discover.math import expert_ce,multiple_choice_loss
from .centered_model import CenteredGateSIRM


def hard_winner_loss(global_logits,delta,y,temperature):
    # Preserve the original differentiable soft responsibilities for L_balance.
    _,balance_choice=multiple_choice_loss(global_logits,delta,y,temperature)
    losses=expert_ce(global_logits[:,None,:]+delta,y)
    # amin shares derivatives across exact ties, unlike index-selecting min.
    # Zero-initialized experts therefore receive symmetric first-step gradients.
    return losses.amin(1).mean(),balance_choice


class HardWinnerGateSIRM(CenteredGateSIRM):
    def forward(self,x):
        result=super().forward(x)
        if self.training and self.stage()!='global' and self.config.get('discovery_assignment')=='hard_winner':
            self._assignment_outputs=(result['global_logits'].detach(),result['delta'].detach())
        return result

    def update(self,minibatches):
        if self.config.get('discovery_assignment')!='hard_winner':
            return super().update(minibatches)
        assert self.config.get('balance_assignment')=='original_soft'
        with patch('experiments.sirm_discover.model.multiple_choice_loss',hard_winner_loss):
            result=super().update(minibatches)
        with torch.no_grad():
            g,delta=self._assignment_outputs;y=torch.cat([b for a,b in minibatches])
            losses=expert_ce(g[:,None,:]+delta,y);best=losses.amin(1)
            tied=losses.eq(best[:,None]);assignment=tied/tied.sum(1,keepdim=True)
            _,soft=multiple_choice_loss(g,delta,y,self.config['discovery_temperature'])
            self.last_diagnostics.update(discovery_assignment='hard_winner',balance_assignment='original_soft',
                discovery_hard_loss_replay=float(best.mean()),discovery_minimum_tie_images=int(tied.sum(1).gt(1).sum()),
                discovery_effective_winner_load=assignment.mean(0).tolist(),
                discovery_soft_balance_load=soft.mean(0).tolist(),
                discovery_soft_entropy_normalized=float(-(soft*soft.clamp_min(1e-30).log()).sum(1).mean()/torch.log(soft.new_tensor(float(self.num_experts)))))
        del self._assignment_outputs
        return result
