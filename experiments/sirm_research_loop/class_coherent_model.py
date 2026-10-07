"""Class-block discovery with the existing centered, capped-router update."""
from unittest.mock import patch
import torch
from experiments.sirm_discover.math import expert_ce, multiple_choice_loss
from .hard_router_cap_model import HardRouterCapGateSIRM
from .class_coherent_discovery import class_coherent_loss, class_coherent_cost


class ClassCoherentGateSIRM(HardRouterCapGateSIRM):
    def forward(self, x):
        result = super().forward(x)
        if self._cap_recording:
            self._class_outputs = (result['global_logits'].detach(), result['delta'].detach())
        return result

    def update(self, minibatches):
        assert self.config['discovery_assignment'] == 'hard_winner'
        assert self.config['discovery_scope'] == 'label_class_batch'
        # The inherited hard-discovery adapter resolves this name when entering
        # its single base update. Retain its optimizer, centering and cap hooks.
        with patch('experiments.sirm_research_loop.hard_discovery.hard_winner_loss', class_coherent_loss):
            row = super().update(minibatches)
        with torch.no_grad():
            g, delta = self._class_outputs
            y = torch.cat([labels for _, labels in minibatches])
            losses = expert_ce(g[:, None, :] + delta, y)
            assignment = torch.zeros_like(losses)
            blocks = []
            for label in y.unique(sorted=True):
                mask = y == label
                means = losses[mask].mean(0)
                ties = means == means.amin()
                assignment[mask] = ties / ties.sum()
                blocks.append(dict(label=int(label), count=int(mask.sum()),
                                   mean_expert_CE=means.tolist(), winners=ties.nonzero().flatten().tolist()))
            _, soft = multiple_choice_loss(g, delta, y, self.config['discovery_temperature'])
            d = self.last_diagnostics
            # Delete inherited image-hard claims; class-block loss is different.
            for key in ['discovery_hard_loss_replay', 'discovery_minimum_tie_images']:
                d.pop(key, None)
            d.update(discovery_assignment='class_coherent_hard', discovery_scope='label_class_batch',
                     discovery_class_loss_replay=float(class_coherent_cost(losses, y)),
                     discovery_class_blocks=blocks,
                     discovery_effective_winner_load=assignment.mean(0).tolist(),
                     discovery_soft_balance_load=soft.mean(0).tolist())
            assert abs(d['discovery_class_loss_replay'] - row['loss_discover']) < 1e-6
        del self._class_outputs
        return row
