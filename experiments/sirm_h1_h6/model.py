"""Standalone protocol adaptation; legacy SubsetIRM code remains untouched."""
import torch
import torch.nn.functional as F
from domainbed.subset_irm import MESSISubsetIRM


class HypothesesSIRM(MESSISubsetIRM):
    def _subset_forward(self, x):
        z = self.featurizer(x)
        z_moe = self.moe_head.input_proj(z)
        features = torch.stack([expert(z_moe) for expert in self.moe_head.experts], 1)
        dense = F.softmax(self.moe_head.router(z_moe), dim=-1)
        responsibilities = self._routing_responsibilities(dense)
        shared = self.moe_head.classifier((dense.unsqueeze(-1) * features).sum(1))
        expert_logits = self.expert_heads(features)
        logits = (responsibilities.unsqueeze(-1) * expert_logits).sum(1)
        # Only auxiliary router loss gets this branch. Main CE retains its graph.
        auxiliary_dense = F.softmax(self.moe_head.router(z_moe.detach()), dim=-1)
        return logits, auxiliary_dense, responsibilities, features, expert_logits, shared

    def _assignment(self, dense_pi, domain_ids):
        if self.assignment_mode == 'hypotheses_external':
            return self.hypotheses_q.detach()
        return super()._assignment(dense_pi, domain_ids)

    def set_assignment(self, q):
        if not hasattr(self, 'hypotheses_q'):
            self.register_buffer('hypotheses_q', torch.zeros_like(q, device=next(self.parameters()).device))
        self.hypotheses_q.copy_(q.to(self.hypotheses_q))
        self.assignment_mode = 'hypotheses_external'
