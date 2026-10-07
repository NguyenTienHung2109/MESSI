"""Dynamic detached source Q; explicit freeze option for historical controls."""
import torch
from experiments.sirm_discover.model import SIRMDiscover

class GateSIRM(SIRMDiscover):
    def __init__(self,*args,policy='learned',assignment_seed=0,**kwargs):
        super().__init__(*args,**kwargs);self.policy=policy
        self.freeze_stage3=self.config.get('q_freeze_stage3',False)
        gen=torch.Generator().manual_seed(704219+assignment_seed)
        permutation=torch.randperm(self.num_experts,generator=gen)
        while (permutation==torch.arange(self.num_experts)).any():permutation=torch.randperm(self.num_experts,generator=gen)
        self.register_buffer('assignment_permutation',permutation)
        self.register_buffer('q_learned',torch.zeros_like(self.q))
        self.register_buffer('q_locked',torch.zeros_like(self.q))
        self.register_buffer('null_locked',torch.ones_like(self.q_null))
        self.register_buffer('confidence_locked',torch.zeros_like(self.q_confidence))
        self.register_buffer('assignment_locked',torch.tensor(False))
    @torch.no_grad()
    def apply_policy(self):
        if self.freeze_stage3 and self.assignment_locked:
            self.q_learned.copy_(self.q_locked);self.q_null.copy_(self.null_locked);self.q_confidence.copy_(self.confidence_locked)
        if self.stage()!='invariance' or self.policy=='learned':self.q.copy_(self.q_learned)
        elif self.policy=='random':self.q.copy_(self.q_learned[:,self.assignment_permutation])
        else:raise ValueError('unknown Q policy')
    @torch.no_grad()
    def update_utility(self,marginal,domains):
        super().update_utility(marginal,domains)
        if self.freeze_stage3 and self.stage()=='invariance' and not self.assignment_locked:
            self.q_locked.copy_(self.q);self.null_locked.copy_(self.q_null);self.confidence_locked.copy_(self.q_confidence);self.assignment_locked.fill_(True)
        if not self.freeze_stage3 or not self.assignment_locked:self.q_learned.copy_(self.q)
        self.apply_policy()
    def update(self,minibatches):
        self.apply_policy();values=super().update(minibatches)
        self.last_diagnostics.update(q_learned=self.q_learned.tolist(),q_policy=self.policy,assignment_permutation=self.assignment_permutation.tolist(),assignment_locked=bool(self.assignment_locked))
        return values
