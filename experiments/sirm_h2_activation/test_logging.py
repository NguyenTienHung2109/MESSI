"""Verify diagnostics compute nonzero counterfactual gradients without changing CE updates."""
import copy,types,unittest
import torch
import torch.nn.functional as F
from .model import DiagnosticSIRM

class LoggingTest(unittest.TestCase):
    def test_counterfactual_does_not_change_update(self):
        torch.manual_seed(73)
        model=DiagnosticSIRM.__new__(DiagnosticSIRM);torch.nn.Module.__init__(model)
        model.featurizer=torch.nn.Linear(4,4);model.moe_head=torch.nn.Module()
        model.moe_head.router=torch.nn.Linear(4,2);model.moe_head.experts=torch.nn.ModuleList([torch.nn.Linear(4,4) for _ in range(2)])
        model.expert_heads=torch.nn.ModuleList([torch.nn.Linear(4,3) for _ in range(2)])
        model.min_effective_samples=2;model.responsibility_detach=True
        model._assignment=types.MethodType(lambda self,dense,domain:torch.full((3,2),.5),model)
        reference=copy.deepcopy(model);x=torch.randn(24,4);y=torch.arange(24)%3;domain=torch.arange(24)//8
        def forward(m):
            z=m.featurizer(x);gamma=m.moe_head.router(z).softmax(1)
            features=torch.stack([expert(z) for expert in m.moe_head.experts],1)
            expert_logits=torch.stack([m.expert_heads[i](features[:,i]) for i in range(2)],1)
            logits=(gamma[:,:,None]*expert_logits).sum(1)
            m._diagnostic_forward=(logits,gamma,gamma,features,expert_logits,None)
            m._diagnostic_labels=y;m._diagnostic_domains=domain
            return F.cross_entropy(logits,y)
        loss=forward(model);metrics=model._gradient_diagnostics({'cls':loss},'grad_raw')
        self.assertGreater(metrics['diagnostic_counterfactual_sirm_norm_backbone'],0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        loss.backward();forward(reference).backward()
        for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)

if __name__=='__main__':unittest.main()
