import copy,unittest
import torch
from torch import nn
from experiments.sirm_discover.math import multiple_choice_loss,expert_ce
from .hard_discovery import hard_winner_loss,HardWinnerGateSIRM
from .test_centered import CenteringChecks


class HardDiscoveryChecks(unittest.TestCase):
    def test_unique_winner_only_receives_direct_output_gradient(self):
        g=torch.zeros(2,3,dtype=torch.double);delta=torch.tensor([[[2.,0.,0.],[0.,2.,0.]],[[0.,2.,0.],[2.,0.,0.]]],dtype=torch.double,requires_grad=True);y=torch.tensor([0,0])
        loss,_=hard_winner_loss(g,delta,y,.25);gradient,=torch.autograd.grad(loss,delta)
        self.assertTrue(torch.equal(gradient[0,1],torch.zeros(3,dtype=torch.double)))
        self.assertTrue(torch.equal(gradient[1,0],torch.zeros(3,dtype=torch.double)))
        self.assertGreater(float(gradient[0,0].norm()),0);self.assertGreater(float(gradient[1,1].norm()),0)

    def test_exact_ties_are_symmetric_without_expert0_bias(self):
        g=torch.randn(9,3,dtype=torch.double);delta=torch.zeros(9,6,3,dtype=torch.double,requires_grad=True);y=torch.arange(9)%3
        hard,_=hard_winner_loss(g,delta,y,.25);soft,_=multiple_choice_loss(g,delta,y,.25)
        a,=torch.autograd.grad(hard,delta,retain_graph=True);b,=torch.autograd.grad(soft,delta)
        self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=1e-12))
        self.assertTrue(all(torch.equal(a[:,0],a[:,i]) for i in range(6)))

    def test_balance_responsibility_and_gradient_are_unchanged(self):
        g=torch.randn(13,4,dtype=torch.double);delta=torch.randn(13,6,4,dtype=torch.double,requires_grad=True);y=torch.arange(13)%4
        _,a=hard_winner_loss(g,delta,y,.25);_,b=multiple_choice_loss(g,delta,y,.25)
        self.assertTrue(torch.equal(a,b));ga,=torch.autograd.grad(a.mean(0).square().sum(),delta,retain_graph=True);gb,=torch.autograd.grad(b.mean(0).square().sum(),delta)
        self.assertTrue(torch.equal(ga,gb))

    def model(self,on):
        old=CenteringChecks().model();config=copy.deepcopy(old.config);config.update(discovery_assignment='hard_winner',balance_assignment='original_soft',lambda_sirm=1. if on else 0.)
        torch.manual_seed(918);m=HardWinnerGateSIRM(nn.Linear(4,4),4,3,3,config)
        return m

    def test_actual_update_logs_hard_loss_and_preserves_paired_controls(self):
        a=self.model(False);b=self.model(True);b.load_state_dict(a.state_dict())
        globals_before={k:v.clone() for k,v in a.state_dict().items() if k.startswith(('featurizer.','global_head.'))}
        for step in range(3):
            torch.manual_seed(400+step);batches=[(torch.randn(8,4),torch.arange(8)%3) for _ in range(3)]
            aa=a.update(batches);bb=b.update(batches)
            self.assertEqual(aa,bb)
            self.assertAlmostEqual(aa['loss_discover'],a.last_diagnostics['discovery_hard_loss_replay'],places=7)
            if step==0:
                self.assertEqual(a.last_diagnostics['discovery_minimum_tie_images'],24)
                self.assertTrue(all(abs(v-1/6)<1e-7 for v in a.last_diagnostics['discovery_effective_winner_load']))
            self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        self.assertTrue(all(torch.equal(v,a.state_dict()[k]) for k,v in globals_before.items()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for name in ['featurizer','global_head'] for p in getattr(a,name).parameters()))
        self.assertTrue(all(torch.isfinite(p).all() for p in a.parameters()))

    def test_actual_invariance_update_executes_eligible_sirm(self):
        m=self.model(True);CenteringChecks().nonzero_heads(m);m.step.fill_(251)
        m.q_learned.zero_();m.q_learned[0,0]=m.q_learned[1,0]=1.
        torch.manual_seed(171);batches=[(torch.randn(12,4),torch.arange(12)%3) for _ in range(3)]
        result=m.update(batches)
        self.assertTrue(result['sirm_active']);self.assertGreater(result['loss_sirm'],0.)
        self.assertEqual(result['effective_lambda_sirm'],1.)
        self.assertEqual(m.last_diagnostics['discovery_assignment'],'hard_winner')
        self.assertTrue(all(p.grad is None for name in ['featurizer','global_head'] for p in getattr(m,name).parameters()))


if __name__=='__main__':torch.set_num_threads(2);unittest.main()
