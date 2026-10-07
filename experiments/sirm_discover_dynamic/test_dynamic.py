import unittest
import torch
from torch import nn
from experiments.sirm_discover.test_discover import cfg
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.sirm_discover.math import soft_null_q,component_eligibility

class DynamicTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(7);c=cfg();c['q_freeze_stage3']=False
        return GateSIRM(nn.Linear(4,4),4,3,3,c)

    def test_refresh_is_current_detached_Q_after_activation(self):
        m=self.model();ds=torch.arange(3).repeat_interleave(2);m.step.fill_(4)
        u=torch.tensor([[.2,.01,.15],[.18,.19,0.],[0.,.17,.14]],requires_grad=True)
        m.update_utility(u.repeat_interleave(2,0),ds);old=m.q.clone()
        m.step.fill_(5);m.update_utility(u.flip(1).repeat_interleave(2,0),ds)
        expected=soft_null_q(m.utility_ema,m.utility_valid,m.config['q_temperature'],0.,0.)[0]
        torch.testing.assert_close(m.q,expected,rtol=0,atol=0)
        self.assertFalse(torch.equal(old,m.q));self.assertFalse(m.q.requires_grad);self.assertFalse(m.utility_ema.requires_grad);self.assertFalse(m.assignment_locked)

    def test_refresh_only_on_interval_and_Q_is_retained_between(self):
        m=self.model();m.config.update(utility_interval=2,log_sirm_gradient_norm=True)
        torch.manual_seed(11);data=[(torch.randn(4,4),torch.tensor([0,1,2,i])) for i in range(3)]
        for _ in range(4):m.update(data)
        row=m.update(data);self.assertTrue(m.last_diagnostics['q_refreshed']);q=m.q.clone()
        m.update(data);self.assertFalse(m.last_diagnostics['q_refreshed']);torch.testing.assert_close(q,m.q,rtol=0,atol=0)
        m.update(data);self.assertTrue(m.last_diagnostics['q_refreshed'])

    def test_ESS_blocker_distinguished_from_shared_support(self):
        ds=torch.arange(2).repeat_interleave(4);pi=torch.tensor([[1.,0.],[0.,1.],[0.,1.],[0.,1.]]).repeat(2,1);q=torch.ones(2,1)
        valid,ess,count=component_eligibility(pi,ds,q,2.)
        self.assertEqual(count,0);self.assertFalse(valid.any());torch.testing.assert_close(ess,torch.ones_like(ess))

if __name__=='__main__':unittest.main()
