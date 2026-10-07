import copy,json,unittest
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .math import center_logits,entmax15,residual_utilities,multiple_choice_loss,router_teacher,soft_null_q,component_irm
from .model import SIRMDiscover


def cfg():
    c=json.loads(Path(__file__).with_name('config.json').read_text())
    c.update(num_experts=3,residual_hidden=8,global_steps=2,discover_steps=2,invariance_steps=4,sirm_ramp_steps=2,utility_interval=1,utility_min_samples=2,utility_floor=0.,interaction_floor=0.)
    return c


class MathTests(unittest.TestCase):
    def test_entmax_uniform_sparse_translation_and_nearby_scores(self):
        x=torch.zeros(2,4,dtype=torch.double)
        torch.testing.assert_close(entmax15(x),torch.full_like(x,.25))
        x=torch.tensor([[3.,0.,-2.],[.2,.3,-.5]],dtype=torch.double)
        p=entmax15(x);torch.testing.assert_close(p.sum(1),torch.ones(2,dtype=torch.double))
        torch.testing.assert_close(p,entmax15(x+17))
        self.assertEqual(p[0].tolist(),[1.,0.,0.])
        self.assertLess(float((entmax15(x+1e-5*torch.tensor([[1.,0,0],[1.,0,0]]))-p).abs().max()),1e-4)

    def test_center_removes_only_constant_softmax_direction(self):
        z=torch.tensor([[1.,2.,4.],[2.,5.,-1.]],dtype=torch.double)
        torch.testing.assert_close(center_logits(z).mean(1),torch.zeros(2,dtype=torch.double),atol=1e-15,rtol=0)
        torch.testing.assert_close(F.cross_entropy(z,torch.tensor([0,1])),F.cross_entropy(center_logits(z),torch.tensor([0,1])))

    def test_marginal_utility_exact_removal_no_renormalization(self):
        g=torch.tensor([[.2,-.1]]);delta=center_logits(torch.tensor([[[1.,-1.],[-.5,.5]]]))
        pi=torch.tensor([[.3,.2,.5]]);y=torch.tensor([0])
        full,c,marginal,single=residual_utilities(g,delta,pi,y)
        torch.testing.assert_close(full,g+.3*delta[:,0]+.2*delta[:,1])
        expected=F.cross_entropy(g+.2*delta[:,1],y)-F.cross_entropy(full,y)
        torch.testing.assert_close(marginal[0,0],expected)
        self.assertLess(float(marginal[0,1]),0.)

    def test_discovery_prefers_good_correction_without_all_expert_ce(self):
        g=torch.zeros(1,2);delta=torch.tensor([[[2.,-2.],[-2.,2.]]],requires_grad=True)
        loss,choice=multiple_choice_loss(g,delta,torch.tensor([0]),.1)
        loss.backward();self.assertGreater(float(choice[0,0]),.99)
        self.assertGreater(float(delta.grad[:,0].abs().sum()),100*float(delta.grad[:,1].abs().sum()))

    def test_no_interaction_or_nonpositive_utility_chooses_null(self):
        u=torch.tensor([[.1,.3,.2],[.1,.3,.2],[.1,.3,.2]])
        q,null,confidence,interaction=soft_null_q(u,torch.ones_like(u,dtype=torch.bool))
        torch.testing.assert_close(q,torch.zeros_like(q));torch.testing.assert_close(null,torch.ones(3))
        u=torch.tensor([[-.01,-.5,-.4],[-.4,-.01,-.5],[-.5,-.4,-.01]])
        q,null,_,_=soft_null_q(u,torch.ones_like(u,dtype=torch.bool))
        self.assertEqual(float(q.sum()),0.);self.assertEqual(null.tolist(),[1.,1.,1.])

    def test_pair_shared_utility_q_and_missing_statistics(self):
        u=torch.tensor([[.20,.01,.15],[.18,.19,0.],[0.,.17,.14]],requires_grad=True)
        valid=torch.ones_like(u,dtype=torch.bool);q,null,confidence,_=soft_null_q(u,valid,utility_floor=0.,interaction_floor=0.)
        self.assertGreater(float(q[0,0]),0.);self.assertGreater(float(q[1,0]),0.)
        self.assertEqual(float(q[2,0]),0.)
        torch.testing.assert_close(q.sum(1)+null,torch.ones(3));self.assertFalse(q.requires_grad)
        valid[0,0]=False;q,null,_,_=soft_null_q(u,valid)
        self.assertEqual(float(q.sum()),0.);self.assertEqual(null.tolist(),[1.,1.,1.])

    def test_router_teacher_null_and_stop_gradient(self):
        u=torch.tensor([[-.1,-.2],[2.,-.1]],requires_grad=True)
        teacher=router_teacher(u,.1)
        self.assertEqual(teacher[0].tolist(),[0.,0.,1.]);self.assertEqual(teacher[1].tolist(),[1.,0.,0.])
        self.assertFalse(teacher.requires_grad)

    def test_q_effect_floor_is_continuous(self):
        u=torch.tensor([[.20,.01,.15],[.18,.19,0.],[0.,.17,.14]],dtype=torch.double)
        valid=torch.ones_like(u,dtype=torch.bool)
        a=soft_null_q(u,valid,utility_floor=.18)[0]
        changed=u.clone();changed[1,0]+=1e-7
        b=soft_null_q(changed,valid,utility_floor=.18)[0]
        self.assertLess(float((a-b).abs().max()),1e-5)

    def test_single_domain_membership_does_not_activate_subset_irm(self):
        delta=torch.randn(8,2,3,requires_grad=True)
        q=torch.tensor([[1.,0.],[0.,1.]])
        loss,info=component_irm(torch.zeros(8,3),delta,torch.full((8,3),1/3),torch.zeros(8,dtype=torch.long),torch.arange(2).repeat_interleave(4),q)
        self.assertFalse(info['active']);self.assertEqual(float(loss),0.)
        loss.backward();torch.testing.assert_close(delta.grad,torch.zeros_like(delta))

    def test_component_scale_gradient_and_gradient_isolation(self):
        torch.manual_seed(4)
        g=torch.randn(8,3,requires_grad=True);delta=torch.randn(8,2,3,requires_grad=True)
        pi=torch.full((8,3),1/3,requires_grad=True);y=torch.tensor([0,1,2,0,1,2,0,1]);domains=torch.arange(2).repeat_interleave(4)
        q=torch.full((2,2),.5,requires_grad=True)
        loss,info=component_irm(g,delta,pi,y,domains,q)
        self.assertTrue(info['active']);loss.backward(retain_graph=True)
        self.assertGreater(float(delta.grad.abs().sum()),0.)
        self.assertIsNone(g.grad);self.assertIsNone(pi.grad);self.assertIsNone(q.grad)
        term=pi.detach()[:,:2,None]*delta.detach();full=g.detach()+term.sum(1)
        for d in range(2):
            mask=domains==d
            for m in range(2):
                analytic=((full[mask].softmax(1)-F.one_hot(y[mask],3))*term[mask,m]).sum(1).mean()
                self.assertAlmostEqual(info['G'][f'{d},{m}'],float(analytic),places=6)
        half,_=component_irm(g,delta,pi,y,domains,q*.5)
        torch.testing.assert_close(half,loss*.25)
        null,info=component_irm(g,delta,pi,y,domains,q*0)
        self.assertEqual(float(null),0.);self.assertFalse(info['active'])


class StageTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(44)
        return SIRMDiscover(nn.Sequential(nn.Linear(4,4),nn.Dropout(.1)),4,3,3,cfg())

    def data(self):
        torch.manual_seed(81)
        return [(torch.randn(4,4),torch.tensor([0,1,2,i])) for i in range(3)]

    def test_stage_freezing_zero_residual_and_ramp(self):
        m=self.model();data=self.data();before=[p.clone() for p in m.residuals.parameters()]
        a=m.update(data);self.assertEqual(a['stage'],'global');self.assertEqual(a['effective_lambda_sirm'],0.)
        for p,q in zip(m.residuals.parameters(),before):torch.testing.assert_close(p,q)
        m.update(data);m.configure_stage();m.eval();f=m(data[0][0]);torch.testing.assert_close(f['logits'],f['global_logits'])
        anchor=[p.clone() for p in list(m.featurizer.parameters())+list(m.global_head.parameters())]
        for expected_stage,expected_lambda in [('discover',0.),('discover',0.),('invariance',0.),('invariance',.05),('invariance',.1)]:
            r=m.update(data);self.assertEqual(r['stage'],expected_stage);self.assertAlmostEqual(r['effective_lambda_sirm'],expected_lambda)
            self.assertFalse(m.featurizer.training)
        for p,q in zip(list(m.featurizer.parameters())+list(m.global_head.parameters()),anchor):torch.testing.assert_close(p,q)
        self.assertEqual(float(m(data[0][0])['delta'].mean(-1).abs().max())<1e-6,True)

    def test_state_optimizer_roundtrip_preserves_stage_and_next_update(self):
        m=self.model();data=self.data()
        for _ in range(5):m.update(data)
        weights=copy.deepcopy(m.state_dict());optimizer=copy.deepcopy(m.optimizer.state_dict())
        restored=self.model();restored.load_state_dict(weights);restored.configure_stage();restored.optimizer.load_state_dict(optimizer)
        a=m.update(data);b=restored.update(data)
        self.assertEqual(a,b)
        for k,v in m.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
