import math
import unittest
import numpy as np
import torch
from types import SimpleNamespace
from domainbed.transfer_compatibility import (
    bernstein_upper,entropy_box_lower,ce_interval,hull_audit,ridge_head,train_transfer_map)


class TransferCompatibilityTests(unittest.TestCase):
    def test_zero_direction_exact(self):
        self.assertEqual(bernstein_upper(np.zeros(20),0,0,.01)['upper'],0)
        with self.assertRaises(ValueError):bernstein_upper([0,2],0,1,.01)

    def test_bernstein_translation_and_scale(self):
        x=np.linspace(.1,.8,30)
        a=bernstein_upper(x,0,1,.1)
        b=bernstein_upper(3*x-2,-2,1,.1)
        self.assertAlmostEqual(b['upper'],3*a['upper']-2)

    def test_entropy_box_binary_exact_vertices(self):
        r=entropy_box_lower([100,100],.01)
        low=max(r['class_lower'][0],1-r['class_upper'][1])
        high=min(r['class_upper'][0],1-r['class_lower'][1])
        h=lambda p:-p*math.log(p)-(1-p)*math.log1p(-p)
        self.assertAlmostEqual(r['lower'],min(h(low),h(high)))
        self.assertLess(r['lower'],math.log(2))

    def test_ce_interval_and_direction_range(self):
        torch.manual_seed(0)
        w=torch.randn(3,4,dtype=torch.float64)
        z=torch.nn.functional.normalize(torch.randn(200,4,dtype=torch.float64),dim=1)
        y=torch.arange(200)%3
        ce=torch.nn.functional.cross_entropy(z@w.T,y,reduction='none')
        lo,hi=ce_interval(w)
        self.assertGreaterEqual(float(ce.min()),lo)
        self.assertLessEqual(float(ce.max()),hi)
        other=torch.randn_like(w)
        # Exercises range validation on each actual directional observation.
        hull_audit(w,torch.stack([w,other]),z,y,.12,.05,.01)

    def test_compatible_signal_and_incompatible_head(self):
        rng=np.random.default_rng(0);n=5000
        y=torch.from_numpy(rng.integers(0,2,n))
        sign=2*y.numpy()-1
        z=torch.from_numpy((sign*np.where(rng.random(n)<.8,1,-1))[:,None]).double()
        w=torch.tensor([[-math.log(4)/2],[math.log(4)/2]],dtype=torch.float64)
        vertices=torch.stack([w,-w,torch.zeros_like(w)])
        good=hull_audit(w,vertices,z,y,.12,.05,.001)
        self.assertTrue(good['accepted'])
        bad=hull_audit(torch.zeros_like(w),vertices,z,y,.12,.05,.001)
        self.assertFalse(bad['compatible'])
        constant=hull_audit(w,vertices,torch.zeros_like(z),y,.12,.05,.001)
        self.assertFalse(constant['predictive'])

    def test_hull_must_contain_deployed_head(self):
        with self.assertRaises(ValueError):
            hull_audit(torch.ones(2,1),torch.zeros(1,2,1),torch.zeros(10,1),torch.zeros(10,dtype=torch.long),.12,.05,.01)

    def test_ridge_stationarity_and_gradient(self):
        torch.manual_seed(0)
        z=torch.randn(7,3,dtype=torch.float64,requires_grad=True)
        target=torch.randn(7,2,dtype=torch.float64)
        a=ridge_head(z,target,.1)
        residual=z.T@(z@a-target)/len(z)+.1*a
        self.assertLess(float(residual.detach().norm()),1e-12)
        self.assertTrue(torch.autograd.gradcheck(lambda x:ridge_head(x,target,.1),(z,)))

    def test_tangent_gap_bounds_hull_risk_improvement(self):
        torch.manual_seed(3)
        z=torch.nn.functional.normalize(torch.randn(80,4,dtype=torch.float64),dim=1)
        y=torch.arange(80)%3
        vertices=torch.randn(3,3,4,dtype=torch.float64);w=vertices[0]
        residual=(z@w.T).softmax(1)-torch.nn.functional.one_hot(y,3)
        gap=max(float((residual*(z@(w-v).T)).sum(1).mean()) for v in vertices)
        base=torch.nn.functional.cross_entropy(z@w.T,y)
        for _ in range(30):
            alpha=torch.randn(3,dtype=torch.float64).softmax(0)
            v=(alpha[:,None,None]*vertices).sum(0)
            gain=float(base-torch.nn.functional.cross_entropy(z@v.T,y))
            self.assertLessEqual(gain,gap+1e-12)

    def test_episode_domain_exclusion(self):
        torch.manual_seed(0)
        bank={d:(torch.randn(120,4),torch.arange(120)%2) for d in [0,1]}
        splits={d:{'train':np.arange(100),'audit':np.arange(100,120)} for d in bank}
        args=SimpleNamespace(dimension=3,hidden=5,learning_rate=.001,map_steps=2,
            batch_size=8,radius=1.,ridge_lambda=.01,transfer_weight=1.)
        for mode in ['cross_domain','within_domain']:
            _,trace=train_transfer_map(bank,splits,[0,1],args,2,0,mode=mode)
            for row in trace:
                if mode=='cross_domain':self.assertNotIn(row['query_domain'],row['inner_domains'])
                else:self.assertEqual(row['inner_domains'],[row['query_domain']])
                self.assertTrue(math.isfinite(row['query_squared_error']))


if __name__=='__main__':unittest.main()
