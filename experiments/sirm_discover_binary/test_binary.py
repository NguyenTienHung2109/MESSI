import unittest
import torch
from experiments.sirm_discover.math import component_irm

class BinaryTests(unittest.TestCase):
    def data(self):
        torch.manual_seed(19)
        return torch.randn(12,3),torch.randn(12,2,3,requires_grad=True),torch.full((12,3),1/3),torch.arange(12)%3,torch.arange(3).repeat_interleave(4)

    def test_same_eligibility_and_active_mean_without_attenuation(self):
        g,d,pi,y,ds=self.data();q=torch.tensor([[1e-5,0.],[2e-5,3e-5],[0.,4e-5]],requires_grad=True)
        binary,info=component_irm(g,d,pi,y,ds,q,weighting='binary')
        confidence,old=component_irm(g,d,pi,y,ds,q)
        self.assertEqual(info['pair_count'],2);self.assertEqual(info['pair_count'],old['pair_count'])
        G=info['G'];expected=(G['0,0']**2+G['1,0']**2+G['1,1']**2+G['2,1']**2)/2
        self.assertAlmostEqual(float(binary),expected,places=6)
        unchanged,_=component_irm(g,d,pi,y,ds,q*1e-4,weighting='binary')
        torch.testing.assert_close(binary,unchanged,rtol=0,atol=0)
        self.assertGreater(float(binary),1e7*float(confidence))
        binary.backward();self.assertIsNone(q.grad);self.assertGreater(float(d.grad.abs().sum()),0.)

    def test_no_eligible_pairs_remains_zero(self):
        g,d,pi,y,ds=self.data();q=torch.tensor([[1.,0.],[0.,1.],[0.,0.]])
        loss,info=component_irm(g,d,pi,y,ds,q,weighting='binary')
        self.assertEqual(float(loss),0.);self.assertFalse(info['active']);loss.backward();self.assertEqual(float(d.grad.abs().sum()),0.)

if __name__=='__main__':unittest.main()
