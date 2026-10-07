import unittest,torch
from experiments.sirm_research_loop.shrinkage import quantities
class TestScaling(unittest.TestCase):
    def test_analytic_scale_derivative_and_global_limit(self):
        torch.manual_seed(0);g=torch.randn(8,3,dtype=torch.double);delta=torch.randn(8,2,3,dtype=torch.double);pi=torch.randn(8,3,dtype=torch.double).softmax(1);y=torch.arange(8)%3;domain=torch.arange(8)%2
        result=quantities(g,delta,pi,y,domain,.7);scale=torch.tensor(1.,dtype=torch.double,requires_grad=True)
        for d in range(2):
            for m in range(2):
                term=.7*pi[:,:2,None]*delta;f=g+term.sum(1)+(scale-1)*term[:,m,:];loss=torch.nn.functional.cross_entropy(f[domain==d],y[domain==d]);derivative=torch.autograd.grad(loss,scale,retain_graph=True)[0]
                self.assertAlmostEqual(result['G_matrix'][d][m],float(derivative.abs()),places=12)
        zero=quantities(g,delta,pi,y,domain,0);self.assertEqual(zero['residual_delta'],0);self.assertEqual(zero['correction_RMS'],0);self.assertEqual(zero['utility_matrix'],[[0.,0.],[0.,0.]])
    def test_positive_scale_preserves_directions(self):
        torch.manual_seed(2);g=torch.randn(10,4);delta=torch.randn(10,3,4);pi=torch.randn(10,4).softmax(1);y=torch.arange(10)%4;domains=torch.zeros(10,dtype=torch.long)
        full=quantities(g,delta,pi,y,domains,1.);scaled=quantities(g,delta,pi,y,domains,.5)
        self.assertAlmostEqual(full['residual_cosine_mean'],scaled['residual_cosine_mean'],places=12)
        self.assertAlmostEqual(full['correction_RMS']*.5,scaled['correction_RMS'],places=12)
        self.assertEqual(full['router_NULL'],scaled['router_NULL'])

if __name__=='__main__':unittest.main()
