import unittest
import numpy as np
from experiments.sirm_h2_interaction.run import decomposition, gap, target_means, top2

class AnalysisTests(unittest.TestCase):
    def test_global_quality_does_not_create_interaction(self):
        r=np.array([.1,.3,.5])[:,None]+np.array([.2,.4,.6,.8,1.,1.2])[None,:]
        d=decomposition(r);np.testing.assert_allclose(d['interaction'],0,atol=1e-15)
        selected=top2(r)
        self.assertAlmostEqual(gap(r,selected),gap(d['beta'][None,:],selected))
        self.assertAlmostEqual(gap(d['interaction'],selected),0)

    def test_interaction_reconstruction_and_additive_shifts(self):
        r=np.array([[1.,4.,3.],[4.,1.,3.],[2.,2.,4.]])
        a=decomposition(r);b=decomposition(r+np.array([2.,3.,5.])[:,None]+np.array([7.,8.,9.])[None,:])
        np.testing.assert_allclose(a['interaction'],b['interaction'],atol=1e-14)
        np.testing.assert_allclose(a['mu']+a['alpha'][:,None]+a['beta'][None,:]+a['interaction'],r)
        self.assertAlmostEqual(a['ss_total'],a['ss_domain']+a['ss_expert']+a['ss_interaction'])

    def test_fixed_masks_expose_composition_change(self):
        losses=np.array([1.,2.,4.,8.]);a=np.array([[True,True,False,False],[True,False,True,False]])
        q,n=target_means(losses,a)
        self.assertEqual(q,2.);self.assertEqual(n,5.5)
        self.assertEqual(target_means(losses+1,a),(3.,6.5))

if __name__=='__main__':unittest.main()
