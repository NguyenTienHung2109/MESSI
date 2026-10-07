import unittest
import torch
from domainbed.predictive_compatibility import make_cell,uniform_risk_bound
from domainbed.scripts.check_compatibility_mechanisms import (
    covering_terms,merged_cell,paired_stats,first_feasible_n,fit_constant_router)


class DiagnosticTests(unittest.TestCase):
    def test_bound_decomposition(self):
        for n in [334,10000,1000000]:
            self.assertAlmostEqual(covering_terms(n)['epsilon'],uniform_risk_bound(n,7,16,1.,.05/228),places=12)

    def test_sample_threshold(self):
        self.assertEqual(first_feasible_n(lambda n:1/n,.1),10)

    def test_domain_balancing(self):
        cell=merged_cell([make_cell([[1.]],[0]),make_cell([[-1.],[-1.]],[1,1])])
        self.assertTrue(torch.allclose(cell[2],torch.tensor([.5,.25,.25],dtype=torch.float64)))

    def test_paired_identity(self):
        y=torch.tensor([0,1]*10);z=torch.zeros(20,2)
        self.assertEqual(paired_stats(z,z,y),dict(ce_delta=0.,bootstrap_low=0.,bootstrap_high=0.))

    def test_constant_router_learns_better_expert(self):
        y=torch.tensor([0,1]*10)
        logits=torch.stack([torch.nn.functional.one_hot(y,2).float()*3,
                            torch.nn.functional.one_hot(1-y,2).float()*3],1)
        weights=fit_constant_router(logits,y)
        self.assertGreater(weights[0],weights[1])
        self.assertAlmostEqual(float(weights.sum()),1.,places=6)


if __name__=='__main__':unittest.main()
