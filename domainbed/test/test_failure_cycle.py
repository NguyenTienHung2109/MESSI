import unittest
import numpy as np
import torch
from domainbed.scripts.analyze_compatibility_failures import hierarchical_scatter,fit_scale
from domainbed.scripts.check_deit_token_features import assemble


class FailureCycleTests(unittest.TestCase):
    def test_matched_calibration_allocation(self):
        from domainbed.scripts.check_global_bank_control import allocation
        for total in [0,1,7,23]:
            counts=allocation(total,[3,7,13])
            self.assertEqual(counts.sum(),total)
            self.assertTrue((counts<=np.array([3,7,13])).all())
        with self.assertRaises(ValueError):allocation(24,[3,7,13])

    def test_pretrained_maps_reject_wrong_token_mode(self):
        from domainbed.scripts.run_transfer_compatibility import run_fold
        with self.assertRaises(ValueError):
            run_fold(None,None,0,'main','erm',None,.01,feature_mode='dist')

    def test_variance_identity_with_unequal_domain_sizes(self):
        rng=np.random.default_rng(0)
        z=rng.normal(size=(31,5));y=np.arange(31)%3;d=np.r_[np.zeros(11),np.ones(20)]
        r=hierarchical_scatter(z,y,d)
        self.assertAlmostEqual(r['total'],r['between_class']+r['within_class_domain']+r['within_class_domain_residual'])

    def test_pure_class_signal(self):
        y=np.tile([0,1],20);z=np.eye(2)[y];d=np.tile([0,0,1,1],10)
        r=hierarchical_scatter(z,y,d)
        self.assertAlmostEqual(r['between_class_fraction'],1.)
        self.assertAlmostEqual(r['domain_given_class_fraction'],0.)

    def test_temperature_preserves_predictions_and_reduces_fit_ce(self):
        logits=torch.tensor([[.1,0.],[0.,.1],[.1,0.],[0.,.1]],dtype=torch.float64)
        y=torch.tensor([0,1,0,1]);beta=fit_scale([(logits,y)])
        self.assertTrue(torch.equal(logits.argmax(1),(beta*logits).argmax(1)))
        self.assertLess(float(torch.nn.functional.cross_entropy(beta*logits,y)),float(torch.nn.functional.cross_entropy(logits,y)))

    def test_token_capacity_control(self):
        cls=torch.randn(4,3);dist=torch.randn(4,3)
        self.assertEqual(assemble(cls,dist,'repeat_cls').shape,assemble(cls,dist,'concat_tokens').shape)
        self.assertTrue(torch.equal(assemble(cls,dist,'repeat_cls')[:,:3],cls))
        self.assertTrue(torch.equal(assemble(cls,dist,'mean_tokens'),(cls+dist)/2))


if __name__=='__main__':unittest.main()
