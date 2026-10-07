import copy
import unittest
import torch
from experiments.sirm_h1_h6.test_protocol import toy_cache
from experiments.sirm_h1_h6.controls import fit_router
from experiments.sirm_followup.run import fit_component,predict,evaluate_bank,validate_sources,assignment_diagnostic


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_target_rejected(self):
        a=toy_cache();b=copy.deepcopy(a);b['metadata']['kind']='target';b['metadata']['split']='out'
        with self.assertRaises(ValueError):validate_sources(a,b)

    def test_all_fits_finite_and_fixed_route_calibration_preserves_top1(self):
        a=toy_cache();b=copy.deepcopy(a)
        a['metadata'].update(seed=0,checkpoint_sha256='x',split_hash='s',target_env=0)
        b['metadata']=dict(a['metadata'],split='out');b['image_ids']=[f'val_{i}' for i in range(len(b['labels']))]
        c=dict(router_hidden=8,router_lr=.003,router_epochs=2)
        old={'input_sigma0':fit_router(a,b,c),'global':fit_router(a,b,c,global_only=True)}
        old['fixed_pair']=old['global']
        components={k:fit_component(a,b,old,k,epochs=3,device='cpu') for k in ['ce_top2','ce_dense','class','calibration']}
        result=evaluate_bank(b,old,components)
        self.assertEqual(result['calibrated_top1']['accuracy'],result['kl_top1']['accuracy'])
        for d in result.values():self.assertTrue(torch.isfinite(torch.tensor(d['nll'])))
        # Deployable class routing must not consult labels.
        changed=dict(b,labels=1-b['labels'])
        torch.testing.assert_close(predict(components['class'],b,old),predict(components['class'],changed,old))

    def test_zero_compatibility_is_not_a_gate_pass(self):
        stats={'risk':torch.rand(3,6),'grad':torch.zeros(8,6),'gap':torch.zeros(8,6),'gap_se':torch.ones(8,6)}
        d=assignment_diagnostic({'h6_statistics':stats})
        self.assertFalse(d['Grad']['eligible']);self.assertFalse(d['Regret']['eligible'])


if __name__=='__main__':unittest.main()
