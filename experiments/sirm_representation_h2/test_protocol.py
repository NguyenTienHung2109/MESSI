import random,unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from experiments.sirm_h1_h6 import run as base
from .protocol import ViewTransform,config_for,assert_only_sirm_changed,validate_source,branch_weights
from .diagnostics import cosine_margin,router_pair
from .summarize import EFFECTS,paired_bootstrap

class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_off_changes_only_invariance_coefficient(self):
        on,off=config_for(True),config_for(False);assert_only_sirm_changed(on,off)
        self.assertEqual(off['hparams']['subset_irm_lambda_route'],.1)
        self.assertEqual(off['hparams']['subset_irm_lambda_expert'],.1)
        bad=config_for(False);bad['hparams']['num_experts']=2
        with self.assertRaises(ValueError):assert_only_sirm_changed(on,bad)

    def test_baseline_unchanged_and_eval_deterministic(self):
        cfg=config_for(True);image=Image.fromarray(np.random.default_rng(4).integers(0,256,(250,250,3),dtype=np.uint8))
        new=ViewTransform(cfg,train=True);old=base.ViewTransform(cfg,train=True)
        random.seed(5);torch.manual_seed(5);a=new(image)
        random.seed(5);torch.manual_seed(5);b=old(image)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual([type(t) for t in new.prepare.transforms],[T.RandomResizedCrop,T.RandomHorizontalFlip,T.ColorJitter,T.RandomGrayscale])
        self.assertEqual(new.prepare.transforms[-1].p,.1)
        for view in ['original','grayscale','blur','gray_blur']:
            transform=ViewTransform(cfg,view=view,train=False,cohort='balanced_views')
            a,b=transform(image),transform(image);self.assertEqual(tuple(a.shape),(3,224,224));torch.testing.assert_close(a,b,rtol=0,atol=0)
        with self.assertRaises(ValueError):ViewTransform(cfg,view='gray_blur',train=True)(image)

    def test_h2_branch_frequencies_and_shapes(self):
        cfg=config_for(True);transform=ViewTransform(cfg,train=True,cohort='balanced_views')
        self.assertEqual(transform.prepare.transforms[-1].p,.1)
        transform.prepare=T.Resize((224,224));image=Image.new('RGB',(224,224),(90,140,200))
        original_choice=random.choice;counts=dict(original=0,grayscale=0,blur=0)
        def choice(options):
            result=original_choice(options);counts[result]+=1;return result
        random.seed(901)
        with patch('experiments.sirm_h1_h6.run.random.choice',side_effect=choice):
            for _ in range(1500):self.assertEqual(tuple(transform(image).shape),(3,224,224))
        for count in counts.values():self.assertLess(abs(count/1500-1/3),.04)

    def test_global_source_only_and_normalized(self):
        w=branch_weights(dict(kind='global',state=torch.tensor([2.,-1.,0.,3.,1.,-.5])))
        self.assertAlmostEqual(float(w.sum()),1.,places=6)
        source=dict(metadata=dict(kind='source'),physical_domains=torch.tensor([1,2,3]));validate_source(source,0)
        with self.assertRaises(ValueError):validate_source(dict(metadata=dict(kind='target'),physical_domains=torch.tensor([0])),0)
        with self.assertRaises(ValueError):validate_source(dict(metadata=dict(kind='source'),physical_domains=torch.tensor([0,1])),0)

    def test_collapse_and_paired_routing_controls(self):
        x=torch.ones(12,5);y=torch.arange(12)%3
        self.assertAlmostEqual(cosine_margin(x,y)['cosine_margin'],0.,places=6)
        p=torch.tensor([[.5,.5,0.,0.],[0.,0.,.5,.5]])
        pair=router_pair(dict(router_dense=p),dict(router_dense=p),2)
        self.assertAlmostEqual(pair['router_js'],0.);self.assertAlmostEqual(pair['topk_jaccard'],1.)

    def test_factorial_interaction_and_paired_bootstrap(self):
        a,b,c,d=.7,.75,.72,.8;co=EFFECTS['interaction']
        self.assertAlmostEqual(sum(x*y for x,y in zip(co,[a,b,c,d])),.03)
        ci=paired_bootstrap([.03]*12,[(t,s) for t in range(4) for s in range(3)])
        self.assertAlmostEqual(ci['ci95_cell_low'],.03);self.assertAlmostEqual(ci['ci95_seed_block_high'],.03)

if __name__=='__main__':unittest.main()
