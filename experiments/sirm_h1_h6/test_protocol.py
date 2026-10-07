"""Numerical and gradient contracts for the isolated H1-H6 pipeline."""
import unittest
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.subset_irm import ExpertClassifierHeads, subset_irm_penalty, router_distillation_loss
from domainbed.lib import misc
from domainbed.scripts.train_subset_irm_pacs import split_keys
from experiments.sirm_h1_h6.model import HypothesesSIRM
from experiments.sirm_h1_h6.controls import metrics, shuffle_routes, fit_router, mixture, route, frozen_controls
from experiments.sirm_h1_h6.statistics import (feasible_supports, choose_assignment, classifier_gradient_gaps,
                                              fit_gap_probes, evaluate_gaps)
from experiments.sirm_h1_h6.run import auxiliary_partition, ScheduledImages, ScheduledBatches, capture_rng, restore_rng, ViewTransform


def toy_cache():
    torch.manual_seed(21)
    y = torch.tensor([0,0,0,0,1,1,1,1]*3)
    domains = torch.arange(3).repeat_interleave(8)
    features = torch.randn(len(y),6,3)
    features[:,:,0] += y[:,None]*2-1
    logits = torch.stack([features[:,:,0],-features[:,:,0]],-1)
    return dict(labels=y,domains=domains,expert_features=features,expert_logits=logits,
                projected_feature=torch.randn(len(y),4),image_ids=[f'i{i}' for i in range(len(y))],
                parent_ids=[f'i{i}' for i in range(len(y))], metadata=dict(kind='source',split='in'))


class ProtocolTests(unittest.TestCase):
    def test_domainbed_split_identical(self):
        for seed in range(5):
            for env in range(4):
                dataset=list(range(101))
                out_,in_=misc.split_dataset(dataset,20,misc.seed_hash(seed,env))
                in_ids,out_ids=split_keys(101,.2,seed,env)
                self.assertEqual(out_.keys,out_ids)
                self.assertEqual(in_.keys,in_ids)
                self.assertFalse(set(in_ids)&set(out_ids))

    def test_matched_support_feasibility_and_determinism(self):
        supports=feasible_supports()
        self.assertTrue((supports.sum(2)==2).all())
        count=supports.sum(1)
        self.assertTrue(((count==0)|(count>=2)).all())
        self.assertTrue((count.gt(0).sum(1)<=3).all())
        risk=torch.rand(3,6,requires_grad=True)
        gap=torch.rand(8,6,requires_grad=True)
        for criterion in ['Random','Risk','Grad','Regret','FixedPairs']:
            q=choose_assignment(risk,gap,criterion,seed=3)
            self.assertFalse(q.requires_grad)
            self.assertTrue((q.gt(0).sum(1)==2).all())
            active=q.gt(0).sum(0)
            self.assertTrue(((active==0)|(active>=2)).all())
            torch.testing.assert_close(q,choose_assignment(risk,gap,criterion,seed=3))

    def test_metrics_hand_calculation(self):
        logits=torch.tensor([[[2.,0.],[0.,2.]],[[0.,2.],[2.,0.]]])
        y=torch.tensor([0,1]);pi=torch.tensor([[.75,.25],[.75,.25]])
        got=metrics(dict(expert_logits=logits,labels=y),pi)
        self.assertAlmostEqual(got['accuracy'],1.)
        self.assertAlmostEqual(got['nll'],F.softplus(torch.tensor(-1.)).item(),places=6)
        self.assertAlmostEqual(got['routing_regret'],.5,places=6)
        self.assertAlmostEqual(got['selection_loss'],F.softplus(torch.tensor(-2.)).item()+.5,places=6)
        self.assertNotAlmostEqual(got['selection_loss'],got['nll'])

    def test_shuffle_keeps_outputs_and_strata(self):
        cache=toy_cache();before=cache['expert_logits'].clone()
        pi=torch.softmax(torch.randn(24,6),1)
        shuffled=shuffle_routes(pi,cache,True,4)
        torch.testing.assert_close(cache['expert_logits'],before)
        for d in range(3):
            for c in range(2):
                mask=(cache['domains']==d)&(cache['labels']==c)
                torch.testing.assert_close(pi[mask].sort(0).values,shuffled[mask].sort(0).values)

    def test_router_gradient_isolated_main_ce_and_sirm_retained(self):
        # Use the real protocol forward and penalty with a small bank.
        model=HypothesesSIRM.__new__(HypothesesSIRM)
        nn.Module.__init__(model)
        model.featurizer=nn.Linear(3,4)
        model.moe_head=nn.Module()
        model.moe_head.input_proj=nn.Linear(4,4)
        model.moe_head.experts=nn.ModuleList([nn.Linear(4,4) for _ in range(6)])
        model.moe_head.router=nn.Linear(4,6)
        model.moe_head.classifier=nn.Linear(4,2)
        model.expert_heads=ExpertClassifierHeads(6,4,2)
        model.router_topk=None;model.topk_warmup_steps=0
        model.register_buffer('subset_step',torch.tensor(0))
        x=torch.randn(18,3);y=torch.arange(18)%2;domains=torch.arange(3).repeat_interleave(6)
        out,dense,gamma,_,expert,_=model._subset_forward(x)
        q=choose_assignment(torch.ones(3,6),torch.zeros(8,6),'FixedPairs')
        auxiliary=router_distillation_loss(dense,domains,q)
        parameters=list(model.featurizer.parameters())+list(model.moe_head.input_proj.parameters())
        grads=torch.autograd.grad(auxiliary,parameters,allow_unused=True,retain_graph=True)
        self.assertTrue(all(g is None or g.abs().sum()==0 for g in grads))
        self.assertGreater(torch.autograd.grad(auxiliary,model.moe_head.router.weight,retain_graph=True)[0].norm(),0)
        ce=F.cross_entropy(out,y)
        for parameter in [model.featurizer.weight,model.moe_head.experts[0].weight,model.moe_head.router.weight]:
            self.assertGreater(torch.autograd.grad(ce,parameter,retain_graph=True)[0].norm(),0)
        penalty,info=subset_irm_penalty(expert,y,domains,gamma,q,min_effective_samples=1)
        self.assertTrue(info['active'])
        for parameter in [model.featurizer.weight,model.moe_head.input_proj.weight,model.moe_head.experts[0].weight,model.expert_heads.heads[0].weight]:
            self.assertGreater(torch.autograd.grad(penalty,parameter,retain_graph=True)[0].norm(),0)
        g=torch.autograd.grad(penalty,model.moe_head.router.weight,allow_unused=True)[0]
        self.assertTrue(g is None or g.abs().sum()==0)

    def test_classifier_analytic_gradient_matches_autograd(self):
        cache=toy_cache()
        # Opposite existing classifier residual gradients give conflict exactly one.
        x=torch.full((24,6,1),100.)
        y=torch.zeros(24,dtype=torch.long)
        logits=torch.zeros(24,6,2)
        x[8:16]=-100.
        data=dict(expert_features=x,expert_logits=logits,labels=y,domains=cache['domains'])
        gaps,_=classifier_gradient_gaps(data)
        self.assertAlmostEqual(gaps[3,0].item(),9999/10001,places=6)
        self.assertAlmostEqual(gaps[5,0].item(),0.,places=6)
        head=nn.Linear(1,2)
        with torch.no_grad():head.weight.zero_();head.bias.zero_()
        loss=F.cross_entropy(head(x[:8,0]),y[:8]);loss.backward()
        analytic=(logits[:8,0].softmax(-1)-F.one_hot(y[:8],2).float()).mean(0)
        torch.testing.assert_close(head.weight.grad[:,0],analytic*100)
        torch.testing.assert_close(head.bias.grad,analytic)

    def test_probe_disjoint_and_gap_independent(self):
        cache=toy_cache();fit,probe=auxiliary_partition(cache)
        self.assertFalse(set(fit['image_ids'])&set(probe['image_ids']))
        probes=fit_gap_probes(fit,max_iter=300)
        gap,se=evaluate_gaps(probes,probe)
        self.assertTrue(torch.isfinite(gap).all())
        self.assertTrue(torch.isfinite(se).all())
        for m in range(6):
            for subset in range(1,8):
                torch.testing.assert_close(probes[m,subset]['mean'],probes[m,1]['mean'])

    def test_resume_augmentation_independent_of_workers_and_rng(self):
        class RandomImage:
            def __len__(self): return 3
            def __getitem__(self, index):
                import random
                import numpy as np
                return torch.tensor([random.random(), np.random.random(), torch.rand(()).item()]), index
        images = ScheduledImages(RandomImage(), 2, 0)
        state = capture_rng()
        first = images[1, 3, 0][0]
        torch.testing.assert_close(state['torch'], torch.get_rng_state())
        torch.randn(13)
        second = images[1, 3, 0][0]
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, images[1, 4, 0][0]))
        import numpy as np
        schedule = np.array([[0,1],[2,0],[1,2]])
        self.assertEqual(list(ScheduledBatches(schedule, 1)), list(ScheduledBatches(schedule, 0))[1:])
        restore_rng(state)

    def test_border_color_stable_across_paired_views(self):
        from PIL import Image
        c = dict(blur_kernel=5, blur_sigma=1., brightness=1.1, contrast=1.1, saturation=.9, border_width=4)
        image = Image.new('RGB', (224,224), (80,100,120))
        a = ViewTransform(c, 'original', env=-1, cohort='border')(image)
        b = ViewTransform(c, 'style', env=-1, cohort='border')(image)
        torch.testing.assert_close(a[:,0,0], b[:,0,0])
        self.assertFalse(torch.equal(a[:,10,10], b[:,10,10]))

    def test_global_control_learns_complementary_mixture(self):
        cache=toy_cache();validation=toy_cache();validation['metadata']['split']='out'
        y=torch.tensor([0,1]*12);logits=torch.zeros(24,6,2)
        logits[:,:4,0]=4;logits[:,4:,1]=4
        for data in [cache,validation]:data.update(labels=y,expert_logits=logits)
        c=dict(router_hidden=8,router_lr=.03,router_epochs=150)
        router=fit_router(cache,validation,c,global_only=True)
        pi=route(router,cache)
        self.assertAlmostEqual(pi[:,:4].sum(1).mean().item(),.5,places=2)
        self.assertLess(router['validation_nll'],.694)
        # A linear selection-risk objective would pick an arbitrary endpoint;
        # the global CE control must learn the stronger interior mixture.

    def test_immediate_ablation_keeps_other_selected_experts(self):
        cache=toy_cache();validation=toy_cache();validation['metadata']['split']='out'
        cache['router_topk']=torch.full((24,6),1/6)
        router=fit_router(cache,validation,dict(router_hidden=8,router_lr=.03,router_epochs=10))
        controls=frozen_controls(cache,{'input_sigma0':router})
        pi=controls['input_sigma0']
        for m in range(6):
            expected=pi.clone();expected[:,m]=0;expected/=expected.sum(1,keepdim=True)
            torch.testing.assert_close(controls[f'drop_expert{m}'],expected)

    def test_fit_refuses_target(self):
        fit=toy_cache();val=copy.deepcopy(fit);val['metadata']=dict(kind='target',split='out')
        with self.assertRaises(ValueError):fit_router(fit,val,{})


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
