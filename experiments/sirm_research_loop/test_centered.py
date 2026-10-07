import copy,io,unittest,tempfile,hashlib
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from experiments.terra_10pct_factor_diagnosis.run import params
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.sirm_discover.math import component_irm
from .centered_model import CenteredGateSIRM,centered_head

class CenteringChecks(unittest.TestCase):
    def model(self,enabled=True):
        c=copy.deepcopy(params(1,'ON',__import__('pathlib').Path('/tmp/centered-test'))['sirm_discover_config'])
        c.update(global_steps=0,discover_steps=200,invariance_steps=200,residual_hidden=8,center_residuals=enabled,utility_min_samples=2)
        torch.manual_seed(51);return CenteredGateSIRM(nn.Linear(4,4),4,3,3,c)

    def nonzero_heads(self,m):
        with torch.no_grad():
            for expert in m.residuals:
                expert[-1].weight.normal_(0,.3);expert[-1].bias.normal_(0,.2)

    def test_offsets_cancel_and_informative_gradient_survives(self):
        m=self.model();self.nonzero_heads(m);x=torch.randn(18,4);m.train()
        a=m(x);self.assertLess(float(a['delta'].mean(0).abs().max()),1e-7)
        with torch.no_grad():
            for e in m.residuals:e[-1].bias.add_(torch.tensor([.3,-.2,.5]))
        b=m(x);self.assertTrue(torch.allclose(a['delta'],b['delta'],atol=2e-7,rtol=1e-6));self.assertTrue(torch.equal(a['global_logits'],b['global_logits']))
        y=torch.arange(18)%3;loss=nn.functional.cross_entropy(b['logits'],y)
        grads=torch.autograd.grad(loss,list(m.residuals.parameters()))
        self.assertGreater(sum(float(g.square().sum()) for g in grads),1e-8)
        for e in m.residuals:
            biasgrad=torch.autograd.grad(nn.functional.cross_entropy(m(x)['logits'],y),e[-1].bias)[0]
            self.assertLess(float(biasgrad.abs().max()),1e-7)
        self.assertTrue(all(p.grad is None and not p.requires_grad for name in ['featurizer','global_head'] for p in getattr(m,name).parameters()))

    def test_saved_mean_inference_is_batch_independent_and_matches_head(self):
        m=self.model();self.nonzero_heads(m);x=torch.randn(19,4);z=m.featurizer(x).detach();d=torch.tensor([0]*3+[2]*6+[3]*10)
        m.calibrate_features(z,d);m.eval();whole=m(x)
        self.assertTrue(torch.allclose(whole['logits'],torch.cat([m(x[:7])['logits'],m(x[7:])['logits']]),atol=1e-7))
        g,delta,pi=centered_head(m,z);self.assertTrue(torch.equal(g+(pi[:,:6,None]*delta).sum(1),whole['logits']))
        # Equal-domain calibration, not a pooled image mean.
        self.assertLess(float(torch.stack([whole['delta'][d==env].double().mean(0) for env in [0,2,3]]).mean(0).abs().max()),1e-7)
        stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0);restored=self.model();restored.load_state_dict(torch.load(stream,weights_only=False));restored.eval()
        self.assertTrue(torch.equal(m(x)['logits'],restored(x)['logits']))

    def test_disabled_reference_objective_and_update_are_exact(self):
        m=self.model(False);torch.manual_seed(51);base=GateSIRM(nn.Linear(4,4),4,3,3,m.config)
        self.assertTrue(all(torch.equal(v,base.state_dict()[k]) for k,v in m.state_dict().items() if k in base.state_dict()))
        batches=[(torch.randn(4,4),torch.arange(4)%3) for _ in range(3)]
        a=m.update(batches);b=base.update(batches)
        self.assertEqual(a,b)
        self.assertTrue(all(torch.equal(v,base.state_dict()[k]) for k,v in m.state_dict().items() if k in base.state_dict()))

    def test_binary_sirm_has_nonzero_gradient_on_centered_factors(self):
        m=self.model();self.nonzero_heads(m);m.train();x=torch.randn(18,4);y=torch.arange(18)%3;domains=torch.arange(3).repeat_interleave(6);f=m(x)
        q=torch.zeros(3,6);q[0,0]=q[1,0]=1
        loss,info=component_irm(f['global_logits'],f['delta'],f['pi'],y,domains,q,min_ess=2.,weighting='binary')
        self.assertTrue(info['active']);self.assertGreater(float(loss),1e-12)
        grad=torch.autograd.grad(loss,list(m.residuals.parameters()),allow_unused=True)
        self.assertGreater(sum(float(g.square().sum()) for g in grad if g is not None),1e-12)
        self.assertTrue(all(p.grad is None and not p.requires_grad for name in ['featurizer','global_head'] for p in getattr(m,name).parameters()))

    def test_runtime_mean_refreshes_after_updates_without_global_changes(self):
        m=self.model();self.nonzero_heads(m)
        m._source_features=torch.randn(21,4);m._source_domains=torch.tensor([0]*4+[2]*7+[3]*10);m.runtime_calibration=True
        before={name:{k:v.clone() for k,v in getattr(m,name).state_dict().items()} for name in ['featurizer','global_head']}
        m.eval();self.assertEqual(int(m.centering_calibration_step),0);old=m.expert_source_mean.clone()
        m.update([(torch.randn(4,4),torch.arange(4)%3) for _ in range(3)])
        self.assertEqual(int(m.centering_calibration_step),0)
        m.eval();self.assertEqual(int(m.centering_calibration_step),1);self.assertFalse(torch.equal(old,m.expert_source_mean))
        self.assertTrue(all(torch.equal(v,getattr(m,name).state_dict()[k]) for name,state in before.items() for k,v in state.items()))
        self.assertTrue(all(p.grad is None for name in before for p in getattr(m,name).parameters()))

    def test_adapter_saves_current_source_mean_for_exact_restore(self):
        from .centered_train import StrongCenteredAnchor
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference=self.model();state={name+'.'+k:v.clone() for name in ['featurizer','global_head'] for k,v in getattr(reference,name).state_dict().items()}
            anchor=root/'anchor.pkl';torch.save(dict(model_dict=state,selected_step=375),anchor);sha=hashlib.sha256(anchor.read_bytes()).hexdigest()
            cache=root/'cache.pt';torch.save(dict(features=torch.randn(1172,4),domains=torch.tensor([0]*379+[2]*320+[3]*473),anchor_sha256=sha,view='original',target_read=False,label_conditioned=False),cache)
            c=copy.deepcopy(reference.config);c.update(strong_anchor_load_at_start=True,strong_anchor_checkpoint=str(anchor),strong_anchor_sha256=sha,center_feature_cache=str(cache),center_feature_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
            feat=nn.Linear(4,4);feat.n_outputs=4
            with patch('domainbed.sirm_discover.GateSIRM',CenteredGateSIRM),patch('domainbed.sirm_discover.DeiTFeaturizer',return_value=feat):
                adapter=StrongCenteredAnchor((4,),3,3,dict(sirm_discover_config=c,sirm_output_dir=str(root),sirm_source_envs=[0,2,3]))
            adapter.update([(torch.randn(4,4),torch.arange(4)%3) for _ in range(3)])
            adapter.model.step.fill_(201);adapter.eval();x=torch.randn(11,4);expected=adapter.predict(x).detach()
            adapter.train();adapter.on_evaluation({f'env{d}_out_acc':.5 for d in [0,2,3]},200,True)
            payload=torch.load(root/'source_best.pkl',weights_only=False);restored=self.model();restored.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});restored.eval()
            self.assertEqual(int(restored.centering_calibration_step),201)
            self.assertTrue(torch.equal(expected,restored(x)['logits']))
            self.assertTrue(all(torch.equal(v,getattr(restored,name).state_dict()[k]) for name in ['featurizer','global_head'] for k,v in getattr(reference,name).state_dict().items()))

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
