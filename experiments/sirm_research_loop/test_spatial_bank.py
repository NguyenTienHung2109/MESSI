import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from .spatial_bank_model import SpatialBankGateSIRM,pooled_raw_features
from .test_class_coherent import ClassCoherentChecks

class ToyTokens(nn.Module):
    def __init__(self):
        super().__init__();self.vit=nn.Module();self.vit.norm=nn.LayerNorm(4)
    def forward(self,x):return self.vit.norm(x)[:,0]

class SpatialBankChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'cache.pt'
        gen=torch.Generator().manual_seed(913);raw=torch.randn(24,8,generator=gen);torch.save(dict(features=raw,domains=torch.arange(24)%3,anchor_sha256='test',view='original',label_conditioned=False,target_read=False),self.path)
    def model(self,mode='cls_patch_max',on=False):
        c=copy.deepcopy(ClassCoherentChecks().model(on).config);c.update(global_steps=0,bank_input_mode=mode,center_feature_cache=str(self.path),strong_anchor_sha256='test')
        return SpatialBankGateSIRM(ToyTokens(),4,3,3,c)
    def heads(self,m):
        torch.manual_seed(713)
        for e in m.residuals:e[-1].weight.data.normal_(0,.1);e[-1].bias.data.normal_(0,.1)
    def test_original_CLS_and_no_prefix_in_spatial_pool(self):
        f=ToyTokens().eval();x=torch.randn(7,5,4);cls,raw=pooled_raw_features(f,x);tokens=f.vit.norm(x)
        self.assertTrue(torch.equal(cls,tokens[:,0]));self.assertTrue(torch.equal(raw[:,4:],tokens[:,2:].amax(1)));self.assertEqual(len(f.vit.norm._forward_hooks),0)
    def test_modes_have_identical_initialized_parameters_and_count(self):
        torch.manual_seed(55);a=self.model('repeat_cls');torch.manual_seed(55);b=self.model()
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        self.assertTrue(all(torch.equal(v,dict(b.named_parameters())[k]) for k,v in a.named_parameters()))
        self.assertEqual({id(p) for g in a.optimizer.param_groups for p in g['params']},{id(p) for p in a.parameters()})
    def test_bank_normalization_uses_only_source_cache(self):
        m=self.model();raw=torch.load(self.path,weights_only=False)['features'];expected=(raw-raw.double().mean(0).float())/raw.double().std(0,unbiased=False).float()/2**.5
        self.assertTrue(torch.allclose(m.bank_features(raw),expected));x=torch.randn(8,5,4);before=m.bank_source_feature_mean.clone();m(x);self.assertTrue(torch.equal(before,m.bank_source_feature_mean))
    def test_global_is_independent_of_bank_mode(self):
        torch.manual_seed(62);a=self.model('repeat_cls');torch.manual_seed(62);b=self.model();self.heads(a);self.heads(b);x=torch.randn(8,5,4)
        self.assertTrue(torch.equal(a(x)['global_logits'],b(x)['global_logits']));self.assertFalse(torch.equal(a(x)['z'],b(x)['z']))
    def test_calibration_is_equal_domain_and_matches_inference(self):
        m=self.model();self.heads(m);cache=torch.load(self.path,weights_only=False);m.calibrate_features(cache['features'],cache['domains']);m.eval()
        delta=m.residual_logits(cache['features']);means=torch.stack([delta[cache['domains'].eq(d)].double().mean(0) for d in range(3)]).mean(0)
        self.assertLess(float(means.abs().max()),1e-7)
    def test_single_update_frozen_pair_and_formula(self):
        torch.manual_seed(15);a=self.model();torch.manual_seed(15);b=self.model(on=True);self.heads(a);b.load_state_dict(a.state_dict());before={k:v.clone() for k,v in a.state_dict().items() if k.startswith(('featurizer.','global_head.'))}
        torch.manual_seed(56);batch=[(torch.randn(8,5,4),torch.arange(8)%3) for _ in range(3)]
        with patch.object(a.optimizer,'step',wraps=a.optimizer.step) as spy:ra=a.update(batch);self.assertEqual(spy.call_count,1)
        rb=b.update(batch);self.assertEqual(ra,rb);self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()));self.assertTrue(all(torch.equal(v,a.state_dict()[k]) for k,v in before.items()));self.assertTrue(all(not p.requires_grad and p.grad is None for mod in [a.featurizer,a.global_head] for p in mod.parameters()));self.assertEqual(a.last_diagnostics['bank_input_dimension'],8);self.assertAlmostEqual(ra['loss_discover'],a.last_diagnostics['discovery_class_loss_replay'],places=6)
    def test_eligible_SIRM_executed_with_spatial_bank(self):
        m=self.model(on=True);self.heads(m);m.step.fill_(251);m.q_learned.zero_();m.q_learned[0,0]=m.q_learned[1,0]=1.;torch.manual_seed(171);r=m.update([(torch.randn(12,5,4),torch.arange(12)%3) for _ in range(3)])
        self.assertTrue(r['sirm_active']);self.assertGreater(r['loss_sirm'],0);self.assertEqual(r['effective_lambda_sirm'],1);self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
