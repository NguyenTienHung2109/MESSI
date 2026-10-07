import json,copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.terra_10pct_factor_diagnosis.subsets import MANIFEST,RESULT,verify
from experiments.terra_10pct_factor_diagnosis.run import params,command
from experiments.terra_10pct_factor_diagnosis.erm import ERMH2Diagnostic
from experiments.sirm_discover_gate.model import GateSIRM
class Tiny(torch.nn.Module):
    n_outputs=4
    def __init__(self,**kw):super().__init__();self.layer=torch.nn.Linear(4,4)
    def forward(self,x):return self.layer(x)
class Tests(unittest.TestCase):
    def test_manifest_and_roles(self):
        m=json.loads(MANIFEST.read_text());verify(m);self.assertEqual(MANIFEST.read_bytes(),(RESULT/'subset_manifest.json').read_bytes())
        for target in range(4):
            target_ids=set(sum([m['sources'][str(target)][s+'_ids'] for s in ['in','out']],[]))
            source_ids=set(sum([m['sources'][str(e)][s+'_ids'] for e in range(4) if e!=target for s in ['in','out']],[]));self.assertFalse(target_ids&source_ids)
    def test_min_two_enforced(self):
        m=json.loads(MANIFEST.read_text());e=m['sources']['0'];groups={c:[i for i in e['out'] if e['images'][i]['class']==c] for c in range(len(m['class_names']))};ids=next(v for v in groups.values() if len(v)==2)
        e['out'].remove(ids[0]);e['out_ids']=[e['images'][i]['id'] for i in e['out']]
        with self.assertRaises(AssertionError):verify(m)
    def test_configuration_control(self):
        for target in range(4):
            a=params(target,'OFF',Path('/tmp/test'));b=params(target,'ON',Path('/tmp/test'))
            self.assertEqual(a['sirm_discover_config'].pop('lambda_sirm'),0.);self.assertEqual(b['sirm_discover_config'].pop('lambda_sirm'),1.);self.assertEqual(a,b)
    def test_shortened_schedule(self):
        for method in ['ERM_H2','OFF','ON']:
            hp=params(1,method,Path('/tmp/test'));c=hp['sirm_discover_config']
            self.assertEqual([c[k] for k in ['global_steps','discover_steps','invariance_steps']],[100,100,301])
            self.assertEqual(c['utility_interval'],5);self.assertEqual(c['sirm_ramp_steps'],50);self.assertEqual(c['checkpoint_freq'],25)
            self.assertEqual(c['diagnostic_steps'],[199,200,350,500])
            cmd=command(1,method,Path('/tmp/test'));self.assertEqual(cmd[cmd.index('--steps')+1],'501');self.assertEqual(cmd[cmd.index('--checkpoint_freq')+1],'25')
    def test_pure_erm_and_same_global_initialization(self):
        with tempfile.TemporaryDirectory() as d:
            hp=params(1,'ERM_H2',Path(d));c=hp['sirm_discover_config']
            with patch('experiments.terra_10pct_factor_diagnosis.erm.DeiTFeaturizer',Tiny):
                torch.manual_seed(0);a=ERMH2Diagnostic((4,),10,3,hp)
                torch.manual_seed(0);b=GateSIRM(Tiny(),4,10,3,c)
                self.assertFalse(hasattr(a,'residuals'));self.assertFalse(hasattr(a,'router'));self.assertTrue(all(p.requires_grad for p in a.parameters()))
                for x,y in zip(a.global_head.parameters(),b.global_head.parameters()):self.assertTrue(torch.equal(x,y))
                batch=[(torch.randn(4,4),torch.arange(4)) for _ in range(3)]
                before=a.featurizer.layer.weight.clone();row=a.update(batch);self.assertFalse(torch.equal(before,a.featurizer.layer.weight));self.assertEqual(row['loss'],row['loss_cls'])
if __name__=='__main__':unittest.main()
