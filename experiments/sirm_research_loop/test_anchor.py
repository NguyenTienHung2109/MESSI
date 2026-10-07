import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import torch
from torch import nn
from experiments.sirm_discover_gate.model import GateSIRM
from .anchor_train import StrongAnchorSIRM
from .anchor_run import configuration
from .anchor_run import invocation
from unittest.mock import patch
from experiments.terra_10pct_factor_diagnosis.run import params

class AnchorControl(unittest.TestCase):
    def test_only_anchor_metadata_changes(self):
        bank=Path('/tmp/anchor-config-test')
        old=params(1,'ON',bank);new=configuration('ON',bank)
        for k in ['strong_anchor_checkpoint','strong_anchor_sha256']:new['sirm_discover_config'].pop(k)
        for k in ['strong_anchor_load_at_start']:new['sirm_discover_config'].pop(k)
        self.assertEqual([new['sirm_discover_config'][k] for k in ['global_steps','discover_steps','invariance_steps']],[0,200,200])
        self.assertEqual(new['sirm_discover_config']['diagnostic_steps'],[0,199,200,399])
        for k in ['global_steps','discover_steps','invariance_steps','diagnostic_steps']:new['sirm_discover_config'][k]=old['sirm_discover_config'][k]
        new['sirm_discover_config']['protocol']=old['sirm_discover_config']['protocol']
        self.assertEqual(old,new)
        cmd=invocation('ON',bank);self.assertEqual(cmd[cmd.index('--steps')+1],'400')

    def test_startup_load_and_all_base_losses_cannot_change_global(self):
        torch.manual_seed(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=copy.deepcopy(params(1,'ON',root)['sirm_discover_config'])
            c.update(global_steps=0,discover_steps=200,invariance_steps=200,residual_hidden=8,utility_min_samples=2,utility_interval=5)
            feat=nn.Linear(4,4);feat.n_outputs=4
            reference=GateSIRM(nn.Linear(4,4),4,3,3,c)
            state={name+'.'+k:v.clone() for name in ['featurizer','global_head'] for k,v in getattr(reference,name).state_dict().items()}
            path=root/'anchor.pkl';torch.save(dict(model_dict=state,step=500),path)
            c.update(strong_anchor_checkpoint=str(path),strong_anchor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),strong_anchor_load_at_start=True)
            with patch('domainbed.sirm_discover.DeiTFeaturizer',return_value=feat):
                adapter=StrongAnchorSIRM((4,),3,3,dict(sirm_discover_config=c,sirm_output_dir=str(root),sirm_source_envs=[0,2,3]))
            self.assertEqual(json.loads((root/'anchor_loaded.json').read_text())['step'],0)
            batches=[(torch.randn(4,4),torch.arange(4)%3) for _ in range(3)]
            for _ in range(3):adapter.update(batches)
            for name in ['featurizer','global_head']:
                module=getattr(adapter.model,name)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
                self.assertTrue(all(torch.equal(v,state[name+'.'+k]) for k,v in module.state_dict().items()))
            self.assertEqual(len((root/'frozen_anchor_checks.jsonl').read_text().splitlines()),3)

    def test_copy_at_boundary_and_freeze_preserving_residuals(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=copy.deepcopy(params(1,'OFF',root)['sirm_discover_config'])
            c.update(global_steps=1,discover_steps=1,invariance_steps=1,residual_hidden=8,utility_min_samples=2,utility_interval=1)
            model=GateSIRM(nn.Linear(4,4),4,3,3,c)
            anchor={}
            for name in ['featurizer','global_head']:
                for k,v in getattr(model,name).state_dict().items():anchor[name+'.'+k]=v.clone()+.25
            path=root/'anchor.pkl';torch.save(dict(model_dict=anchor,selected_step=7),path)
            c.update(strong_anchor_checkpoint=str(path),strong_anchor_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            adapter=StrongAnchorSIRM.__new__(StrongAnchorSIRM);nn.Module.__init__(adapter)
            adapter.model=model;adapter.config=c;adapter.output=root;adapter.pending_refresh=None
            batches=[(torch.randn(4,4),torch.arange(4)%3) for _ in range(3)]
            before={k:v.clone() for k,v in model.residuals.state_dict().items()}
            adapter.update(batches)
            self.assertFalse((root/'anchor_loaded.json').exists())
            self.assertTrue(all(torch.equal(before[k],v) for k,v in model.residuals.state_dict().items()))
            adapter.update(batches)
            self.assertEqual(json.loads((root/'anchor_loaded.json').read_text())['step'],1)
            for name in ['featurizer','global_head']:
                module=getattr(model,name)
                self.assertTrue(all(not p.requires_grad for p in module.parameters()))
                self.assertTrue(all(torch.equal(v,anchor[name+'.'+k]) for k,v in module.state_dict().items()))
            adapter.update(batches)
            self.assertTrue(all(torch.equal(v,anchor[name+'.'+k]) for name in ['featurizer','global_head'] for k,v in getattr(model,name).state_dict().items()))
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.residuals.state_dict().items()))

if __name__=='__main__':unittest.main()
