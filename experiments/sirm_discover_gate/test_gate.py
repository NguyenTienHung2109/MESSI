import copy,tempfile,unittest
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from experiments.sirm_discover.test_discover import cfg
from experiments.sirm_discover.run import weights_hash
from experiments.sirm_h1_h6 import run as base
from .model import GateSIRM
from .data import Adapter,SPECS


class AssignmentTests(unittest.TestCase):
    def model(self,policy):
        torch.manual_seed(21)
        config=cfg();config['q_freeze_stage3']=True
        return GateSIRM(nn.Linear(4,4),4,3,3,config,policy=policy)

    def test_initial_state_matches_and_assignment_rng_is_local(self):
        a=self.model('learned');rng=torch.get_rng_state().clone()
        b=self.model('random')
        self.assertEqual(weights_hash(a),weights_hash(b))
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        self.assertTrue((a.assignment_permutation!=torch.arange(3)).all())

    def test_random_weights_sparsity_pair_strength_and_lock(self):
        a=self.model('learned');b=self.model('random')
        u=torch.tensor([[.20,.01,.15],[.18,.19,0.],[0.,.17,.14]])
        ds=torch.arange(3).repeat_interleave(2);samples=u.repeat_interleave(2,0)
        for m in [a,b]:
            m.step.fill_(3);m.update_utility(samples,ds)
        torch.testing.assert_close(a.q,b.q,rtol=0,atol=0)
        for m in [a,b]:
            m.step.fill_(4);m.update_utility(samples,ds)
        torch.testing.assert_close(a.q,b.q[:,torch.argsort(b.assignment_permutation)],rtol=0,atol=0)
        torch.testing.assert_close(a.q.sort(1).values,b.q.sort(1).values,rtol=0,atol=0)
        torch.testing.assert_close(a.q@a.q.T,b.q@b.q.T)
        torch.testing.assert_close(a.q_null,b.q_null,rtol=0,atol=0)
        old_q=b.q.clone();old_ema=b.utility_ema.clone()
        b.step.fill_(5);b.update_utility(samples.flip(1),ds)
        torch.testing.assert_close(old_q,b.q,rtol=0,atol=0)
        self.assertFalse(torch.equal(old_ema,b.utility_ema))
        restored=self.model('random');restored.load_state_dict(copy.deepcopy(b.state_dict()));restored.apply_policy()
        torch.testing.assert_close(restored.q,b.q,rtol=0,atol=0)

    def test_trajectories_identical_through_zero_ramp_activation(self):
        models=[self.model(p) for p in ['learned','learned','random']]
        models[0].config['lambda_sirm']=0.
        torch.manual_seed(31);data=[(torch.randn(4,4),torch.tensor([0,1,2,i])) for i in range(3)]
        for _ in range(5):
            rows=[m.update(data) for m in models]
            for key in ['loss_cls','loss_discover','loss_router','loss_balance','loss']:
                self.assertEqual(rows[0][key],rows[1][key]);self.assertEqual(rows[0][key],rows[2][key])
        for m in models:self.assertTrue(m.assignment_locked)


class SplitTests(unittest.TestCase):
    def test_terra_infers_classes_and_never_inventories_target(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter=Adapter('TerraIncognita');sources=[0,2,3]
            for env in sources:
                for cls in ['cat','dog']:
                    folder=adapter.folder(temp,env)/cls;folder.mkdir(parents=True)
                    for i in range(5):Image.new('RGB',(8,8)).save(folder/f'{i}.png')
            split=adapter.inventory(temp,sources,0,cfg())
            self.assertEqual(adapter.classes,['cat','dog']);self.assertEqual(set(split['sources']),{'0','2','3'})
            self.assertFalse(adapter.folder(temp,1).exists())
            for env in sources:
                fit,out=base.split_keys(10,.2,0,env)
                self.assertEqual(split['sources'][str(env)]['in'],fit)
                self.assertEqual(split['sources'][str(env)]['out'],out)
            validation=adapter.source_data(temp,split,cfg(),'out',limit=2)
            self.assertEqual(set(validation),set(sources))

if __name__=='__main__':unittest.main()
