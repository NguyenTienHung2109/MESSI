import unittest
import torch
import torch.nn.functional as F
from experiments.sirm_q_followup.diagnose import scalar_cells
from experiments.sirm_q_followup.pilot import penalty_q
from domainbed.subset_irm import subset_irm_penalty

class DiagnosticsTests(unittest.TestCase):
    def test_derivative_matches_autograd_and_production_penalty(self):
        gen=torch.Generator().manual_seed(49)
        z=torch.randn(36,6,7,generator=gen);y=torch.randint(7,(36,),generator=gen);domains=torch.arange(3).repeat_interleave(12)
        gamma=torch.rand(36,6,generator=gen).softmax(1)
        rows=scalar_cells(z,y,domains,gamma)
        for r in rows:
            mask=domains==r['domain']
            for m in range(6):
                scale=torch.ones((),dtype=torch.double,requires_grad=True)
                ce=F.cross_entropy(scale*z[mask,m].double(),y[mask],reduction='none')
                w=gamma[mask,m].double();risk=(ce*w).sum()/w.sum()
                g=torch.autograd.grad(risk,scale)[0]
                torch.testing.assert_close(g,r['weighted'][m])
        q=torch.full((3,6),1/6);loss,info=subset_irm_penalty(z,y,domains,gamma,q)
        expected=torch.stack([r['weighted'].square() for r in rows]).mean()
        torch.testing.assert_close(loss.double(),expected,atol=1e-7,rtol=1e-5)
        self.assertTrue(info['active_cells'].all())

    def test_random_is_competence_independent_feasible_and_global_normalized(self):
        q=torch.tensor([[.7,.3,0,0,0,0],[0,.4,.6,0,0,0],[.2,0,.8,0,0,0]])
        perm=torch.tensor([2,3,4,5,0,1]);random=penalty_q(q,'random',perm)
        torch.testing.assert_close(random.sum(1),q.sum(1))
        torch.testing.assert_close(random,penalty_q(q.flip(1),'random',perm))
        self.assertTrue(random.gt(0).sum(0)[random.gt(0).any(0)].eq(2).all())
        self.assertTrue(random.gt(0).sum(1).eq(2).all())
        torch.testing.assert_close(penalty_q(q,'global',perm).sum(1),torch.ones(3))
        torch.testing.assert_close(penalty_q(q,'learned',perm),q)

if __name__=='__main__':unittest.main()
