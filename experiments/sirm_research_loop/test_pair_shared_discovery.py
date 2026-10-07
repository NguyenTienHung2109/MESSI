import torch
from experiments.sirm_research_loop.pair_shared_discovery import paired_shared_loss

def test_shared_beats_domain_exclusive():
    g=torch.zeros(2,2);y=torch.zeros(2,dtype=torch.long);d=torch.tensor([0,1])
    delta=torch.tensor([[[3.,-3.],[-3.,3.],[1.,-1.]], [[-3.,3.],[3.,-3.],[1.,-1.]]],requires_grad=True)
    loss,info=paired_shared_loss(g,delta,y,d);loss.backward()
    assert info['pairs'][0]['worst_excess'][2]<min(info['pairs'][0]['worst_excess'][:2])
    assert delta.grad[:,:2].abs().max()==0 and delta.grad[:,2].abs().sum()>0
    assert loss<0

def test_zero_ties_symmetric():
    g=torch.zeros(4,2);y=torch.zeros(4,dtype=torch.long);d=torch.tensor([0,0,1,1]);delta=torch.zeros(4,3,2,requires_grad=True)
    loss,_=paired_shared_loss(g,delta,y,d);loss.backward();assert loss==0
    assert torch.allclose(delta.grad,delta.grad[:1,:1].expand_as(delta.grad))
    assert delta.grad.abs().sum()>0

def test_permutations_and_frozen_global():
    torch.manual_seed(0);g=torch.randn(6,3,requires_grad=True);delta=torch.randn(6,4,3,requires_grad=True);y=torch.tensor([0,0,1,1,2,2]);d=torch.tensor([0,1,0,1,0,1]);p=torch.tensor([4,0,3,5,1,2]);a,_=paired_shared_loss(g,delta,y,d);b,_=paired_shared_loss(g[p],delta[p],y[p],9-d[p]);assert torch.allclose(a,b);a.backward();assert g.grad is None and delta.grad.abs().sum()>0

def test_no_pair_fallback():
    g=torch.zeros(2,2);delta=torch.zeros(2,3,2,requires_grad=True);y=torch.tensor([0,1]);d=torch.tensor([0,1]);loss,info=paired_shared_loss(g,delta,y,d);assert info['fallback'];loss.backward();assert delta.grad.abs().sum()>0
