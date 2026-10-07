"""Independent geometry and routing derivative checks for diagnostic guards."""
from pathlib import Path
import torch
from experiments.sirm_discover.math import router_teacher
from .differentiable_expected_route import differentiable_route,Entmax15Jacobian
from .repair_margin_guard_helpers import project
from .routed_discovery_run import DEST
from .shrinkage import write,digest

def run():
 torch.set_num_threads(4);torch.manual_seed(731)
 scores=torch.randn(4,7,dtype=torch.float64,requires_grad=True)
 entmax_check=torch.autograd.gradcheck(Entmax15Jacobian.apply,(scores,),eps=1e-6,atol=1e-5,rtol=1e-4)
 g=torch.randn(3,10,dtype=torch.float64);delta=(.2*torch.randn(3,6,10,dtype=torch.float64)).requires_grad_();post=torch.randn(3,10,dtype=torch.float64).softmax(-1)
 route_check=torch.autograd.gradcheck(lambda d:differentiable_route(g,d,post),(delta,),eps=1e-6,atol=1e-5,rtol=1e-4)
 u=((g[:,None]+delta).log_softmax(-1)-g.log_softmax(-1)[:,None]).mul(post[:,None]).sum(-1)
 difference=float((differentiable_route(g,delta,post)-router_teacher(u,.1,.002)).abs().max());assert difference<1e-12
 v=torch.tensor([-1.,2.,-3.],dtype=torch.float64);guards=[torch.tensor([1.,0.,0.],dtype=torch.float64),torch.tensor([0.,0.,1.],dtype=torch.float64)];p,info=project(v,guards);assert torch.allclose(p,torch.tensor([0.,2.,0.],dtype=torch.float64),atol=1e-8)
 out=DEST/'source_expected_utility_routing_aware_guard';write(out/'math_verification.json',dict(entmax_finite_difference=entmax_check,complete_route_finite_difference=route_check,teacher_value_max_difference=difference,known_projection_solution=p.tolist(),projection_certificate=info,seed=731,source_only=True,target_read=False,bank_updates=False,scope='Independent nonsingular numerical fixtures; not a guarantee against nonsmooth support changes or finite-step cohort margin decreases.',code_sha256=digest(Path(__file__))))
 (out/'math_verification_code.py').write_bytes(Path(__file__).read_bytes());print('Independent guard checks pass')
if __name__=='__main__':run()
