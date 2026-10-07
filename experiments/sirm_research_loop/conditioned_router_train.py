"""Process-local matched-budget confidence routing on immutable ERM."""
import runpy
import torch
from domainbed import sirm_discover
from .spatial_bank_train import StrongSpatialAnchor
from .conditioned_router_model import ConditionedRouterSpatialGateSIRM
if __name__=='__main__':
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 sirm_discover.GateSIRM=ConditionedRouterSpatialGateSIRM
 sirm_discover.SIRMDiscoverV2=StrongSpatialAnchor
 runpy.run_module('domainbed.scripts.train',run_name='__main__')
