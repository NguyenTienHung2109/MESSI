import runpy
import torch
from domainbed import sirm_discover
from .spatial_bank_train import StrongSpatialAnchor
from .direct_router_risk_model import DirectRouterRiskSpatialGateSIRM
if __name__=='__main__':
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 sirm_discover.GateSIRM=DirectRouterRiskSpatialGateSIRM
 sirm_discover.SIRMDiscoverV2=StrongSpatialAnchor
 runpy.run_module('domainbed.scripts.train',run_name='__main__')
