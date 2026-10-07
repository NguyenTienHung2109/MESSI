import runpy
from domainbed import sirm_discover
from .spatial_bank_train import StrongSpatialAnchor
from .correct_guard_model import CorrectGuardSpatialGateSIRM
if __name__=='__main__':
 sirm_discover.GateSIRM=CorrectGuardSpatialGateSIRM
 sirm_discover.SIRMDiscoverV2=StrongSpatialAnchor
 runpy.run_module('domainbed.scripts.train',run_name='__main__')
