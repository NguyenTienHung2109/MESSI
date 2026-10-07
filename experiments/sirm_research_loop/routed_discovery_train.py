"""Process-local class-balanced spatial bank; standard frozen-anchor runner hooks."""
import runpy
from domainbed import sirm_discover
from .spatial_bank_train import StrongSpatialAnchor
from .routed_discovery_model import RoutedDiscoverySpatialGateSIRM
if __name__=='__main__':
 sirm_discover.GateSIRM=RoutedDiscoverySpatialGateSIRM
 sirm_discover.SIRMDiscoverV2=StrongSpatialAnchor
 runpy.run_module('domainbed.scripts.train',run_name='__main__')
