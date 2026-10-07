"""Process-local auxiliary cap with the unchanged strong frozen anchor adapter."""
import runpy
from domainbed import sirm_discover
from .centered_train import StrongCenteredAnchor
from .router_cap_model import RouterCapGateSIRM

if __name__ == '__main__':
    sirm_discover.GateSIRM = RouterCapGateSIRM
    sirm_discover.SIRMDiscoverV2 = StrongCenteredAnchor
    runpy.run_module('domainbed.scripts.train', run_name='__main__')
