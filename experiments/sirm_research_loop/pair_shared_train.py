"""Process-local paired discovery, existing frozen centered anchor adapter."""
import runpy
from domainbed import sirm_discover
from .centered_train import StrongCenteredAnchor
from .pair_shared_model import PairSharedGateSIRM
if __name__=='__main__':
    sirm_discover.GateSIRM=PairSharedGateSIRM
    sirm_discover.SIRMDiscoverV2=StrongCenteredAnchor
    runpy.run_module('domainbed.scripts.train',run_name='__main__')
