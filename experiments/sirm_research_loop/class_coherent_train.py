"""Process-local training adapter; frozen anchor and optimizer unchanged."""
import runpy
from domainbed import sirm_discover
from .centered_train import StrongCenteredAnchor
from .class_coherent_model import ClassCoherentGateSIRM

if __name__ == '__main__':
    sirm_discover.GateSIRM = ClassCoherentGateSIRM
    sirm_discover.SIRMDiscoverV2 = StrongCenteredAnchor
    runpy.run_module('domainbed.scripts.train', run_name='__main__')
