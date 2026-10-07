"""Process-local paired hard-discovery training, preserving source centering."""
import runpy
from domainbed import sirm_discover
from .centered_train import StrongCenteredAnchor
from .hard_discovery import HardWinnerGateSIRM


def main():
    sirm_discover.GateSIRM=HardWinnerGateSIRM
    sirm_discover.SIRMDiscoverV2=StrongCenteredAnchor
    runpy.run_module('domainbed.scripts.train',run_name='__main__')


if __name__=='__main__':main()
