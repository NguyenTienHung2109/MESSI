"""Explicit process-local restore/replay integration for centered checkpoints."""
from .centered_model import CenteredGateSIRM,centered_head

def install():
    from experiments.sirm_discover_binary import run as binary
    from . import shrinkage,case_eda,bias_control
    binary.GateSIRM=CenteredGateSIRM
    for module in [shrinkage,case_eda,bias_control]:
        module.GateSIRM=CenteredGateSIRM;module.head=centered_head
