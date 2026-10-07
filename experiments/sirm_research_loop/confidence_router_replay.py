"""Process-local exact confidence-router adapters for completed-bank inference."""
from .confidence_router_model import ConfidenceRouterSpatialGateSIRM
from .spatial_bank_replay import install as spatial_install,RawTokenFeaturizer,spatial_head

def install():
 spatial_install()
 from experiments.sirm_discover_binary import run as binary
 from . import shrinkage,case_eda,bias_control
 binary.GateSIRM=ConfidenceRouterSpatialGateSIRM
 for module in [shrinkage,case_eda,bias_control]:
  module.GateSIRM=ConfidenceRouterSpatialGateSIRM
  module.DeiTFeaturizer=RawTokenFeaturizer
  module.head=spatial_head
