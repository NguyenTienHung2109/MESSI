"""Frozen-feature classifiers with a common fit/predict/reset interface."""

from .primitives import NCM, KNN, Ridge, SimpleShotSupport, VisualCache, CLIPZeroShot, TipAdapter, TipAdapterFull
from .gda import GDA, GDAClip
from .ape import APETrainingFree
from .ape_t import APET
from .calip import CALIPFixed
from .timo import TIMOFixed
from .proker import ProKeR, ProKeROrthogonalRFF
from .taskres import TaskRes
from .clip_adapter import CLIPAdapter
from .tip_adapter_f import TipAdapterF
from .clap import CLAP
from .logistic_probe import LogisticProbe
from .lp_pp import LPPlusPlus
from .cross_modal_probe import CrossModalLinearProbe
from .cross_modal_clap import CrossModalCLAP
from .ecalp_inductive import ECALPInductive

__all__ = ["NCM", "KNN", "Ridge", "SimpleShotSupport", "VisualCache", "CLIPZeroShot", "TipAdapter", "TipAdapterFull", "TipAdapterF", "GDA", "GDAClip", "APETrainingFree", "APET", "CALIPFixed", "TIMOFixed", "ProKeR", "ProKeROrthogonalRFF", "TaskRes", "CLIPAdapter", "CLAP", "LogisticProbe", "LPPlusPlus", "CrossModalLinearProbe", "CrossModalCLAP", "ECALPInductive"]
