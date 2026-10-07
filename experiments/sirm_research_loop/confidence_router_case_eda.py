"""Locked selected-checkpoint original-image EDA for iteration18."""
import torch
from .shrinkage import ROOT,digest,write
from .routed_discovery_run import ANCHOR
from .confidence_router_replay import install
from . import case_eda
from pathlib import Path

def main():
 install();case_eda.DEST=ROOT/'results/sirm_research_loop/iteration_18';case_eda.EXPERIMENT=case_eda.DEST/'experiment';case_eda.ANCHOR=ANCHOR
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 write(case_eda.DEST/'case_eda_plan.json',dict(anchor_sha256=digest(ANCHOR),selected_checkpoints_only=True,training=False,checkpoint_changes=False,target_selection=False,source_then_target_replay=True,code_sha256=digest(Path(__file__)),engine_sha256=digest(Path(case_eda.__file__)),purpose='Per-image original source/target classification and utility EDA after completed source review. Target exploratory, no candidate tuning.'))
 case_eda.run();print('Actual source/target case replay complete',flush=True)
if __name__=='__main__':main()
