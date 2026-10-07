"""Iteration18 source replay with exact new confidence router; no target access."""
from pathlib import Path
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from .conditioned_router_model import ConditionedRouterSpatialGateSIRM
from . import direct_router_risk_source_analysis as engine
from .shrinkage import ROOT
DEST=ROOT/'results/sirm_research_loop/iteration_19'
def load(root,state):
 path=root/'experiment/runs/L38/OFF'/('source_best.pkl' if state=='selected' else 'last.pkl')
 p=torch.load(path,map_location='cpu',weights_only=False);c=p['config']
 m=ConditionedRouterSpatialGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c)
 m.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()});m.configure_stage();m.eval()
 assert m.router.owner is m
 return m,p,path
if __name__=='__main__':
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 engine.DEST=DEST;engine.OUT=DEST/'source_analysis';engine.load=load
 engine.run()
