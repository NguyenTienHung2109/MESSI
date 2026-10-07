"""Iteration18 source grids from completed replays, no new inference."""
import json
from pathlib import Path
from . import direct_router_risk_source_grids as engine
from .shrinkage import ROOT,digest,write
if __name__=='__main__':
 engine.DEST=ROOT/'results/sirm_research_loop/iteration_18'
 draw=engine.grids.draw
 def confidence_draw(name,title,*args,**kwargs):
  return draw(name,title.replace('iteration17','iteration18').replace('actual-mixture router risk','shared confidence router'),*args,**kwargs)
 engine.grids.draw=confidence_draw
 engine.main()
 index=engine.DEST/'source_failure_grids/index.json'
 proof=json.loads(index.read_text());proof['wrapper_code_sha256']=digest(Path(__file__));write(index,proof)
