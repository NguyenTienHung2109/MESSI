"""Early completed OFF source analysis while paired ON trains; no target access."""
import json,os,time
import torch
from .conditioned_router_source_analysis import DEST,load
from . import direct_router_risk_source_analysis as engine
from .shrinkage import write
if __name__=='__main__':
 worker=json.loads((DEST/'worker_pid.json').read_text())['pid']
 while not (DEST/'experiment/runs/L38/OFF/done').exists():
  progress=json.loads((DEST/'progress.json').read_text());assert progress['status']!='failed',progress;os.kill(worker,0);time.sleep(5)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 engine.DEST=DEST;engine.OUT=DEST/'source_analysis';engine.load=load
 engine.run(only_off=True)
 write(DEST/'OFF_source_review_progress.json',dict(status='OFF_source_replay_complete',ON='pending training/full paired analysis',target_read=False,observed_unix=time.time()))
