"""Wait for a verified specific training worker, then source-only analysis."""
import os,json,time
from pathlib import Path
from .shrinkage import ROOT,write
DEST=ROOT/'results/sirm_research_loop/iteration_17'
def main():
 p=json.loads((DEST/'worker_pid.json').read_text());pid=p['pid'];os.kill(pid,0)
 write(DEST/'source_waiter_progress.json',dict(status='waiting',verified_worker_pid=pid))
 while not (DEST/'training_audit.json').exists():
  progress=json.loads((DEST/'progress.json').read_text())
  assert progress['status']!='failed',progress
  try:os.kill(pid,0)
  except ProcessLookupError:raise RuntimeError('Specific training worker stopped before training audit; inspect progress and logs, do not restart automatically')
  time.sleep(10)
 assert json.loads((DEST/'training_audit.json').read_text())['passed']
 write(DEST/'source_waiter_progress.json',dict(status='source_analysis'))
 from .direct_router_risk_source_analysis import run
 run()
 write(DEST/'source_waiter_progress.json',dict(status='source_analysis_complete',target_read=False))
if __name__=='__main__':
 try:main()
 except Exception as e:write(DEST/'source_waiter_progress.json',dict(status='failed',error=str(e)));raise
