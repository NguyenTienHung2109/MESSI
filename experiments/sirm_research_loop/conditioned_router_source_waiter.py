"""Watch the existing iteration19 worker, analyze only completed paired source runs."""
import json,os,subprocess,time
from pathlib import Path
from .shrinkage import ROOT,write,digest
DEST=ROOT/'results/sirm_research_loop/iteration_19'
if __name__=='__main__':
 worker=json.loads((DEST/'worker_pid.json').read_text())['pid']
 while not (DEST/'training_audit.json').exists():
  progress=json.loads((DEST/'progress.json').read_text());assert progress['status']!='failed',progress
  os.kill(worker,0)
  write(DEST/'source_waiter_progress.json',dict(status='waiting_existing_training_worker',worker_pid=worker,observed_unix=time.time(),target_read=False))
  time.sleep(5)
 assert json.loads((DEST/'training_audit.json').read_text())['passed']
 for arm in ['OFF','ON']:assert (DEST/'experiment/runs/L38'/arm/'done').exists()
 result=subprocess.run([os.sys.executable,'-m','experiments.sirm_research_loop.conditioned_router_source_analysis'],cwd=ROOT)
 assert result.returncode==0
 write(DEST/'source_waiter_progress.json',dict(status='completed_source_replay',target_read=False,source_result_sha256=digest(DEST/'source_analysis/results.json'),observed_unix=time.time()))
