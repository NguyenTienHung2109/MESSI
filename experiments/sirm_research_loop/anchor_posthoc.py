"""Source-calibrated shrinkage control after frozen-ERM400 pair completion."""
import json,time
from .anchor_run import DEST,EXPERIMENT
from .router_weight import live,write

def main():
    pid=json.loads((DEST/'worker_pid.json').read_text())['pid']
    while live(pid):
        write(DEST/'analysis_progress.json',dict(status='waiting_for_paired_training',worker_pid=pid,worker_live=True,observed_unix=time.time()));time.sleep(20)
    assert json.loads((DEST/'progress.json').read_text())['status']=='paired_training_complete'
    write(DEST/'analysis_progress.json',dict(status='source_calibrated_shrinkage_control'))
    from .shrinkage import run
    run(suite=EXPERIMENT,dest=DEST/'shrinkage_control')
    write(DEST/'analysis_progress.json',dict(status='diagnostics_complete',remaining='required frozen-ERM report, scientific verdict, leaderboard and completion audit'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(DEST/'analysis_progress.json',dict(status='failed',error=str(exc)));raise
