"""Queued CPU diagnostics after verified paired training completion."""
import json,time
from .router_weight import DEST,EXPERIMENT,live,write,ROOT

def main():
    pid=json.loads((DEST/'worker_pid.json').read_text())['pid']
    while live(pid):
        write(DEST/'analysis_progress.json',dict(status='waiting_for_paired_training',worker_pid=pid,worker_live=True,observed_unix=time.time()))
        time.sleep(20)
    assert json.loads((DEST/'progress.json').read_text())['status']=='paired_training_complete','training worker did not complete'
    from .protocol_audit import audit as audit_protocol
    for arm in ['OFF','ON']:
        result=audit_protocol(EXPERIMENT/'runs/L38'/arm)
        write(DEST/f'{arm}_full_protocol_audit.json',result)
        assert result['passed'],result['failed_checks']
    from .router_alignment import audit
    for arm in ['OFF','ON']:
        write(DEST/'analysis_progress.json',dict(status='source_alignment',method=arm))
        audit(EXPERIMENT/'runs/L38'/arm,DEST/f'{arm}_router_alignment.json')
    write(DEST/'analysis_progress.json',dict(status='source_calibrated_controls'))
    from .router_controls import run
    run()
    write(DEST/'analysis_progress.json',dict(status='diagnostics_complete',remaining='scientific verdict, iteration report, leaderboard and next-hypothesis/stop audit'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(DEST/'analysis_progress.json',dict(status='failed',error=str(exc)));raise
