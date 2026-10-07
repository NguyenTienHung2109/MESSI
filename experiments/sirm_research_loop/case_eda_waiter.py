import json,time
from .anchor_run import DEST
from .router_weight import live,write

if __name__=='__main__':
    try:
        pid=json.loads((DEST/'analysis_worker_pid.json').read_text())['pid']
        while live(pid):
            write(DEST/'eda_progress.json',dict(status='waiting_for_main_diagnostics',analysis_pid=pid,analysis_live=True,observed_unix=time.time()));time.sleep(20)
        assert json.loads((DEST/'analysis_progress.json').read_text())['status']=='diagnostics_complete'
        write(DEST/'eda_progress.json',dict(status='matched_case_replay'))
        from .case_eda import run
        run()
        write(DEST/'eda_progress.json',dict(status='case_replay_complete',remaining='visual inspection, EDA interpretation and evidence-based hypothesis'))
    except Exception as exc:
        write(DEST/'eda_progress.json',dict(status='failed',error=str(exc)));raise
