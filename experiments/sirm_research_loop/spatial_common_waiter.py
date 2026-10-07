"""Verify reporting terminal state before common-step source comparison."""
import json,time
from pathlib import Path
from .spatial_bank_run import DEST
from .router_weight import live
from .shrinkage import write,digest
from .spatial_common_state import run

def main():
    code=Path(__file__).with_name('spatial_common_state.py');sha=digest(code)
    pid=json.loads((DEST/'reporting_worker_pid.json').read_text())['pid']
    while live(pid):
        assert digest(code)==sha;write(DEST/'common_state_progress.json',dict(status='waiting_for_verified_reporting',pid=pid,verified_live=True,observed_unix=time.time()));time.sleep(20)
    assert json.loads((DEST/'reporting_progress.json').read_text())['status']=='evidence_draft_complete'
    assert digest(code)==sha;write(DEST/'common_state_progress.json',dict(status='source_only_replay'));run()
    write(DEST/'common_state_progress.json',dict(status='complete',code_sha256=sha,target_read=False,remaining='Manual visual EDA and final interpretation'))

if __name__=='__main__':
    try:main()
    except Exception as exc:write(DEST/'common_state_progress.json',dict(status='failed',error=str(exc)));raise
