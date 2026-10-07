"""Hard-discovery evidence draft after verified paired analysis."""
import argparse,json,time
from pathlib import Path
from .router_weight import live
from experiments.terra_10pct_factor_diagnosis.run import ROOT,digest,write

def run(dest):
    files=[Path(__file__),Path(__file__).with_name('source_selected_report.py'),Path(__file__).with_name('source_router_case_review.py')]
    plan=dict(code_hashes={str(p.relative_to(ROOT)):digest(p) for p in files},scope='Evidence draft only; scientific verdict and visual EDA remain pending manual review.')
    lock=dest/'reporting_plan.json'
    if lock.exists():assert json.loads(lock.read_text())==plan
    else:
        write(lock,plan)
        for p in files:
            target=dest/'reporting_code_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    pid=json.loads((dest/'analysis_worker_pid.json').read_text())['pid']
    while live(pid):
        assert all(digest(ROOT/p)==h for p,h in plan['code_hashes'].items())
        write(dest/'reporting_progress.json',dict(status='waiting_for_analysis',analysis_pid=pid,verified_live=True,observed_unix=time.time()));time.sleep(20)
    assert json.loads((dest/'analysis_progress.json').read_text())['status']=='diagnostics_complete'
    assert all(digest(ROOT/p)==h for p,h in plan['code_hashes'].items())
    from .source_router_case_review import run as cases
    cases(dest)
    from .source_selected_report import main
    main(destination=dest)
    report=dest/'report.md'
    text=report.read_text()
    text=text.replace('Stage-2 configuration:', 'Stage-2 configuration:\n\nHard-discovery intervention: only L_discover becomes the mean minimum single-expert CE, with equal derivatives across exact ties. L_balance still uses its original differentiable soft responsibility at temperature0.25; router teacher, centering, all weights, Q/ESS and SIRM settings remain unchanged. Assignment loads/ties are logged from pre-update predictions in diagnostics.jsonl. No claim of factor learning follows from hard assignment alone.')
    report.write_text(text)
    from .leaderboard import update
    update()
    write(dest/'reporting_progress.json',dict(status='evidence_draft_complete',remaining='Manual original-image review, joint control interpretation, scientific verdict and completion audit',scientific_goal_achieved=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path);args=parser.parse_args()
    try:run(args.destination.resolve())
    except Exception as exc:write(args.destination/'reporting_progress.json',dict(status='failed',error=str(exc)));raise
