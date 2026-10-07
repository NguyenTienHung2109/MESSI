"""Draft evidence watcher; final EDA and scientific audit remain manual."""
import json,time
from pathlib import Path
from .class_coherent_run import DEST,ROOT
from .router_weight import live
from .shrinkage import digest,write

def main():
 files=[Path(__file__),Path(__file__).with_name('source_selected_report.py'),Path(__file__).with_name('source_router_case_review.py')]
 plan={'code_hashes':{str(p.relative_to(ROOT)):digest(p) for p in files},'scope':'Generic draft after locked diagnostics; no scientific success asserted'}
 write(DEST/'reporting_plan.json',plan)
 for p in files:
  target=DEST/'reporting_code_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
 pid=json.loads((DEST/'analysis_worker_pid.json').read_text())['pid']
 while live(pid):
  assert all(digest(ROOT/p)==h for p,h in plan['code_hashes'].items())
  write(DEST/'reporting_progress.json',{'status':'waiting_for_analysis','analysis_pid':pid,'verified_live':True,'observed_unix':time.time()});time.sleep(20)
 assert json.loads((DEST/'analysis_progress.json').read_text())['status']=='diagnostics_complete'
 from .source_router_case_review import run
 run(DEST)
 from .source_selected_report import main as report
 report(destination=DEST)
 from .router_cap_audit import run as cap_audit
 from .class_coherent_gate_audit import run as class_audit
 for arm in ['OFF','ON']:
  cap_audit(DEST,arm);class_audit(DEST,arm)
 p=DEST/'report.md';s=p.read_text();s+='\n\nDiscovery selects one minimum class-mean CE expert per observed minibatch label, weighting each class by its image count. Original soft balance, detached router cap and all inference remain unchanged. Original-image EDA, common-state comparisons and final scientific interpretation remain pending.\n';p.write_text(s)
 write(DEST/'reporting_progress.json',{'status':'evidence_draft_complete','scientific_goal_achieved':False,'remaining':'manual EDA, controls, tensor audits and interpretation'})
if __name__=='__main__':
 try:main()
 except Exception as exc:write(DEST/'reporting_progress.json',{'status':'failed','error':str(exc)});raise
