"""Evidence draft after completed pair analysis; manual EDA/verdict still required."""
import json,time
from pathlib import Path
from .router_weight import live
from .router_cap_run import DEST,ROOT
from .shrinkage import digest,write

def main():
 files=[Path(__file__),Path(__file__).with_name('source_selected_report.py'),Path(__file__).with_name('source_router_case_review.py')];plan={'code_hashes':{str(p.relative_to(ROOT)):digest(p) for p in files},'scope':'Evidence draft, not final scientific verdict. Wait for verified analysis process termination and completed diagnostics.'};write(DEST/'reporting_plan.json',plan)
 for p in files:
  target=DEST/'reporting_code_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
 pid=json.loads((DEST/'analysis_worker_pid.json').read_text())['pid']
 while live(pid):
  assert all(digest(ROOT/p)==h for p,h in plan['code_hashes'].items());write(DEST/'reporting_progress.json',{'status':'waiting_for_analysis','analysis_pid':pid,'verified_live':True,'observed_unix':time.time()});time.sleep(20)
 assert json.loads((DEST/'analysis_progress.json').read_text())['status']=='diagnostics_complete'
 assert all(digest(ROOT/p)==h for p,h in plan['code_hashes'].items())
 from .source_router_case_review import run
 run(DEST)
 from .source_selected_report import main as report
 report(destination=DEST)
 summaries={}
 for arm in ['OFF','ON']:
  bank=DEST/'experiment/runs/L38'/arm;rows=[json.loads(l) for l in (bank/'diagnostics.jsonl').read_text().splitlines()];assert [r['step'] for r in rows]==list(range(400));errors=[abs(r['discovery_matched_parameter_grad_norm']-r['discovery_hard_parameter_grad_norm'])/max(r['discovery_hard_parameter_grad_norm'],1e-20) for r in rows];assert max(errors)<1e-5;summaries[arm]={'updates':len(rows),'norm_error_max':max(errors),'scale_min':min(r['discovery_detached_scale'] for r in rows),'scale_max':max(r['discovery_detached_scale'] for r in rows),'no_pair_fallbacks':sum(r['discovery_no_pair_fallback'] for r in rows),'zero_pair_gradient_fallbacks':sum(r['discovery_zero_pair_gradient_fallback'] for r in rows),'active_steps':sum(r['sirm_active'] for r in rows)}
 write(DEST/'discovery_norm_control_audit.json',{'scope':'All400 logged same-state residual-parameter discovery norm matches; not cross-run,total-base or cumulative update matching.','arms':summaries,'scientific_goal_achieved':False})
 write(DEST/'router_norm_control_audit.json',router_audit(DEST));p=DEST/'report.md';text=p.read_text();text=text.replace('Stage-2 configuration:', 'Stage-2 configuration:\n\nPair-shared discovery intervention: within each class, compare source-domain pairs using the same expert\'s worst-endpoint excess CE relative to frozen global, then average pairs/classes. Discovery gradient norm is matched each minibatch to per-image hard discovery at the current state with a detached scale. Existing soft balance, router teacher, centering,lambda,Q and ESS remain unchanged. Norm matching does not establish equal total gradients, optimizer history or cumulative strength. No-pair and zero-pair-gradient fallbacks are logged.');text+='\n\nRouter auxiliary norm is capped per minibatch at the same-state classification router norm, with detached scale and nominal lambda_router unchanged. This excludes SIRM gradients and does not bound total gradients or Adam updates. See [router audit](router_norm_control_audit.json).\n\nAll-step discovery norm audit: '+json.dumps(summaries)+'. See [raw audit](discovery_norm_control_audit.json). Manual original-image review and joint scientific interpretation remain pending.\n';p.write_text(text);write(DEST/'reporting_progress.json',{'status':'evidence_draft_complete','remaining':'Manual EDA, tensor comparisons, fixed-mask/strength/bias/shrinkage interpretation, scientific verdict and completion audit','scientific_goal_achieved':False})

def router_audit(dest):
 summaries={}
 for arm in ['OFF','ON']:
  rows=[json.loads(l) for l in (dest/'experiment/runs/L38'/arm/'diagnostics.jsonl').read_text().splitlines()];assert len(rows)==400
  assert all(0<=r['router_aux_detached_scale']<=1 and r['router_aux_post_cap_parameter_grad_norm']<=r['router_cls_parameter_grad_norm']*1.00001+1e-10 for r in rows)
  summaries[arm]={'updates':len(rows),'capped_updates':sum(r['router_aux_capped'] for r in rows),'zero_classification_norm_updates':sum(r['router_cls_parameter_grad_norm']==0 for r in rows),'scale_min':min(r['router_aux_detached_scale'] for r in rows),'scale_max':max(r['router_aux_detached_scale'] for r in rows)}
 return {'scope':'All logged same-state router-only gradients; not total gradients or Adam updates.','arms':summaries}

if __name__=='__main__':
 try:main()
 except Exception as exc:write(DEST/'reporting_progress.json',{'status':'failed','error':str(exc)});raise
