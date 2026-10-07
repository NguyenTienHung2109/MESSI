"""Numerical evidence watcher; original-image EDA remains an explicit pending step."""
import json,os,time
from pathlib import Path
from .class_balanced_run import DEST,PRIOR,ROOT,ARMS
from .router_weight import live
from .shrinkage import digest,write

def draft():
 from .class_balanced_audit import run as weighting
 from .router_cap_audit import run as cap
 from .class_balanced_pair_audit import main as pair
 from .class_balanced_source_comparison import run as source
 for arm in ARMS:weighting(DEST,arm,complete=True);cap(DEST,arm)
 pair_result=pair();source_result=source();assert json.loads((DEST/'original_control_reuse_audit.json').read_text())['passed']
 r=json.loads((DEST/'shrinkage_control/shrinkage_results.json').read_text());b=r['baseline'];control=json.loads((PRIOR/'experiment/runs/L38/OFF/posthoc/complete.json').read_text());p=json.loads((DEST/'plan.json').read_text())
 s=['# Iteration14: equal-class discovery under matched raw gradient strength','','Numerical evidence draft; manual original-image EDA and final scientific verdict remain pending. No scientific success is inferred.','','Goal:','Test source-derived class-allocation hypothesis without increasing discovery raw gradient norm, then test any genuinely eligible subset invariance.','','Stage-1 ERM checkpoint:',f"Same completed501-update source-selected375 ERM-H2, SHA{p['anchor_sha256']}; frozen original CLS backbone/global, no ERM retraining.",'','Stage-2 configuration:','Fresh equal-class OFF/ON400 updates0–399, discovery0–199 and invariance200–399; existing spatial768 residual/router bank, same learned initialization and source schedules as original-weighting control. Equal-present-class discovery uses a detached raw residual-gradient norm match to original count/N, lambda_discover0.1 unchanged. Class winners, soft balance, centering, router cap, fixed normalization, Q refresh5/floors.002/temp.05/ESS2/componentmin2/EMAminimum32, ONlambda1/ramp50 unchanged. Source-only invariance selection25/final399; target deferred.','','B0/B1/B2 accuracy:','','| Split | ERM % | Original weighting OFF % | Equal-class OFF % | Equal-class ON % | Discovery Δ versus ERM pp | ON−OFF pp |','|---|---:|---:|---:|---:|---:|---:|']
 for split in ['source_out','target_in','target_out']:
  vals=[b['ERM_H2']['splits'][split]['source_mean'],control['splits'][split]['source_mean'],b['OFF']['splits'][split]['source_mean'],b['ON']['splits'][split]['source_mean']];s.append('| '+split+' | '+' | '.join(f'{100*x:.6f}' for x in vals)+f' | {100*(vals[2]-vals[0]):+.6f} | {100*(vals[3]-vals[2]):+.6f} |')
 s+=['','Source scores average domains equally; target scores are fixed10% in/test and out/validation. Selected states may differ; common399 source comparison is separate.','','Factor discovery result:','','| Bank/state | Source % | Repairs | Harms | Unscaled expert capacity /70 |','|---|---:|---:|---:|---:|']
 for key,v in source_result.items():s.append(f"| {key} | {100*v['accuracy']:.6f} | {v['repairs']} | {v['harms']} | {v['unscaled_single_expert_capacity']} |")
 s+=['','Capacity uses true labels and unscaled individual expert corrections; it is not deployable performance. Full per-class source groups and margin diagnostics: [common-state source replay](source_common_state/results.json).','','Factor transfer result:','Per-expert target removal utility, source→target six-expert CE correlation and cosine matrices are in [locked replay](shrinkage_control/shrinkage_results.json). Target effects are exploratory, not candidate-selection signals.','','Q behavior:']
 for arm in ARMS:s+=['',arm+': '+json.dumps(b[arm]['mechanism'],indent=2)]
 s+=['','SIRM mechanism result:',json.dumps(pair_result,indent=2),'','Only genuine eligible exposure with verified parameter/logit divergence can test an active penalty. If masks are empty, fixed-mask G and gradient ratios are undefined, not evidence of successful invariance. Actual direct logits and image cases still require manual review.','','Shrinkage analysis:',f"Source-RMS matched alpha={r['alpha_source_matched']}; {r['status']}: {r['reason']}",'[Constant-bias and input-dependent correction controls](bias_control/results.json). Norm matching controls the raw discovery gradient, not Adam updates or residual magnitude.','','Primary bottleneck:','Pending original-image EDA and replicated per-class/domain source utility analysis; do not infer a cause from aggregate accuracy alone.','','Hypothesis status:','Local source gradient and finite perturbation evidence motivated this run. Full training can falsify that hypothesis despite favorable local diagnostics. Manual EDA and final interpretation pending.','','Recommended next change:','No further method change before analysis completion. Preserve thresholds/lambda and source-only selection; derive any next hypothesis from source cases and previous metrics, with explicit rejection conditions. A single repeatedly inspected target cannot establish convincing ERM superiority.','','[Main OFF/ON cases](eda/case_summary.json); [original versus equal weighting cases](matched_weighting_control/eda/case_summary.json). In the latter B1/B2 mean original OFF/equal-class OFF, not SIRM OFF/ON.']
 (DEST/'report_draft.md').write_text('\n'.join(s)+'\n')

def main():
 paths=[Path(__file__)]+[Path(__file__).with_name(x) for x in ['class_balanced_audit.py','class_balanced_pair_audit.py','class_balanced_source_comparison.py','router_cap_audit.py','protocol_audit.py']];hashes={str(p.relative_to(ROOT)):digest(p) for p in paths};write(DEST/'reporting_plan.json',{'code_hashes':hashes,'scope':'Independent class-weight formula/norm/frozen audits, tensors, source common399 and numerical draft; manual visual EDA required.'})
 for p in paths:target=DEST/'reporting_code_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
 pid=json.loads((DEST/'analysis_worker_pid.json').read_text())['pid']
 while live(pid):
  assert all(digest(ROOT/p)==h for p,h in hashes.items());write(DEST/'reporting_progress.json',{'status':'waiting_for_analysis','verified_live':True,'analysis_pid':pid,'observed_unix':time.time()});time.sleep(20)
 assert json.loads((DEST/'analysis_progress.json').read_text())['status']=='diagnostics_complete';assert all(digest(ROOT/p)==h for p,h in hashes.items());draft();write(DEST/'reporting_progress.json',{'status':'evidence_draft_complete','scientific_goal_achieved':False,'remaining':'Manual source/target original-image EDA, utility reliability and final hypothesis verdict; no next method yet.'})
if __name__=='__main__':
 write(DEST/'reporting_worker_pid.json',{'pid':os.getpid(),'started_unix':time.time(),'module':'experiments.sirm_research_loop.class_balanced_report'})
 try:main()
 except Exception as exc:write(DEST/'reporting_progress.json',{'status':'failed','error':str(exc)});raise
