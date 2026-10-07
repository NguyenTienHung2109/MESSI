"""Evidence draft watcher; visual EDA and final scientific verdict remain manual."""
import json,time
from pathlib import Path
from .spatial_bank_run import DEST,ROOT,ARMS
from .router_weight import live
from .shrinkage import digest,write

def draft():
    from .spatial_source_capacity import run
    capacity=run()
    from .router_cap_audit import run as cap
    from .class_coherent_gate_audit import run as coherent
    for arm in ARMS:cap(DEST,arm);coherent(DEST,arm)
    p=json.loads((DEST/'plan.json').read_text());r=json.loads((DEST/'shrinkage_control/shrinkage_results.json').read_text());b=r['baseline'];ctrl=json.loads((DEST/'experiment/runs/L38/CLS_CONTROL/posthoc/complete.json').read_text())
    s=['# Iteration13: matched spatial-bank evidence draft','','Goal:','Determine whether spatial features supply complementary transferable residual structure beyond matched CLS capacity, and whether dynamic binary SIRM adds benefit beyond spatial OFF. Scientific success remains unproven.','','Stage-1 ERM checkpoint:',f"Completed501 ERM-H2 updates; source-selected375 SHA {p['anchor_sha256']}. Same frozen DeiT-S/16 backbone/global for all arms. No ERM retraining.",'','Stage-2 configuration:','CLS_CONTROL repeated CLS OFF, spatial OFF, spatial ON; each400 updates (discovery0–199, invariance200–399). Bank768/global384; trainable bank1,201,987. Fixed original source-in normalization; class-coherent discovery/centering/router cap retained. Q every5, binary eligibility, unchanged thresholds/ESS/minimum2. ON lambda1/ramp50, OFF0. Source-selected checkpoints every25/final399; target deferred. See preregistration and code snapshots.','','B0/B1/B2 accuracy:','','| Split | B0 % | CLS control % | Spatial OFF % | Spatial ON % | Spatial−CLS pp | ON−OFF pp |','|---|---:|---:|---:|---:|---:|---:|']
    for split in ['source_out','target_in','target_out']:
        vals=[b['ERM_H2']['splits'][split]['source_mean'],ctrl['splits'][split]['source_mean'],b['OFF']['splits'][split]['source_mean'],b['ON']['splits'][split]['source_mean']]
        s.append('| '+split+' | '+' | '.join(f'{100*v:.6f}' for v in vals)+f' | {100*(vals[2]-vals[1]):+.6f} | {100*(vals[3]-vals[2]):+.6f} |')
    s+=['','Source accuracy averages domains equally; target-in/out use fixed subsets. Each bank uses independently locked earliest-best invariance checkpoint; endpoint/control selection effects need joint review.','','Factor discovery result:',json.dumps(capacity,indent=2),'','Oracle capacity uses true labels and unscaled individual experts; actual repairs/harms remain decisive. Class/domain groups require at least2 images. [Per-image source cases](source_spatial_capacity/cases.csv).','','Factor transfer result:','Per-expert marginal CE utility, double-centered source interaction, target leave-one-expert-out effects, source→target correlations and all output cosine matrices: [locked replay](shrinkage_control/shrinkage_results.json). Positive removal drop means removal hurts; effects overlap. Target diagnostics are exploratory and do not select candidates.','','Q behavior:']
    for arm in ['OFF','ON']:s += ['',arm+': '+json.dumps(b[arm]['mechanism'],indent=2)]
    s+=['','SIRM mechanism result:','Active-refresh gradients and executed support are above. Fixed common-mask G and OFF/ON comparisons are in locked replay; absent shared support/empty masks do not test invariance. Check exact tensor/trajectory divergence before inferring a mechanism.','','Shrinkage analysis:',f"Source-RMS matched alpha={r['alpha_source_matched']}; {r['status']}: {r['reason']}",'[Source-calibrated bias and utility reliability](bias_control/results.json). Lower G, smaller corrections or lower cosine alone are not evidence of useful invariant factors.','','Primary bottleneck:','Pending original-image EDA and parameter-matched source/control analysis.','','Hypothesis status:','Experiment and locked numerical diagnostics complete; manual visual EDA, final protocol/tensor audit and causal interpretation pending. No success or failure of spatial factor discovery is asserted by this draft.','','Recommended next change:','No method change until all current evidence and EDA have been analyzed. Any next hypothesis must follow source evidence; target outcomes cannot select the next candidate. Convincing ERM superiority requires independent locked seed/domain replication.','','[Spatial OFF/ON cases](eda/case_summary.json); [CLS control versus spatial OFF cases](matched_CLS_control/eda/case_summary.json). B1/B2 labels in the latter mean CLS OFF/spatial OFF, not SIRM OFF/ON.']
    (DEST/'report_draft.md').write_text('\n'.join(s)+'\n')

def main():
    paths=[Path(__file__),Path(__file__).with_name('spatial_source_capacity.py'),Path(__file__).with_name('router_cap_audit.py'),Path(__file__).with_name('class_coherent_gate_audit.py')]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths};write(DEST/'reporting_plan.json',dict(code_hashes=hashes,scope='Draft evidence and source-only capacity; manual visual EDA still required'))
    for p in paths:
        target=DEST/'reporting_code_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    pid=json.loads((DEST/'analysis_worker_pid.json').read_text())['pid']
    while live(pid):
        assert all(digest(ROOT/p)==h for p,h in hashes.items());write(DEST/'reporting_progress.json',dict(status='waiting_for_analysis',analysis_pid=pid,verified_live=True,observed_unix=time.time()));time.sleep(20)
    assert json.loads((DEST/'analysis_progress.json').read_text())['status']=='diagnostics_complete'
    assert all(digest(ROOT/p)==h for p,h in hashes.items());draft()
    write(DEST/'reporting_progress.json',dict(status='evidence_draft_complete',scientific_goal_achieved=False,remaining='Visual original-case EDA, common-state control/tensor/support audits, final interpretation and next source hypothesis'))

if __name__=='__main__':
    try:main()
    except Exception as exc:write(DEST/'reporting_progress.json',dict(status='failed',error=str(exc)));raise
