"""Assemble the locked frozen-ERM400 report after matched-case EDA."""
import json
from pathlib import Path
import numpy as np
from .anchor_run import DEST, EXPERIMENT

def read(p):return json.loads(p.read_text())

def main():
    control=read(DEST/'shrinkage_control/shrinkage_results.json')
    reports=control['baseline'];eda=read(DEST/'eda/case_summary.json')
    assert read(DEST/'eda_progress.json')['status'] in ['case_replay_complete','completed']
    assert (DEST/'eda/analysis.md').exists()
    lines=['# Frozen ERM400: factor discovery and matched-error analysis','',
      'Goal:','Test whether residual factors add transferable information on the exact same fully trained, frozen ERM-H2 backbone/global head, and whether binary dynamic-Q SIRM adds an effect beyond OFF.','',
      'Stage-1 ERM checkpoint:',
      'Existing Terra L38 seed-0 DeiT-S/16 ERM-H2 `last.pkl`, SHA256 `049d72af652dda9784130b0aad228caa817b82342f6b74dad79ec69dbdd07bfc`. Logged step 500 corresponds to 501 optimizer updates (legacy inclusive numbering). No ERM retraining. This common anchor is the predeclared fixed full-budget endpoint, not the earlier source-best ERM step 375. This is a selection-policy limitation relative to the requested source-validation protocol; it is disclosed rather than retroactively changing the anchor after target inspection. Stage-2 checkpoints are selected only by equal-domain source validation.','',
      'Stage-2 configuration:',
      'Fixed stratified 10% subset, seed 0, held-out env1/location_38. Exactly 400 updates: discovery 0–199; invariance 200–399. Six residual experts; detached dynamic Q every 5 updates; binary eligible-pair penalty; ON lambda=1 with 50-update ramp; OFF lambda=0. Original U/V floors 0.002, Q temperature 0.05, ESS 2 and all other base settings unchanged. Source selection every 25 updates plus final 399. Both arms select step 375. Backbone/global requires_grad=False and tensor equality are verified at initialization, each update, selected and final states. [Control audit](experiment/runs/L38/control_audit.json); [ON frozen-anchor audit](experiment/runs/L38/ON/strong_anchor_audit.json).','',
      'B0/B1/B2 accuracy:','',
      '| Split | B0 frozen ERM % | B1 OFF % | B2 ON % | B1−B0 pp | B2−B1 pp |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for split in ['source_out','target_in','target_out']:
        a=[reports[m]['splits'][split]['source_mean'] for m in ['ERM_H2','OFF','ON']]
        lines.append(f'| {split} | {100*a[0]:.6f} | {100*a[1]:.6f} | {100*a[2]:.6f} | {100*(a[1]-a[0]):+.6f} | {100*(a[2]-a[1]):+.6f} |')
    lines+=['','Source-out averages domains equally; target rows use fixed DomainBed in/out image subsets. All B1/B2 global-only accuracies equal B0 exactly. Earlier source-best ERM target-in accuracy is 43.222506%, which is higher than this endpoint B0 and B1/B2: this experiment does not establish superiority over the strongest source-selected ERM.','',
      'Factor discovery result:',
      'OFF improves source validation +1.698756 pp and target-in +0.639386 pp. There is a small positive corrective effect over the endpoint anchor, but no demonstrated complementary factor mechanism. Source-out residual repairs 7 cases and harms 2; target-in repairs 10 and harms 5; target-out repairs 3 and harms 2. Every target repair is raccoon→opossum. The opossum-minus-raccoon residual correction is positive on every image in all three splits. Mean residual output cosine is 0.675168 source-out and 0.705280 target-in, suggesting substantial shared output direction. [All expert norms/cosine matrices/U/V](shrinkage_control/shrinkage_results.json); [matched-case EDA and falsifiable hypotheses](eda/analysis.md).','',
      'Factor transfer result:',
      'Target-in CE worsens 2.131363→2.146689 despite five net correct images; target-out CE worsens 2.054545→2.062296 despite one net correct image. E4 is target CE-positive, but E0 has positive mean source utility and negative target utility. Source↔target CE Spearman over six experts: 0.142857 target-in and 0.771429 target-out; small, class-dependent cohorts make neither a reliability guarantee.','',
      '| Expert | Equal-domain source U | Target-in U | Target-in removal accuracy drop pp | Target-out U | Target-out removal accuracy drop pp |',
      '|---|---:|---:|---:|---:|---:|']
    src=control['splits']['source_out']['methods']['OFF'];ti=control['splits']['target_in']['methods']['OFF'];to=control['splits']['target_out']['methods']['OFF']
    for e in range(6):
        lines.append(f"| E{e} | {np.array(src['utility_matrix']).mean(0)[e]:+.7f} | {ti['utility_matrix'][0][e]:+.7f} | {100*ti['expert_accuracy_drop'][e]:+.6f} | {to['utility_matrix'][0][e]:+.7f} | {100*to['expert_accuracy_drop'][e]:+.6f} |")
    lines+=['','Positive U/drop means the expert helps; negative means removal helps. Removal keeps remaining routing fixed; effects overlap. Full three-domain U and double-centered V matrices are preserved in the linked raw diagnostic, rather than replacing them with equal-domain averages.','',
      'Q behavior:',
      '40 invariance refreshes; four distinct Q matrices; three weight changes and two support changes; 0–2 selected cells; zero shared experts/eligible pairs at every refresh. SIRM active fraction 0/200 = 0%. Shared support exists at 13/40 refreshes under U-floor alone but 0/40 after the existing V-floor condition; this structural gate diagnostic is not a trained threshold ablation. ESS is not the observed blocker. [Gate decomposition](OFF_Q_gate_diagnosis.json); [refresh history](experiment/runs/L38/ON/posthoc/Q_refresh_history.json).','',
      'SIRM mechanism result:',
      'No active SIRM exposure occurs, including before the source-selected checkpoint. Active-refresh gradient ratios have n=0: median/mean/p90/max are undefined, not measured zeros. Every executed-mask entry is false, so common-executed-mask G is undefined. Per-cell G is saved but cannot measure reduction from an unexecuted treatment. OFF/ON prediction tensors are exactly equal (22,279,757 elements) at selected and last checkpoints; matched target logits/predictions also coincide. This is a verified no-op, not evidence for or against an active IRMv1 objective. [Tensor comparison](OFF_ON_parameter_comparison.json).','',
      'Shrinkage analysis:',
      f"Source-RMS-matched ON/OFF alpha={control['alpha_source_matched']:.6f}; ON and OFF are identical. Thus shrinkage is INCONCLUSIVE as an explanation of ON benefit: no ON benefit exists. Relative to B0, residual composition remains nonzero; this is not SIRM-induced contraction.", '',
      '| Split | Mean residual norm | Correction/global RMS | Router NULL | Residual output cosine |',
      '|---|---:|---:|---:|---:|']
    for split,d in control['splits'].items():
        v=d['methods']['OFF'];lines.append(f"| {split} | {np.mean(v['residual_norm_per_expert']):.6f} | {v['correction_global_RMS']:.6f} | {v['router_NULL']:.6f} | {v['residual_cosine_mean']:.6f} |")
    lines+=['','Primary bottleneck:',
      'Two separate issues remain: (1) the current Q interaction condition yields no shared eligible subset on this strong-anchor trajectory, preventing any SIRM mechanism test; (2) the small factor gain is concentrated in one confusion pair and may mostly reflect a shared class-direction correction. Frozen global training fixes the early-freeze comparison confound, but not checkpoint-policy differences or discovery semantics.','',
      'Hypothesis status:',
      'PARTIALLY SUPPORTED for a small residual corrective gain over the fixed endpoint; UNPROVEN for complementary transferable factors; UNTESTED for active subset invariance and ON improvement. Convincing superiority over source-selected ERM is NOT ACHIEVED. The broad research goal remains open. Target-informed EDA hypotheses require fresh source-only development and independent confirmation.','',
      'Recommended next change:',
      'Do not change lambda. First run a preregistered source-only constant-logit correction control and source leave-one-domain-out reliability diagnostics for shared U. These distinguish common class bias from instance-dependent factor benefit and determine whether Q misses reliable shared utility. Only after that analysis should one specific discovery/Q hypothesis be changed. Target accuracy must not choose bias vectors, thresholds, checkpoints or candidate settings. See [EDA hypotheses with counterevidence and falsification](eda/analysis.md).']
    (DEST/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
