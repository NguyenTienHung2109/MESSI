"""Evidence tables for the source-selected frozen-ERM400 comparison."""
import json
import numpy as np
from .source_selected_run import DEST

def main(destination=None):
    DEST=destination if destination is not None else globals()['DEST']
    r=json.loads((DEST/'shrinkage_control/shrinkage_results.json').read_text());b=r['baseline'];on=b['ON']['mechanism'];off=b['OFF']['mechanism']
    cases=json.loads((DEST/'eda/case_summary.json').read_text());bias=json.loads((DEST/'bias_control/results.json').read_text())
    plan=json.loads((DEST/'plan.json').read_text());source=r['splits']['source_out']['methods'];target=r['splits']['target_in']['methods']
    text=['# Source-selected frozen ERM400: evidence report','','Goal:','Determine whether residual discovery adds transferable predictive structure to a fully trained ERM representation, and whether detached binary subset SIRM adds a mechanism/accuracy benefit beyond OFF.','','Stage-1 ERM checkpoint:',
      f"Existing ERM-H2 DeiT-S/16 run completed logged steps 0–500 (501 updates). Reuse its source-validation-best checkpoint at step {plan['anchor_selection']['selected_step']}, SHA256 `{plan['anchor_sha256']}`. Selection uses source-out only; no ERM retraining or target ranking. The selected state is earlier than the completed run endpoint; iteration03 endpoint results remain a separate protocol. All B0/B1/B2 globals share this exact state.",'','Stage-2 configuration:',
      '400 updates: discovery 0–199 and invariance 200–399. Six residual experts, dynamic detached Q every 5 updates, binary eligible-pair SIRM, lambda1 with 50-update ramp (OFF0). Original thresholds, ESS, base losses, H2, fixed10% manifest and seed0 retained. Frozen backbone/global are checked throughout training and at both saved states. Stage-2 selection is equal-source-domain validation every25 plus final399, earliest best in the invariance phase; target evaluation is post-hoc only.','',
      '| Model | Selected Stage-2 step | Executed SIRM steps through selection |','|---|---:|---:|',
      f"| B0 | no Stage2 | 0 |",
      f"| B1 OFF | {b['OFF']['selected_step']} | 0 |",
      f"| B2 ON | {b['ON']['selected_step']} | {r['selected_checkpoint_active_steps']} |",'','B0/B1/B2 accuracy:','',
      '| Split | B0 % | B1 OFF % | B2 ON % | B1−B0 pp | B2−B1 pp |','|---|---:|---:|---:|---:|---:|']
    for split in ['source_out','target_in','target_out']:
        v=[b[m]['splits'][split]['source_mean'] for m in ['ERM_H2','OFF','ON']]
        text.append(f'| {split} | {100*v[0]:.6f} | {100*v[1]:.6f} | {100*v[2]:.6f} | {100*(v[1]-v[0]):+.6f} | {100*(v[2]-v[1]):+.6f} |')
    text+=['',f"Router distillation weight: {plan['hparams']['OFF']['sirm_discover_config']['lambda_router']}. See the locked [preregistration](preregistration.md) for the candidate hypothesis.",'','Source accuracy averages domains equally; target-in/out are the fixed held-out DomainBed subsets. Global-only equals B0 exactly. All comparisons in this table share the same anchor; endpoint results are not substituted.','','Factor discovery result:','',
      '| Arm / split | Full−global pp | Residual mean output cosine | Correction/global RMS | Router NULL | Mean expert residual norm |','|---|---:|---:|---:|---:|---:|']
    if plan['hparams']['OFF']['sirm_discover_config'].get('center_residuals',False):
        text[ text.index('Stage-2 configuration:')+1:text.index('Stage-2 configuration:')+1 ]=['', 'Centering intervention: training subtracts each expert’s differentiable mean over equal-source minibatches; inference uses an equal-domain mean recalculated under current heads from fixed original source-in frozen features. Mean buffers are saved with each checkpoint and restored without target recalibration. Classification, discovery, teacher, utility/Q and SIRM all consume centered delta. Train minibatch/H2 and inference original source-cache distributions differ; inspect this normalization gap alongside the outcome. No global logits change.']
    for split,d in r['splits'].items():
        for arm in ['OFF','ON']:
            v=d['methods'][arm];text.append(f"| {arm}/{split} | {100*v['residual_delta']:+.6f} | {v['residual_cosine_mean']:.6f} | {v['correction_global_RMS']:.6f} | {v['router_NULL']:.6f} | {np.mean(v['residual_norm_per_expert']):.6f} |")
    text+=['','Output similarity and shrinkage quantities pool images; source accuracy/U average domains equally. All pairwise matrices and individual norms are in [raw diagnostics](shrinkage_control/shrinkage_results.json). These metrics do not by themselves establish complementary useful factors.','','Factor transfer result:','',
      '| Arm/expert | Source U | Target-in U | Target-in removal drop pp | Target-out U | Target-out removal drop pp |','|---|---:|---:|---:|---:|---:|']
    for arm in ['OFF','ON']:
        s=np.array(source[arm]['utility_matrix']).mean(0);ti=target[arm];to=r['splits']['target_out']['methods'][arm]
        for e in range(6):text.append(f"| {arm}/E{e} | {s[e]:+.7f} | {ti['utility_matrix'][0][e]:+.7f} | {100*ti['expert_accuracy_drop'][e]:+.6f} | {to['utility_matrix'][0][e]:+.7f} | {100*to['expert_accuracy_drop'][e]:+.6f} |")
    for arm in ['OFF','ON']:
        text+=['',f"{arm} source→target-in CE Spearman: {target[arm]['source_target_CE_spearman']}; target-out: {r['splits']['target_out']['methods'][arm]['source_target_CE_spearman']} (six experts, descriptive)."]
        for label,key in [('Marginal source U','utility_matrix'),('Double-centered source V','utility_interaction')]:
            text+=['',f'{arm}: {label}', '', '| Domain | E0 | E1 | E2 | E3 | E4 | E5 |','|---|---:|---:|---:|---:|---:|---:|']
            for domain,row in zip([0,2,3],source[arm][key]):text.append('| '+str(domain)+' | '+' | '.join(f'{x:+.7f}' for x in row)+' |')
    text+=['','Positive U/removal drop means the expert helps. Removal does not renormalize routing; effects overlap and cannot be summed. Source-selected checkpoint times may differ; check the step table above.','','Q behavior:']
    for arm,mech in [('OFF',off),('ON',on)]:text+=['',f"{arm}: Q={json.dumps(mech['Q'])}; active={mech['active_steps']}/{mech['stage3_steps']} ({100*mech['active_fraction']:.4f}%); first nonzero={mech['first_nonzero_step']}; no-shared refreshes={mech['no_shared_refreshes']}; ESS-blocked refreshes={mech['ESS_blocked_refreshes']}."]
    text+=['','SIRM mechanism result:',f"ON active-refresh residual-parameter gradient ratios: {json.dumps(on['gradient_ratio'])}. These sample Q refreshes, not all optimizer steps. Undefined values with n=0 are not measured zeros.",
      f"Common executed-cell-mask source G: OFF={source['OFF']['G_common_executed']}, ON={source['ON']['G_common_executed']}. Empty masks give undefined G; per-cell G is available in raw diagnostics. G is absolute domain-mean scale derivative, not squared minibatch loss or proof of causal invariance.",'','Shrinkage analysis:',
      f"Source-RMS matched alpha={r['alpha_source_matched']:.8f}. {r['status']}: {r['reason']}",'',
      '| Split | Global CE | OFF CE | ON CE | Source-matched scaled OFF CE |','|---|---:|---:|---:|---:|']
    for split,d in r['splits'].items():text.append('| '+split+' | '+' | '.join(f"{d['methods'][m]['CE']:.7f}" for m in ['OFF_global_only','OFF','ON','OFF_source_matched'])+' |')
    text+=['','CE in this table pools source images; [bias control](bias_control/results.json) instead averages source-domain CE equally. Target weighting matches. Both controls are source-calibrated, with no target-selected scale/vector.','','Matched-case and bias-control evidence:','',
      '| Split | B1 repairs B0 | B1 harms B0 | B2 repairs B1 | B2 harms B1 | Constant bias accuracy % | Centered instance accuracy % |','|---|---:|---:|---:|---:|---:|---:|']
    for split,c in cases.items():
        t=c['transitions'];v=bias['splits'][split]['scores'];text.append(f"| {split} | {t['B1_repairs_global']} | {t['B1_harms_global']} | {t['B2_repairs_B1']} | {t['B2_harms_B1']} | {100*v['constant_bias']['accuracy']:.6f} | {100*v['centered_instance']['accuracy']:.6f} |")
    text+=['','All images, class scores, confusion matrices and contact sheets are in [case EDA](eda/case_summary.json); [source utility halves](bias_control/source_utility_stability.json) are descriptive, not independent validation. Bias calibration-domain holdouts exclude only calibration data; the residual bank was trained on all three sources. Prior target EDA makes these exploratory.','','Primary bottleneck:','Pending joint interpretation of these metrics and original-image EDA; do not infer a factor mechanism from accuracy/cosine/G alone.','','Hypothesis status:','Scientific verdict pending manual matched-image review and control assessment. No success is claimed by generation of this report.','','Recommended next change:','No method modification until this corrected experiment and EDA have been analyzed. Keep target scores out of checkpoint/hyperparameter selection; use source-only reliability and matched controls to preregister any next hypothesis.']
    alignment=DEST/'source_router_alignment.json'
    if alignment.exists():
        a=json.loads(alignment.read_text());text+=['','Source routing diagnostic:','',a['scope'],'',
          '| Domain | Actual CE | Label-teacher CE | Actual NULL | Teacher NULL | Negative utility mass | Teacher KL |','|---|---:|---:|---:|---:|---:|---:|']
        for d,v in a['per_domain'].items():text.append(f"| {d} | {v['actual']['CE']:.7f} | {v['label_teacher_diagnostic']['CE']:.7f} | {v['router_NULL']:.6f} | {v['teacher_NULL']:.6f} | {v['router_negative_utility_mass']:.6f} | {v['teacher_KL']:.6f} |")
        if 'fixed_baseline_teacher_and_head_controls' in a:
            text+=['','Fixed iteration04 teacher and matched-head source controls:','','| Domain | Old router KL | New router KL | Old negative-utility mass | New mass on old negative terms | New router / old heads CE | Old router / new heads CE |','|---|---:|---:|---:|---:|---:|---:|']
            for d,v in a['fixed_baseline_teacher_and_head_controls'].items():text.append(f"| {d} | {v['baseline_router_fixed_teacher_KL']:.6f} | {v['new_router_fixed_teacher_KL']:.6f} | {v['baseline_negative_utility_mass']:.6f} | {v['new_router_baseline_negative_utility_mass']:.6f} | {v['new_router_old_heads']['CE']:.7f} | {v['old_router_new_heads']['CE']:.7f} |")
            text+=['','These interventions hold cached heads/routing fixed for attribution; they are diagnostic mixtures, not retrained or checkpoint-selected models. The label-derived teacher is not deployable.']
    (DEST/'report.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':main()
