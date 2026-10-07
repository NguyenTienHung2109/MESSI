# Binary-SIRM five-part report

Read-only monitoring and post-hoc analysis for the existing Terra L38 seed-0
R0 OFF / R1 binary ON pair. This module is outside the experiment's locked
training source paths. It does not alter parameters, thresholds, lambda,
normalization, schedules, selection cadence or discovery.

```bash
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/ntduc/miniconda3/envs/dg/bin/python -u \
  -m experiments.sirm_discover_binary_analysis.run --watch
```

While training runs, only completed diagnostic records are read. The watcher
waits for the paired runner's completed state before loading source-selected
models or doing additional GPU inference. Outputs are under the pair's
`analysis/` directory:

- `summary.md`, `five_results.json`: the five requested result groups.
- `gradient_ratio_history.csv`: weighted SIRM/base residual gradient ratios.
- `Q_evolution.csv`, `Q_feedback_comparison.csv`: full source Q/EMA matrices,
  selected/shared/eligible/executed counts and matched R0/R1 differences.
- `R0_decomposition.json`, `R1_decomposition.json`, `decomposition.csv`:
  source-out and target residual cosine, norms, correction/global RMS and NULL.
- `common_mask_G.json`: domain-mean absolute scale derivatives for both arms
  with the same R1-selected mask and the union of cells actually exposed to IRM.

Gradient norms in the locked training code are measured on residual parameters
at **active Q refreshes**, every 50 steps. Median/mean/p90/max are therefore
sampled refresh statistics, not all-optimizer-step statistics. The active-only
distribution and all-refresh distribution (inactive refreshes have zero penalty)
are shown separately. Missing active measurements remain unknown. The weighted
norm already includes the current ramped lambda; it is not multiplied again.
SIRM active fraction, however, uses every Stage-3 optimizer-step record.

Residual cosine compares class-centered expert output logits pooled over images
and classes. It does not prove causal factor recovery. Router NULL is measured
on the stated image split; Q NULL always describes source-domain assignment.
The same saved Q masks are used for G comparisons so feedback-driven mask
changes do not masquerade as improvement. These derivatives use full domain
means and are distinct from mean squared minibatch IRMv1 loss. Checkpoints are
independently source-selected and may come from different steps; source-out
scores are selection data, not untouched validation.

A lower R1 target accuracy together with dominant gradient ratios is consistent
with excessive effective regularization after removing attenuation. It alone
does not establish that binary gating or residual IRMv1 is intrinsically bad.
No automatic tuning or additional training is triggered by this report.

## User-requested reuse of R0

After the duplicate DomainBed R0 reproduced historical R0 exactly through step
3750, the user requested stopping the duplicate and starting R1. The duplicate
is archived at `banks/R0_domainbed_partial/`. The completed historical R0,
including all 5001 step records and its original source-selected checkpoint,
is the control. `reuse_r0.py` verifies initialization, splits, complete source
batch schedules, partial all-step diagnostics and source validation before
reuse. Historical checkpoint tensor contents are verified exactly after adding
the DomainBed `model.` key prefix; the historical training configuration is
preserved. Original inputs and the original locked plan remain unchanged.

The replacement controller runs only fresh R1 with the original locked command
and performs the existing paired audit/target evaluation afterward. It records
reuse in `R0_reuse_plan.json` and in the reports. R0 has historical source-score
logs; the stopped partial R0 and fresh R1 have DomainBed per-domain in/out logs.

```bash
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/ntduc/miniconda3/envs/dg/bin/python -u \
  -m experiments.sirm_discover_binary_analysis.reuse_r0
```

This command requires the recorded user-requested stop and an absent R1 output;
it does not resume R1 from weights or retrain R0.

## DomainBed scores during training

The watcher also updates `DomainBed_scores.md`/JSON and a complete-score-only
`DomainBed_results.jsonl` projection. The local `OracleSelectionMethod` is
called directly: it takes the last complete target-scored checkpoint and
returns target-out validation / target-in test accuracy. It does not choose
the best checkpoint across steps. Best target-validation checkpoints across
all stages and within Stage 3 are separately labeled post-hoc diagnostics.
Extra source-only Q-refresh rows lack target scores and are excluded from this
projection. Target 80/20 in/out scores differ from source-selected full-target
accuracy. No oracle report changes the locked training settings or source-only
checkpoint selection.
