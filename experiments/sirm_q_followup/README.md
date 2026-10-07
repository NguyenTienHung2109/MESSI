# Q / IRMv1 follow-up

`diagnose.py` evaluates all 48 source-selected factorial checkpoints from existing deterministic caches, without training. It verifies cache checkpoint/split identity and physical source/target domains. `scalar_cells` computes the exact derivative of cross entropy at scale=1 in float64 and is tested against autograd and the production penalty. Both unweighted domain risk and detached responsibility-weighted risk are exported, with ESS, validity, Q membership and actual full-domain penalty eligibility kept separate.

Source/target competence is compared across six experts using source-out CE averaged equally across source domains. Saved source risk-EMA ranks are a separate diagnostic (`rho_ema`), since this is the statistic actually used by Q. Target loss gap averages each source Q row's selected/non-selected target CE gap. No target labels fit or select any object. Best checkpoints are late; these results do not supply pre-activation measurements.

```bash
/home/ntduc/miniconda3/envs/dg/bin/python -m experiments.sirm_q_followup.diagnose
/home/ntduc/miniconda3/envs/dg/bin/python -m unittest -q experiments.sirm_q_followup.test_diagnostics
/home/ntduc/miniconda3/envs/dg/bin/python -u -m experiments.sirm_q_followup.pilot suite --smoke --output results/sirm_q_followup/smoke_fixed_random
/home/ntduc/miniconda3/envs/dg/bin/python -u -m experiments.sirm_q_followup.pilot suite --output results/sirm_q_followup/pilot
```

Pilot: Cartoon and Sketch × seeds 0/1/2 × OFF / learned / random / global, 5001 steps, existing 80/20 protocol. Reuse verified B/D; 12 new training jobs. Random is a fixed feasible three-pair graph on three randomly chosen expert slots, independent of competence, normalized 0.5 weights, exactly two per source and two sources per active expert. It is not an unconstrained independent per-row draw. Learned-Q can differ in active capacity, weights and churn, so this control does not isolate assignment criterion alone.

Only penalty Q changes. Router distillation stays on learned Q to isolate the IRM constraint; changing its teacher would test a different system. Global uses all-six normalized Q but retains native routing weights, ESS checks and two-source eligibility. Thus global-Q means all cells are eligible by Q, not that every expert receives actual nonzero penalty on every minibatch. Actual penalty Q and valid cells are saved in `penalty_calls.jsonl` (includes gradient diagnostic calls, not one record per step).

All four arms are trained and matched on initialization, data split, minibatch schedule, pretrained weights and capacity before target evaluation. Checkpoints are locked from source accuracy. The suite writes progress/failures, per-target seed mean/SD, paired deltas and paired bootstrap intervals. Three seed blocks give limited uncertainty precision. The fixed protocol is exploratory after earlier PACS target inspection.

The first `smoke/` directory tested a preliminary random expert permutation control. `smoke_fixed_random/` tests the final competence-independent fixed-pair control; only the latter authorizes full training. Smoke accuracy is not an experimental result.
