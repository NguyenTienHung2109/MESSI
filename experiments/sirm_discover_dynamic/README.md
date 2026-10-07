# Terra L38 dynamic-Q behavioral test

Seed 0, R0 SIRM OFF and R1 ON. The original global/discovery/invariance budgets (1000/1000/3001), ramp, H2 augmentation, residual architecture, discovery objective, temperatures, thresholds, utility EMA, routing ESS and pair-weighted IRMv1 formula remain unchanged. Q refreshes every 50 steps throughout Stage 3 and is detached. The previous fixed-Q experiment remains an historical result; its runner explicitly requests freezing, while GateSIRM now defaults to dynamic Q.

```bash
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 /home/ntduc/miniconda3/envs/dg/bin/python -m unittest experiments.sirm_discover.test_discover experiments.sirm_discover_gate.test_gate experiments.sirm_discover_dynamic.test_dynamic
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled MPLCONFIGDIR=/tmp/sirm-mpl /home/ntduc/miniconda3/envs/dg/bin/python -u -m experiments.sirm_discover_dynamic.run suite --output results/sirm_discover_dynamic/terra_L38_seed0
```

Each refresh logs pre-update utility EMA, Q, selected cells, shared experts, structural and ESS-eligible pair-expert terms, executed terms, loss, lambda, ESS matrix and validity. Source validation accuracy is measured after the update. Extra refresh evaluations do not create selection checkpoints: the original 250-step selection cadence is preserved. R0 reports eligible support but never executes SIRM.

On active refresh steps, optional autograd.grad measurements record the norm of the weighted SIRM gradient and baseline gradient on residual parameters, plus how many float32 baseline gradient elements change when that penalty gradient is added. These queries retain the graph, do not assign .grad and do not alter the objective or optimizer. Norm accumulation is float64 to avoid squaring tiny gradients to zero. No gradient amplification or weight normalization is introduced.

The suite checks initialization, splits, schedules, capacity and exact pre-activation trajectories, then locks both source-selected checkpoints before target evaluation. At completion it writes all eight requested behavioral outcomes to summary.md/behavior.json, Q_evolution.csv, parameter_comparison.json, target logits and the existing rich per-arm target/source diagnostics. No target tuning, R2/R3 or other datasets are run.

Nonzero penalty is distinct from measurable parameter divergence: the original pair weights can be very small. Report actual parameter/logit differences and gradient rounding rather than assuming any nonzero loss guarantees a distinguishable float32 trajectory.
