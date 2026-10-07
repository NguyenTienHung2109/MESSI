# H2 penalty activation / lambda pilot

Imports the original trainer, model and selected-checkpoint evaluator. The subclass adds gradient/Q logging only; objective, optimizer and architecture remain inherited. A synthetic backpropagation test verifies that counterfactual-gradient measurement does not change the classification gradient.

Main pilot: Art and Sketch held-out targets, seeds0/1/2, H2 in every arm. Lambda0/.01/.025/.05/.1, plus lambda.1 with the existing500-step linear ramp beginning at step500. Full accuracy runs retain5001 training steps and source-only checkpoint selection. Twelve exact lambda0/.1 runs are reused;24 new models train. Initial weights, splits, schedules and parameter counts are audited across all arms.

Diagnostic replay:12 reference trajectories to step1050, dense Q/loss/ESS logging around activation500 and the next assignment update1000. Gradient cosine/norm measurements are made on the actual training graph; OFF also reports counterfactual raw SIRM gradients without adding them to the objective. Logging never increases Q-update frequency. Replay losses/source-val scores are checked against the old reference at steps0/500/1000 before new pilot training proceeds.

All source scores must finish before locking a setting. Select highest equal target/seed mean selected-checkpoint source-out accuracy; ties prefer lower lambda then no ramp. Only after that lock evaluate new target checkpoints, using the original evaluator for native/global routing and deterministic views. Prior PACS target inspection makes this exploratory; no target score chooses a setting.

```bash
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/ntduc/miniconda3/envs/dg/bin/python -m unittest experiments.sirm_h2_activation.test_logging
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/ntduc/miniconda3/envs/dg/bin/python -u -m experiments.sirm_h2_activation.run suite --smoke --output results/sirm_representation_h2/activation_smoke_final
/home/ntduc/miniconda3/envs/dg/bin/python -m experiments.sirm_h2_activation.launch
```

The suite writes source locks, command lists, per-run manifests/checkpoints, actual-minibatch dense diagnostics, progressively updated runs.csv/onset_trajectories.csv/summary.md, and explicit failures. A replay is not a shortened-budget accuracy ablation. New accuracy comparisons use the full matched budget.
