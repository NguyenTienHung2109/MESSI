# Completed-gate statistics

Post-hoc diagnostics for the completed six-run gate, without modifying training code, checkpoints, Q or selection.

```bash
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/sirm-mpl /home/ntduc/miniconda3/envs/dg/bin/python -u -m experiments.sirm_discover_analysis.run
```

Writes `results/sirm_discover_gate/phase1/stats/summary.md` and per-dataset JSON/CSV/target NPZ files. It compares prediction parameter tensors at best and last checkpoints, all 5001 training steps, and directly replays each selected model on the target. Checkpoint container hashes can differ due to metadata even when learned tensors match.

The Q refresh replay uses the recorded source-in utility EMA and the original float32 formula/thresholds. It reports potential structural support, excluding routing ESS. These are post-hoc candidates on the observed trajectory, not results from training with dynamic Q. No target statistic is used to fit assignment or revise hyperparameters.

Composition stats distinguish full versus global predictions, helped/harmed examples, router NULL and entropy, correction/global norm ratios, residual expert similarity and per-expert target removal effects. Positive removal ΔAcc indicates useful contribution; negative indicates harmful contribution on this target. Effects are not additive. Source-out remains the selection split, and there is only one training seed.
