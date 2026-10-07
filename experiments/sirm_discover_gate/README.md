# Six-run SIRM-v2 gate

PACS Sketch (env 3) and TerraIncognita L38 (env 1), seed 0, three matched arms:

| Arm | Model/discovery | Assignment | Residual IRMv1 |
|---|---|---|---|
| R0 | global + residuals, discovery ON | learned marginal utility | OFF |
| R1 | identical | learned marginal utility | ON |
| R2 | identical | fixed seeded expert permutation of R1's Q | ON |

**Global means ERM**: global CE with H2 augmentation for 1000 steps, then freeze the backbone and global head. Residual discovery runs for 1000 steps. Stage 3 runs 3001 steps with a 500-step linear SIRM ramp to 0.1. All arms use 5001 steps, the same initialization, source splits, minibatches and augmentation schedule. See the parent prototype README for prediction, loss and Q formulas.

For this gate, Q is locked from source-in utility EMA at the start of stage 3, before the penalty changes any trajectory. R0 and R1 retain that matrix. R2 applies a single seed-fixed expert-column derangement. Every row's weights, sparsity, NULL mass and pair-weight total are preserved; domain overlap structure is moved to different experts. The control randomizes association with learned experts, not the overlap graph itself. Utility EMA continues updating for drift diagnostics, while the penalty Q stays fixed. This deliberately tests fixed assignment and differs from the prototype's continually refreshed Q. It avoids introducing per-batch randomness or differences in assignment weight caused by post-activation trajectories.

Source-only 80/20 DomainBed splits and earliest-tie equal-source validation select stage-3 checkpoints. Adapters infer class counts, retain the exact existing PACS split schema, and exclude the target from training inventory and selection. All three runs must pass paired audits before the target is read. Checkpoint hashes are locked in `target_lock.json`. Target results are exploratory gate diagnostics, never a checkpoint or hyperparameter selector. Source-out diagnostics have also been used for selection and are not an untouched holdout.

Diagnostics include exact routed-component scale derivatives (routing is part of the contribution), source U and double-centered U, selected-minus-nonselected interaction advantage, Q/global-Top-K Jaccard, residual norms, single-correction CE gains, correction/global class-centered logit norm ratio, target expert ablations and source-utility/target-importance Spearman. Domain-mean signed derivatives are averaged before taking absolute values. They differ from mean minibatch squared training penalties.

Artifacts: `summary.md`, `runs.csv`, `deltas.csv`; per dataset `mechanisms.csv`, `expert_importance.csv`, `fixed_selected_G.json`, `paired_audit.json`; per arm `source_diagnostics.json` and `milestones/{before,activation,middle,final}.json`. Milestones are post-update source-out measurements; activation has lambda=0 at the ramp boundary. Fixed-mask G comparisons use R1's source-in Q for both R0 and R1. Target expert ablations remove the contribution without renormalizing routing.

Old PACS H2 OFF/ON results are included only after metadata, full target sample count and exact split-hash checks. Old/new schedules and objectives differ, so the architecture delta compares whole methods. Terra matching old H2 results are unverified and omitted. Old standalone-expert risk gaps and new marginal-utility interaction gaps measure different quantities.

```bash
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 /home/ntduc/miniconda3/envs/dg/bin/python -m unittest experiments.sirm_discover.test_discover experiments.sirm_discover_gate.test_gate
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled MPLCONFIGDIR=/tmp/sirm-mpl /home/ntduc/miniconda3/envs/dg/bin/python -u -m experiments.sirm_discover_gate.run suite --smoke --output results/sirm_discover_gate/smoke
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled MPLCONFIGDIR=/tmp/sirm-mpl /home/ntduc/miniconda3/envs/dg/bin/python -u -m experiments.sirm_discover_gate.run suite --output results/sirm_discover_gate/phase1
```

Smoke compresses stages to 4/8/8 and effect floors to zero; its accuracies are pipeline checks. Full defaults retain positive utility/interaction floors. All-NULL/no shared-Q is reported as an inactive penalty test; assignment is not forced to make IRM active. A fresh result directory is required for a changed protocol. Completed jobs are skipped and incomplete jobs resume from saved checkpoints under exact code/config/environment locks.

No automatic expansion to 16 runs: OfficeHome/DomainNet data are absent locally and R3 is deferred. The two-target, one-seed gate cannot establish the requested three-of-four-target success criterion. The old pilot stays stopped.
