# Raw-gradient diagnosis and binary selection gate

TerraIncognita held-out L38, seed 0 only. This follows the dynamic-Q test without changing Q discovery, temperatures/floors, utility definitions, ESS/validity, residual architecture, discovery objective, lambda, ramp or stage budgets.

The no-retrain diagnosis uses the old R1 source-selected final checkpoint (step 5000), the recorded eligible Q matrices, and exactly reconstructed source minibatches/augmentations at their recorded steps. Historical intermediate weights were not saved; therefore Q/batch step and checkpoint step are exported separately. No historical gradient is claimed. The original checkpoint hash is checked after diagnosis; no target data is read.

The result inspects 23 pairs from 21 refresh points: raw residual-parameter gradient norms 4.51e-4–0.211, while Q weights span 3.31e-13–1.91e-8. This supports testing removal of attenuation. It does not establish that the penalty improves OOD performance.

Binary mode keeps exactly the original eligibility logic. Each eligible pair-expert term contributes G_i²+G_j² with weight 1; average over active terms. No eligible terms yields zero. Confidence mode remains the default for reproducibility of earlier experiments; this two-arm test explicitly sets binary mode. Q stays detached and refreshes every 50 steps. OFF changes only lambda_sirm to zero.

The new runs use `domainbed.scripts.train` with registered `SIRMDiscoverV2`, the standard DomainBed source/target split protocol, per-domain `env*_in_acc` / `env*_out_acc` columns and `results.jsonl`. An opt-in scheduled loader and H2 transform preserve the old experiment's exact source minibatch and augmentation schedule. Initialization is seeded after loader setup to preserve the previous model initialization. Evaluation copies use deterministic original views and retain the split keys without changing training transforms. The one-step pipeline verified all eight per-domain score columns and exact initialization/step-0 classification loss agreement with the old Terra run.

Full per-domain scores are evaluated at the original 250-step selection cadence (plus stage ends/final). Source-out evaluation at intermediate Q refreshes logs source accuracy but cannot select a checkpoint. `source_best.pkl` is chosen only by the equal-source-out mean, earliest ties during Stage 3. Target scores displayed during training are monitoring only; final full-target evaluation follows checkpoint locks. Both R0 and R1 train fresh. Optimizer/batch schedule exact resume is not provided by the standard DomainBed weight-only resume; the paired runner requires a fresh output directory.

Gradient norms/ratios in refresh diagnostics refer to residual parameters, matching the prior dynamic-Q diagnostic. Measurements use autograd.grad without assigning .grad or changing the optimizer update. No automatic lambda/ramp adjustment is made if a ratio exceeds one; that is reported for a subsequent decision.

```bash
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 /home/ntduc/miniconda3/envs/dg/bin/python -m experiments.sirm_discover_binary.inspect_current
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 /home/ntduc/miniconda3/envs/dg/bin/python -m unittest experiments.sirm_discover.test_discover experiments.sirm_discover_gate.test_gate experiments.sirm_discover_dynamic.test_dynamic experiments.sirm_discover_binary.test_binary
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled MPLCONFIGDIR=/tmp/sirm-mpl /home/ntduc/miniconda3/envs/dg/bin/python -u -m experiments.sirm_discover_binary.run suite --output results/sirm_discover_binary/terra_L38_seed0
```

Outputs: `raw_diagnosis/{summary.md,raw_pair_gradients.csv,diagnosis.json}` and `terra_L38_seed0/{summary.md,behavior.json,active_refresh_gradients.csv,parameter_comparison.json,target_metrics.json}`. Full train tables are in `banks/{R0,R1}/out.txt`, exact step/refresh records in diagnostics.jsonl/q_refresh.jsonl. Completed historical results remain separate. No PACS or wider sweep is launched.
