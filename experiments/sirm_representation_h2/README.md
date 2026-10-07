# SIRM representation × H2 factorial

This experiment imports the existing `experiments.sirm_h1_h6.run` trainer and model. It does not change SIRM mathematics or architecture. OFF sets only `subset_irm_lambda_sirm=0`; expert loss and router distillation remain enabled. Q retains the original `risk_ema` rule.

Conditions: A=baseline/OFF, B=H2/OFF, C=baseline/ON, D=H2/ON. Full scope is four PACS held-out targets × seeds 0/1/2 × four conditions. Exact-matching previous native and balanced-view runs supply C/D after source/config/snapshot/commit verification; A/B train from the same pretrained initialization and batch schedules.

Baseline preprocessing is inherited unchanged. H2 retains baseline RandomGrayscale(p=0.1) and appends equiprobable identity, three-channel grayscale, or Gaussian blur (kernel5/sigma1) before normalization. Gray+blur is an additional deterministic evaluation-only view.

Checkpoint selection is equal-domain source-out accuracy with earliest ties. Native evaluation uses the selected model's own router. Global weights reuse the previous source-in mixture-CE fitting routine and source-out NLL selection. A frozen linear backbone probe fits source-in only and selects L2 regularization on source-out. No target data select a checkpoint, global weights, or probe hyperparameters.

Source-validation diagnostics export competence on identical examples across experts, Q/support overlap and functional subset accuracy, paired router stability, backbone/expert cosine stability, and class cosine margins. The backbone cache is valid because the architecture's input projection is verified to be Identity.

```bash
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/ntduc/miniconda3/envs/dg/bin/python -m unittest experiments.sirm_representation_h2.test_protocol
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/ntduc/miniconda3/envs/dg/bin/python -u -m experiments.sirm_representation_h2.runner suite --smoke --output results/sirm_representation_h2/smoke
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/ntduc/miniconda3/envs/dg/bin/python -u -m experiments.sirm_representation_h2.runner suite --output results/sirm_representation_h2/full
```

Explicit configuration entry point:

```bash
/home/ntduc/miniconda3/envs/dg/bin/python -m experiments.sirm_representation_h2.runner train --aug_mode h2_balanced --sirm off --target-env 0 --seed 0 --output results/sirm_representation_h2/manual_B_env0_seed0
```

Each suite emits a locked `plan.json`, executable command list, unique per-run configs/logs, all requested CSVs, and a progressively populated `summary.md`. The CSVs use fractions for accuracies and differences; summary tables use percent and percentage points. Missing runs remain explicitly pending or failed.

The primary interaction is `(D-B)-(C-A)`. Pairing uses target×seed cells; both cell-bootstrap and fixed-target seed-block confidence intervals are reported. Conditions are never independently resampled. Overlapping source data limit independence of cells, and three seed blocks limit bootstrap precision. An interval containing zero is not evidence of practical equivalence (margin ±0.5pp).

This is an exploratory PACS experiment after previous target inspection. Positive interaction plus representation evidence supports augmentation/SIRM synergy, but does not establish representation robustness as the sole causal mediator or semantic decomposition. Gray+blur has no dedicated training branch, although baseline grayscale followed by the H2 blur branch occurs in approximately 3.3% of draws.
