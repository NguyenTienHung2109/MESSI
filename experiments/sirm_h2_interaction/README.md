# H2 competence interactions and coverage

Four read-only post-hoc analyses of the 48 A/B/C/D checkpoints:

1. Double-center each equal-domain source-out expert CE matrix. Export domain/expert/interaction effects, exact decomposition of Q risk advantage and descriptive sum-of-squares shares. Repeat on the saved risk EMA as estimator sensitivity; it is the statistic that built Q and not independent validation.
2. Compare each source Q row against Top-2 of expert CE averaged equally across source domains. Export Jaccard, pair identities and exact-row matches.
3. Compare absolute target selected/non-selected CE in B and D under each model's own Q, fixed B masks and fixed D masks. Fixed masks separate composition changes from changes in expert quality.
4. Compare per-expert target CE and target native/dense routing mass by selection count 0/1/2/3 and low/high groups. Stratify on both B and D Q and export transitions and per-target results.

Additional checks use existing cache data: matched-valid routed-G with fixed membership; symmetric decomposition of routing-weighted expert CE into bank-quality and routing contributions; cached probability/logit swaps. Mixture NLL and accuracy remain distinct from weighted expert CE. Swaps change cached output composition, not a retrained network or a router applied to another representation.

The script verifies checkpoint hashes, cache metadata, physical domains, source domain-name ordering, and paired B/D initialization, splits, minibatch schedules, target image IDs/labels/order. Cached native predictions must reproduce the previous target accuracy. Tests use an additive no-interaction control, known nonadditive effects and fixed-mask composition examples.

```bash
PYTHONWARNINGS=ignore MPLCONFIGDIR=/tmp/sirm-mpl /home/ntduc/miniconda3/envs/dg/bin/python -m unittest -q experiments.sirm_h2_interaction.test_analysis
PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/sirm-mpl /home/ntduc/miniconda3/envs/dg/bin/python -m experiments.sirm_h2_interaction.run
```

Outputs: `results/sirm_q_followup/interaction/summary.md` and detailed CSVs/provenance. Runs use CPU while the previously authorized Q pilot continues on GPU. No loss, assignment or model changes are made by these analyses.

Three seeds and overlapping source data limit uncertainty. Frequency is a count at the source-selected checkpoint, not training-history frequency. Comparing equal initialized expert IDs does not establish stable learned roles. Source-out already selected checkpoints. Scalar stationarity, residual competence and target routing mass do not establish latent subset mechanisms or causal mediation. An interaction-aware assignment may emphasize domain idiosyncrasies; these diagnostics do not validate that proposed intervention.
