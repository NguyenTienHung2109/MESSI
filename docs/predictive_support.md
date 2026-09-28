# Predictive subset supports (`MESSI_Support`)

This implements the supplied proposal as a new method. It retains the existing
DeiT CLS backbone and dense soft MoE head. Previous MESSI, Subset-IRM and
S-IRM-res objectives/checkpoints remain separate experiments. Their results are
not evidence for this method.

## Model and objective

For expert m, normalize its output `z_m = h_m / max(||h_m||, 1e-8)`.
A shared, bias-free linear classifier produces `W z_m`. Each row of W is
projected into the radius-s ball at initialization and after every optimizer
step. Inference computes `sum_m softmax(router(x))_m W z_m`; the mixture is
not normalized. Domain labels and the support table are absent from prediction.
There is no extra encoder, teacher, discriminator, global alignment, separate
expert classifier, sparse router or additional MoE layer.

The training objective after warm-up is exactly

```
L = L_loc + (log(C) + 2s) L_adm + lambda_sub L_sub
L_loc = mean_domain mean_sample max_{m: S[domain,m]=1} CE(W z_m, y)
L_adm = mean_domain mean_sample -log(sum_{m: S[domain,m]=1} pi_m)
L_sub = (1/B) sum_{m,i<j} S[i,m] S[j,m] mean_shared_class MMD^2(z_i, z_j)
```

Admissibility uses a masked log-sum-exp, preserving gradients even for tiny
admissible mass. The Gaussian bandwidth is fixed and positive. Squared MMD uses
the U-statistic (within-domain diagonals removed). Negative finite-sample
values are retained; this estimator is not a nonnegative distance. The
population discrepancy is nonnegative and characteristic.

Support is domain-level, binary and may overlap. Every source domain is covered;
active experts cover at least two domains. Inactive experts are permitted.
The budget counts selected **valid domain pairs**, with multiplicity across
experts. A pair is valid only if its domains share a class in the source training
split. Missing minibatch cells never define population validity. A class unique
to one source has no cross-domain alignment term. Default K=3, M=6, B=3 yields
486 feasible labeled structures (out of 15,625 unfiltered configurations),
including a single global expert and three overlapping pair supports.

## Alternating optimization and sampling

The short warm-up trains every expert with mean expert CE, without alignment or
router loss. Its reported local risk and prediction bound still use the actual
support maximum; only the optimized warm-up objective differs.

At each structure step, the network is held fixed in evaluation mode. Fresh
source-training observations provide per-example expert CE and routing logits.
A separate, larger stratified draw supplies MMD observations for all shared
class-domain cells. All feasible S are scored with the complete objective;
maximum local loss is evaluated per sample, not approximated by a maximum of
mean domain risks. Search is exact, including expert identities. Ties are
resolved by enumeration order. Search limits fail explicitly; there is no
undocumented heuristic for larger domain counts.

Neural steps hold S fixed. Prediction samples are uniform within each source
training split and domain losses are averaged equally, preserving Q. Alignment
samples a selected expert/domain pair uniformly from the B entries, then one
shared class uniformly, then independent samples with replacement from each
cell. Therefore the mean of sampled MMD terms estimates the stated `/B`
objective without class-frequency bias. Sampling with replacement is iid from
the empirical cell law; duplicate indices are possible. Shared cells with fewer
than two distinct available images are rejected rather than presented as
reliable evidence. Evaluation uses deterministic transforms; training and
structure sampling use training augmentation.

Structure observations are structural-training data, never an independent
holdout. Source validation is reserved for checkpoint selection and reporting.
The target is scored only after the source-selected checkpoint is loaded.

## Running

From the repository root:

```bash
python -m unittest domainbed.test.test_predictive_support -v
python -m domainbed.scripts.train_predictive_support \
  --config configs/predictive_support_smoke.json \
  --data-dir domainbed/data --target-env 0 \
  --output-dir predictive_support_outputs/pacs_smoke

python -m domainbed.scripts.train_predictive_support \
  --config configs/predictive_support_pacs.json \
  --data-dir domainbed/data --target-env 0 --seed 0 \
  --output-dir predictive_support_outputs/pacs_env0_seed0
```

Use `configs/predictive_support_terrainc.json` for TerraIncognita. Full
configurations use pretrained DeiT-Small, six experts, 100 warm-up steps,
structure refresh every 100 steps, 128 prediction samples per domain and 16
samples per shared cell. These are starting hyperparameters, not tuned results.
The 3-step smoke uses untrained DeiT-Tiny and tiny structure samples solely to
check execution; its accuracy is not a benchmark.

The dedicated runner is required for stratified alignment and structure steps;
the generic DomainBed `update(minibatches)` loop alone is insufficient. The
algorithm is registered for construction, but deliberately refuses to train
past warm-up without a structure update and alignment observations.

Matched retraining interventions use `--support-mode`:

- `learned`: exact full-objective search at each refresh.
- `fixed`: deterministic initial feasible support, never changed.
- `random`: one random feasible support at initialization, held fixed.
- `no_discrepancy`: exact structure search using local risk and admissibility;
  neural training retains the same MMD term and budget.

Changing S after training cannot affect predictions. Interventions must train or
continue training under the changed support.

`metrics.jsonl` records the binary support, candidate count, structure solver
wall time, total structure time including sampling, neural time, all three
losses, inadmissible routing mass, mixture CE, prediction bound and bound gap.
Source validation reports risk and accuracy for every expert in every domain,
with assignment flags. `best.pt` is selected solely by mean source-validation
accuracy after warm-up; `last.pt` stores the final state. Both store model,
optimizer and configuration; S, class presence and neural step are model
buffers. `results.json` reports the selected checkpoint's source diagnostics
and target accuracy. The runner does not implement automatic checkpoint resume.

## Mathematical scope

The population alignment is zero iff all selected class-conditional laws agree,
since its finite positive-weight sum comprises characteristic MMD terms. This
statement does not apply to zero or negative empirical U-statistics.

For bounded W and normalized expert outputs, expert CE is at most
`Lmax = log(C) + 2s`. Convexity in logits gives, pointwise,

```
CE(f_pi, y) <= sum_m pi_m CE(f_m, y)
            <= max_admissible CE(f_m, y) + Lmax (1 - admissible_mass)
            <= max_admissible CE(f_m, y) + Lmax (-log admissible_mass).
```

Thus `R_Q(f_pi) <= L_loc + Lmax L_adm`. This is a prediction-risk bound,
not a recommendation that Lmax is the best optimization coefficient. For each
active expert, `R_m <= K / |K_m| L_loc`; CE decomposition then gives
`I_Qm(Z_m;Y) >= H_Qm(Y) - R_m`. Constant expert features cannot bypass local
supervision merely by allowing the router to classify.

With fixed representation and full-score separation gap Gamma, uniform score
estimation error epsilon and empirical solver error delta, a selected support
has population score at most `F(S*) + 2 epsilon + delta`. Recovery up to
appropriate equivalence requires `Gamma > 2 epsilon + delta`. Exact enumeration
has zero combinatorial optimization error (apart from numerical precision), but
this implementation does not certify epsilon, Gamma, or emergence of a gap
through joint training. With fixed expert identities, permuting S alone is not
an equivalence; expert outputs must be permuted together when matching.

Support is not class-dependent and each assigned expert must be a local
predictor. These restrictions are intentional. The method does not claim
recovery of causal factors or unconditional unseen-domain generalization.
Target success still requires predictive expert coverage and admissible routing
on the target. Overlap does not impose equality on the final mixed representation.

## Validation status

Tests cover feasible-set enumeration against an independent exhaustive oracle,
missing classes, exact zeros outside support, negative U-statistics, analytic
router gradients, the mixture prediction bound, checkpoint round-trip and
inference independence from S, actual alternating gradient updates and norm
projection. Fixed-representation synthetic cases test known overlapping support
recovery, a single global solution, an ambiguous no-gap case, discrepancy
intervention, and a router shortcut with constant local experts.

These are mechanism checks, not evidence that end-to-end learning recovers
unknown synthetic factors. Before paper claims, run the full multi-seed source /
OOD protocol and matched interventions, and an end-to-end synthetic benchmark
with support matching. No old objective's results should be relabeled as results
of this implementation.

Local verification on 2026-09-27: 45 tests passed across
`test_predictive_support`, `test_sirm_res` and `test_subset_irm`. GPU PACS smoke
runs completed for all four support modes; each saved and reloaded its selected
checkpoint and evaluated the held-out target. Learned supports changed during
refresh; fixed and random controls retained their initial supports. Every
logged support satisfied coverage, minimum active support size and B=3.
Artifacts are under `predictive_support_outputs/pacs_smoke_final` and
`predictive_support_outputs/pacs_smoke_{fixed,random,no_discrepancy}` (gitignored).

## Detailed PACS EDA runs

The four-target seed-0 experiment runs sequentially on one GPU:

```bash
python -m domainbed.scripts.run_support_pacs_queue \
  --output-root predictive_support_outputs/pacs_eda_lodo_seed0
```

`queue.json` records job PIDs and completion/failure; `envN_seed0.log` contains
training progress. Each run has an `eda/report.md` updated at every source
validation checkpoint, with PNG figures and CSV tables. The parent `report.md`
links the four reports and shows available results; `summary.json` is produced
when all four targets finish. Regenerate a run's report with:

```bash
python -m domainbed.scripts.report_predictive_support \
  predictive_support_outputs/pacs_eda_lodo_seed0/env0_seed0
```

Detailed observations are stored in `metrics.jsonl`:

- Every step: loss components and weighted terms, per-domain local risk and
  admissibility, worst-loss expert frequencies, classifier row norms, elapsed
  time and peak allocated CUDA memory. Warm-up monitors these quantities but
  optimizes only mean expert CE; its admissibility/MMD coefficients are not
  active training losses.
- Every 25 steps in the PACS config: total-objective gradient norm for backbone,
  router, classifier and each expert. No diagnostic regularizer is introduced.
- Every structure refresh: changed support bits, active expert count, coverage,
  support sizes, old/new score on the same probe, empirical runner-up gap,
  top-10 candidate structures with each loss component, all valid pair/expert
  MMD values and class-specific MMD values. Finite-sample negative estimates are
  retained. Candidate gaps do not certify population recoverability.
- Every validation checkpoint, per source domain: class counts, confusion
  matrix, accuracy/risk/routing for each class and expert, routing mean/std/top-1
  frequency, entropy and exp(entropy), max-routing quantiles, oracle expert
  accuracy, expert prediction disagreement, feature mean norm and variance
  trace, cross-expert feature cosine, confidence, 10-bin ECE and multiclass Brier
  score. Missing classes are reported as null, not zero accuracy.
- For the selected final checkpoint: the same source diagnostics plus target
  diagnostics. Target admissibility is intentionally undefined because no
  target support is learned. `source_predictions_K.npz` and
  `target_predictions_0.npz` save labels, mixture probabilities, expert
  predictions, routing weights and expert CE for post-hoc EDA. Arrays follow
  deterministic evaluation order; class names and source/target domain mapping
  are stored in `config.json`.

Structure plots show the final training trajectory; final expert/domain figures
use the source-selected checkpoint. Feature variance and cosine are descriptive
statistics, not proofs of predictive information or invariance. Oracle expert
accuracy uses ground-truth labels and is a diagnostic ceiling, not deployable
inference. The initial full experiment uses one seed and has no matched baseline
training, so it cannot alone establish a method improvement.
