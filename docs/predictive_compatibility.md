# Predictive compatibility: fixed-map mechanism prototype

Four mechanism checks, with fixed diagnostic controls and seed 0:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m domainbed.scripts.check_compatibility_mechanisms
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m domainbed.scripts.check_compatibility_mechanisms --matched-probe-followup
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m domainbed.scripts.check_compatibility_mechanisms --standard-probe-check
OMP_NUM_THREADS=2 python -m domainbed.scripts.report_compatibility_checks
OMP_NUM_THREADS=2 python -m unittest domainbed.test.test_compatibility_checks -v
```

These add bound decomposition/sample-size calculations, constant and half-uniform
router controls, normalized backbone linear probes, mean-vs-minimax heads,
head-radius sensitivity and a 64-dimensional MLP intervention. Follow-ups match
the supervised sample pool and also test a fixed-C conventional linear probe
outside the certified class. All adaptation excludes held domains; no control
is promoted into deployment based on target scores. Reused source-set fits are
checked for exact equality of the original frozen map parameters.
Artifacts are in `predictive_support_outputs/pacs_mechanism_checks_seed0`.
No backbone fine-tuning is performed by this diagnostic suite.

Post-hoc t-SNE visualization (requires `scikit-learn`):

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m domainbed.scripts.visualize_compatibility_tsne
```

Outputs: `predictive_support_outputs/pacs_compatibility_tsne_seed0/explorer.html`
(standalone interactive viewer), eight static figures, 24 saved embeddings, and
surrogate fidelity/neighborhood-trustworthiness metrics. Each target run uses
source selection samples and up to 40 held-target samples per class, with the
same sample IDs for backbone and expert embeddings. These balanced/capped
visualization samples do not reproduce the full-test class proportions.
The 2D regions interpolate frozen logits, not ground-truth labels. They are
explicitly approximate: t-SNE has no unique inverse and cannot display the
original high-dimensional decision boundary exactly. Held-point surrogate
fidelity is conditional on the transductive embedding, not new-image accuracy.
No model, threshold or deployment decision is changed by visualization.

Full one-seed PACS behavior audit (four targets plus 12 excluded-source folds):

```bash
OMP_NUM_THREADS=2 python -u -m domainbed.scripts.run_compatibility_pacs
```

Results are written to `predictive_support_outputs/pacs_compatibility_lodo_seed0`.
`report.md`, `metrics.csv`, `expert_behavior.csv` and `hypothesis_audit.csv`
summarize all domains. Each run also preserves live map/router events in
`main/training.jsonl`, per-stage source diagnostics in `main/source_behavior.json`,
and sample-level features/logits/router weights in target and exclusion NPZ files.
Training-event losses are minibatch diagnostics, not independent risk estimates.
Feature variance/effective rank and prediction disagreement are descriptive, not
certificates of complementary causal factors. Removal deltas keep the remaining
experts and router scores fixed, renormalizing weights; they are not retrained
leave-one-expert-out models.

The full suite uses the same model and optimization configuration as the first
Sketch audit below, but allocates α=.05 across **228 events** for all four runs
and their exclusion folds (`--family-runs 4`). This makes confidence bounds
slightly wider. The suite is exploratory: previous PACS results have been seen,
so it must not be presented as a newly untouched external validation dataset.

This implements the new definition requested in the September 2026 review.
It is **not** an equivalent implementation of class-conditional SSI. It tests
common near-optimal prediction relative to a specified linear-head class;
it does not establish equality of class-conditional feature distributions,
causal identification, or unconditional target validity.

The old revision queue was cancelled at the user's request. Historical
checkpoints and results remain intact. The new prototype does not use MMD,
admissibility, positive pair budgets, scalar IRMv1 penalties, diversity losses,
PCGrad, or SWAD.

## Protocol fixed before the first PACS audit

- Seed 0 only. First target: Sketch (environment 3); sources: Art, Cartoon,
  Photo. This is one diagnostic run, not a four-target performance claim.
- Frozen ImageNet-pretrained distilled DeiT-small CLS features, deterministic
  resize to 224 and ImageNet normalization. No backbone updates or target fitting.
- Each source is randomly split, without label stratification, into 50% map
  training, 20% head calibration, 20% audit, 10% composition selection.
  Random permutation uses seed `0 + 1009 * domain_id`. Indices are persisted.
- Each map: independent 384→64→16 MLP, GELU, unit-norm output. Train 500 steps
  with equal-domain CE, Adam lr .001, 64 samples/domain/step. The training head
  is also row-bounded. Maps are frozen thereafter.
- Bank: global baseline plus three pair candidates plus an independently
  initialized global candidate. Duplicate supports are hypotheses, not forbidden.
- Bias-free 7-class linear heads with each row norm ≤1. CE upper bound is
  `L = log(7) + 2`. Calibration minimizes maximum domain-wise excess CE.
- Fixed tolerances: compatibility τ=.12 nats, within-domain information
  margin δ=.05 nats; family failure probability α=.05. These were not adjusted
  after inspecting any PACS target or audit outcomes.
- Routers: a linear layer on frozen CLS features, 300 Adam steps, lr .001,
  batch 64; experts and heads frozen. The objective is actual mixture-logit
  CE. Shorter domains are deterministically repeated to equalize their weight.

The restricted head scale and frozen 16-dimensional candidate class are
explicit modeling choices. This is not a matched V1/V2 accuracy comparison:
the backbone adaptation, data splits, capacity, and optimization differ.

## Convex optimization and numerical evidence

For fixed map h, the class is the product of C Euclidean row balls of radius s.
For each domain, solve `R*_k = min_W R_k(W)` and then
`J_A = min_W max_k (R_k(W) - R*_k)`.

The solver uses float64 projected accelerated descent with backtracking.
Minimax search uses log-sum-exp temperatures [.1, .03, .01, .003], but the
reported certificates bound the **unsmoothed** objective. Search stops at gap
≤1e-5 or 1000 iterations; hitting the budget does not imply convergence.

For feasible W and convex differentiable F, the Frank–Wolfe gap is
`g(W) = <∇F(W), W> + s sum_c ||∇F(W)_c||`.
Convexity gives `F(W)-g(W) ≤ min_V F(V) ≤ F(W)`.
Record the strongest lower bound and best feasible upper bound over iterates.
No scalar gradient norm is substituted for this certificate.

If individual optima lie in `[l_k,u_k]`, a feasible common head gives
`J_A ≤ max_k (R_k(W)-l_k)`. For any simplex vector q, another valid lower bound is
`sum_k q_k(R_k(W)-u_k) - g_q(W)`. Zero is also a lower bound. The report preserves
both bounds, their gap, iteration counts, and convergence flags. Some PACS
calibration solves do not reach 1e-5; their residual gaps remain in the report.
Bounds use standard floating-point arithmetic, not interval-arithmetic proofs.

## Independent certification

Candidate maps and deployed common heads are fixed before audit. For each
candidate/domain, audit solves another domain-specific optimum with certified
lower bound `l_audit`. It then bounds the **deployed head's** population regret:

`r_k(W_deployed) ≤ Rhat_k(W_deployed) - l_audit + 2 ε_k`.

This is stronger than merely certifying the existence of some common head.
Audit optimality error is already included through `l_audit`; calibration
error is reflected in the deployed head's risk, not silently discarded or
double counted. Audit-fitted domain heads are reference optima only, never
deployed and never fed back into map learning.

For fixed normalized features, CE is √2-Lipschitz in W. The parameter set has
ambient dimension p=Cd and Frobenius radius B=s√C. An r-net has at most
`(1+2B/r)^p` members. A union bound and Hoeffding give

`ε(r) = L sqrt((log(2/α_event) + p log(1+2B/r))/(2n)) + 2√2 r`.

We minimize this deterministic formula over 160 geometric radii and cap ε at
the trivial valid bound L. The radius choice depends on sample size and
declared class, never labels or audit risk. We allocate α equally across all
prespecified candidate/domain risk and entropy events and per-domain
composition-gain events, including all requested exclusion folds. For the
first run this is 57 events. Cross-fold independence is not required for a
union bound; within each fold, maps must be independent of their own audit
samples. The probability statement assumes iid sampling within each domain;
the random split does not itself establish that PACS satisfies this assumption.

Predictivity is required in **every assigned domain**:
`Rhat_k(W) + ε_k ≤ H_lower,k(Y) - δ`.
For entropy, simultaneous coordinate-wise Hoeffding bounds imply a TV radius;
the entropy lower bound subtracts the Fannes–Audenaert continuity bound, or
log C when the TV radius is too large. This guards the domain–label-prior
loophole in pooled entropy. It does not establish complementary predictors.

Certification requires both inequalities in all assigned domains. Failure is
reported as `insufficient_evidence`, not acceptance of equality and not proof
that no representation in a larger family could work. Audit reuse to modify
maps or thresholds would require a new statistical protocol.

## Composition, contribution, and transfer

The global baseline is always available even if it is not certified. A second
router is fit on global plus certified extra experts only. A matched diagnostic
router uses the whole bank. Both minimize mixture-logit CE without membership
supervision. Uniform mixing and single-global use exactly the same learned bank.

On the independent selection split, paired CE gain over global lies in [-L,L].
A per-domain lower confidence bound is
`mean_gain - L sqrt(2 log(1/α_event)/n)`.
Use the certified composition only if its average lower bound across domains
is positive; otherwise return global. This tests the additional predictive
value of the whole proposed composition, not distinct causal factors in each
expert. Accepted but redundant experts need not cause deployment of a mixture.

Strict leave-one-source-domain-out diagnostics rebuild maps, heads, audit,
router, and selection using only the other sources. `fit_fold` rejects a bank
containing any excluded domain. Shared cache extraction is an immutable
pretrained transformation, not adaptation. With only two adaptation domains,
the two candidates both have the whole available support; there is no smaller
support of size ≥2. This is an explicit limitation of three-source evaluation.

Target evaluation happens after all source-only selection and all exclusion
folds. Target scores and label-informed single-expert oracles are diagnostics
only. Oracle expert accuracy is not a bound on arbitrary logit mixtures.
Theoretical target-risk conclusions still need persistent conditional
mechanisms, coverage/density-ratio control, realizability and routability.

## Controlled tests and commands

```bash
OMP_NUM_THREADS=2 python -m unittest domainbed.test.test_predictive_compatibility -v
OMP_NUM_THREADS=2 python -m domainbed.scripts.check_compatibility_synthetic
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 python -u \
  -m domainbed.scripts.audit_predictive_compatibility --target 3 --loso
python -m domainbed.scripts.report_predictive_compatibility \
  predictive_support_outputs/pacs_compatibility_seed0
```

The synthetic diagnostic uses known Gaussian mean geometry plus fixed sign
projections, 200,000 iid observations/domain/stage compressed into exact CE
sufficient counts. It checks positive pair supports, a global-only mechanism,
and no-subset cases. The analytic Gaussian SSI criterion and the bounded sign
map's predictive compatibility test are distinct, explicitly labeled outputs.
These validate certification for supplied maps, **not** neural factor discovery
or recovery of an OOD router. Unit tests additionally check analytic optimum
intervals, conflicting predictors, different risks with the same optimum,
insufficient samples, partition disjointness, exclusion, fallback and Jensen.

The runner refuses to overwrite an existing audit directory. Local cached
features and model weights remain outside Git; code snapshots, indices,
configuration, detailed certificates and metrics make each run reviewable.

## First run: evidence and limitations

The first seed-0 Sketch audit and three exclusion folds completed. No candidate
passed certification. In the main fold, empirical audit regrets were roughly
.003–.006 nats, but uniform ε was 3.024–3.515 nats, making the certificate
vacuous at τ=.12. Entropy lower bounds were also conservative. This isolates a
statistical-power limitation; small empirical regret alone is not evidence of
population compatibility.

Selected global Sketch accuracy was 42.91%, versus 45.56% for uniform and
40.85% for the whole-bank learned router. Source selection accuracy of the
global predictor was 93.17% Art, 91.06% Cartoon, 98.20% Photo. The representation
therefore fits source prediction much better than it transfers to Sketch;
these results do not validate target robustness. We did not select uniform
based on its target score or tune thresholds until a subset appeared.

Strict excluded-source global accuracy: Art 75.39%, Cartoon 61.69%, Photo
94.31%. All folds returned global. These are limited diagnostics, not evidence
that subsets cannot exist. The next research decision is whether to develop
sharper valid confidence bounds and/or better prespecified feature candidates,
not to treat a forced subset assignment as evidence of discovery.
