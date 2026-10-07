# Failure-driven revision: reference-hull certification and predictive episodes

**Measured status after the seed-0 suite:** all 16 extra main-run candidates per
variant pass the narrower hull audit; none passes the original full-class audit.
No main composition is deployed because source-selection gain is not positive.
Cross-domain feature episodes do not outperform the within-domain control or
the original ERM on mean source-exclusion accuracy. They remain experimental
and are **not promoted as the default feature learner**. See
`predictive_support_outputs/pacs_transfer_compatibility_seed0/conclusions_vi.md`.

This revision responds to measured failures, with separate interventions and
controls. It does not claim an unconditional DG theorem or discovery of causal
factors. The original data, models and selection decisions are preserved.

## Failure → intervention → falsification

| Observed failure | Intervention | What would falsify its usefulness? |
|---|---|---|
| Full-class uniform bounds require roughly 2.3M audit samples/domain at τ=.12; solver error is negligible. | A calibration-fixed convex reference hull; estimate finitely many directional expectations with variance-sensitive confidence bounds. | Hull bounds still vacuous, nonpredictive candidates, or trivial references; passing is never interpreted as full-class certification. |
| Entropy continuity and worst-case pointwise risk bounds are very conservative. | Exact simultaneous class-probability intervals, entropy minimization over their feasible simplex, and fixed-head empirical Bernstein risk. | Predictivity still fails; no δ reduction to manufacture acceptance. |
| Width16→64 leaves effective rank near five and does not improve transfer. | Learn features using cross-domain ridge-head prediction episodes, not a rank/diversity target. | No gain on strict source-exclusion, or same-domain episodes explain the entire gain. |
| Input-dependent routing loses to uniform on 8/12 exclusion folds; constant barely differs from uniform. | Uniform composition, retaining independent predictive-contribution selection and global fallback. | Accepted experts fail to add predictive value; then keep global rather than forcing mixture use. |

## 1. What changes in the certification hypothesis

Fix a learned map h with ||h(x)||≤1. All the following heads are fit on the
calibration split, before any audit observation is read:

- W₀: common minimax-regret head in the original row-bounded class W;
- Vₖ: the calibrated domain-specific optimum for **every** assigned domain;
- V_pool: equal-domain mean-CE head.

Define H = conv{W₀,V₁,…,V_K,V_pool}. Every vertex has row norm ≤1, so H is a
compact convex subset of W. References are not dropped based on audit results.
The deployed head remains W₀. The new hypothesis is

`R_k(W₀) − min_{V∈H} R_k(V) ≤ τ` for every assigned domain.

This is **reference-hull compatibility**, not compatibility with all of W.
This distinction is mathematically necessary, not a naming detail:

`R_k(W₀) − R*_{W,k} = [R_k(W₀) − R*_{H,k}] + [R*_{H,k} − R*_{W,k}]`.

The last population approximation term is nonnegative and is **not** bounded
by the new certificate. All original full-class audit results remain logged.

The hull is not chosen as a singleton to trivially pass. It contains every
calibration-domain optimum. If Vₖ has certified optimization gap ηₖ, then on
that domain's calibration sample:

`0 ≤ min_{V∈H} Rhat_k(V) − min_{V∈W} Rhat_k(V) ≤ ηₖ`.

Thus the hull retains each empirical optimum up to known solve error. That
does not extend automatically to the population or to target data. Numerical
solver gaps and the actual vertices are saved with the checkpoint.

## 2. A dimension-independent directional certificate for H

For any fixed vertex Vⱼ, define the scalar observation

`Dⱼ(x,y) = <∇_W CE(W₀h(x),y), W₀−Vⱼ>`.

Convexity gives `R_k(V) ≥ R_k(W₀)+<∇R_k(W₀),V−W₀>`. A linear functional reaches
its extremum at a hull vertex, hence

`R_k(W₀) − min_{V∈H}R_k(V) ≤ max_j E_k[Dⱼ]`.

The W₀ vertex gives an exactly zero direction. The other observations are
fixed functions conditional on map training and calibration. There is no
covering net over Cd parameters and no audit-head optimization in this bound.

Let Δ=W₀−Vⱼ and Bⱼ=max_{c,y} ||Δ_c−Δ_y||₂. Since p=softmax(W₀z),

`Dⱼ = Σ_c p_c (Δ_c−Δ_y)ᵀz`, so `|Dⱼ| ≤ Bⱼ`.

This deterministic range depends on the fixed heads, not the observed extrema
of audit samples. It is unchanged by adding a common vector to every row of
Δ, as required by softmax invariance.

For observations X∈[a,b], use the one-sided empirical Bernstein bound

`E X ≤ Xbar + sqrt(2 S² log(2/α_e)/n) + 7(b−a)log(2/α_e)/(3(n−1))`,

where S² is the unbiased sample variance. This is a rescaling of Theorem 4 in
[Maurer & Pontil (2009)](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf).
Cap the upper estimate at b. Apply it to every Dⱼ and take the maximum. The
result bounds deployed-head regret on H. Numerical optimization error is not
discarded: all vertices and W₀ are actual fixed feasible heads; exact optimality
of their calibration fits is not required for this population inequality.

## 3. Within-domain predictivity with sharper valid ingredients

For the fixed deployed W₀, `CE_y = log Σ_c exp((W₀,c−W₀,y)ᵀz)`.
Pairwise row distances give a deterministic lower and upper loss bound:

`l = min_y log Σ_c exp(−||W₀,c−W₀,y||)`;
`u = max_y log Σ_c exp(+||W₀,c−W₀,y||)`.

Apply empirical Bernstein to the observed CE values in [l,u]. To lower-bound
label entropy, construct a Clopper–Pearson interval for each class probability
at failure probability α_e/C, using SciPy's
[exact binomial confidence interval](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
Their intersection with the simplex contains the true probability vector with
probability at least 1−α_e by a union bound.

Entropy is concave, so its minimum over this polytope occurs at a vertex. At
most one coordinate is free at each vertex; enumerate C·2^(C−1) cases. For
PACS C=7 this is small. This yields H_lower without replacing the whole
probability uncertainty region by a coarse total-variation radius.

Accept predictivity only if `risk_upper ≤ H_lower − δ`, with unchanged δ=.05.
Because CE≥H(Y|Z), this gives the original within-domain information statement
`I_k(Y;Z)≥δ` on the confidence event. It says nothing about complementary or
causal factors.

## 4. Feature learning through prediction on another source domain

The fixed pretrained DeiT and 16-dimensional MLP architecture are unchanged.
Keep the equal-domain training CE. Add an auxiliary episodic task:

1. Pick query domain q cyclically from the map's candidate support.
2. Draw inner examples from other domains' **training split**.
3. Solve a regularized least-squares head on their one-hot labels.
4. Compute squared prediction error on training examples of q and differentiate
   through the inner solve into the feature map.

With inner feature matrix Z and one-hot targets Y:

`A*(h) = argmin_A ||ZA−Y||²_F/n + λ||A||²_F`
`       = (ZᵀZ/n + λI)⁻¹ ZᵀY/n`.

For λ=.01>0 the quadratic Hessian is positive definite, so the solution is
unique and differentiable. Differentiating the normal equation gives

`dA* = (C+λI)⁻¹(dB − dC A*)`, with C=ZᵀZ/n and B=ZᵀY/n.

The feature objective is

`L(h,W) = mean_source CE(Wh(X),Y) + mean_query ||h(X_q)A*(h)−onehot(Y_q)||²`.

Its extra coefficient is fixed at 1 before running. Squared prediction error
has a direct classification interpretation: for any real score vector a, an
incorrect argmax implies `||a−onehot(y)||²≥1/2`, hence auxiliary classification
error is at most twice its expected squared loss.

This encourages a predictor fit on other domains to work on q, rather than
rewarding arbitrary variance, dimensionality, or feature discrepancy. It does
**not** guarantee generalization to arbitrary targets. The auxiliary ridge
head differs from the deployed bounded CE head, so improved auxiliary loss is
not sufficient evidence; downstream CE/accuracy must also improve.

The query domain is still a **training** domain: its labels update the map.
It is not claimed as an untouched validation environment. Strict exclusion
folds separately leave an entire domain out of map learning, all head fits,
certification and selection.

## 5. Matched interventions and deployment

Three prespecified variants:

- `erm`: exact saved original maps, to isolate the certification replacement;
- `within_domain`: same auxiliary ridge objective and inner sample count, but
  inner and query examples are disjoint draws from the same domain;
- `cross_domain`: inner examples come from other source-training domains.

The episode variants share the same initialization, ERM batch sequence,
500-step budget, dimensions and inner sample counts. They cost more compute
than ERM due to the auxiliary solve. Comparing cross to within isolates the
cross-domain mechanism better than comparing cross to ERM alone.

Use uniform mixing instead of another learned input router. Candidate admission
requires both hull compatibility and per-domain predictivity. The global
predictor remains available irrespective of certification. Compare the proposed
uniform mixture to global on the independent source-selection split, using a
paired empirical-Bernstein lower bound on CE gain. Logit convexity preserves
the Jensen inequality for mixing distinct expert heads. If average lower gain
is not positive, deploy global; no expert budget forces a mixture.

## 6. Probability accounting and research limits

Before the run, enumerate every risk, entropy, directional and selection event
over three feature variants, four targets, all candidate supports, and 12
exclusion folds per variant. The total is **2484 events**; α_e=.05/2484.
The original full-class audit also consumes events in this allocation.
Each entropy event internally allocates its probability over C class intervals.
No independence between overlapping folds is required for a union bound.

The theorem assumes iid samples per domain, independent audit relative to map
and head fitting, and a fixed candidate-generation procedure. Floating-point
calculations are checked numerically, not with interval arithmetic.

PACS has already been examined repeatedly during this project. The current
replay is exploratory and must not be presented as a fresh prospective 95%
validation of a method designed after seeing earlier audits. Fresh data or a
locked independent evaluation is needed for that stronger claim. A narrower
class passing more often is not itself an improvement in DG accuracy.

## Commands and artifacts

```bash
OMP_NUM_THREADS=2 python -m unittest domainbed.test.test_transfer_compatibility -v
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u -m domainbed.scripts.run_transfer_compatibility
```

Artifacts: `predictive_support_outputs/pacs_transfer_compatibility_seed0`.
Per-fold checkpoints include all reference vertices, maps, heads and selection.
`certificates.csv` keeps old/full and new/hull quantities side by side;
`metrics.csv` keeps matched feature objectives and controls. Training events
record query/inner domains, CE, ridge query error and fixed coefficients.
The suite does not perform backbone fine-tuning or alter τ=.12 / δ=.05.
