# Router-independent predictive support revision

This revision responds to the PACS seed-0 analysis of commit `c5f4ec4`. The old
configuration and saved results retain their original meaning. The existing
`PredictiveSupport` class selects the objective from explicit hyperparameters;
omitting the new flags preserves the original max-CE algorithm and checkpoint
layout. The new configuration is `configs/predictive_support_v2_pacs.json`.

## Structure discovery

For fixed network parameters, the revised structure score is

```
F(S) = lambda_pred * sum_active_m relu(R_m(S) - (H_m(S) - delta))
       + lambda_sub * L_sub(S)
```

`support_structure_router_weight=0` removes admissibility from this score.
Routing can still be logged as a diagnostic, but cannot affect candidate
ordering, support choice, or its score gap. A test changes router logits by
thousands while holding expert outputs fixed and requires identical scores and
supports. MMD, the binary feasible set, overlapping supports, pair budget and
exact enumeration remain as before. `no_discrepancy` removes only the MMD
component of structure selection, keeping neural MMD training unchanged.

The domain-reference law Q is uniform over the K sources, then uniform over
each source's training observations. For active expert m:

```
R_m(S) = mean_{k: S[k,m]=1} E[CE(W z_m, Y) | D=k]
p_m(c) = mean_{k: S[k,m]=1} p_train(Y=c | D=k)
H_m(S) = -sum_c p_m(c) log p_m(c)
```

The fixed per-domain class probabilities come from **all source-training split
labels**. They are buffers saved in v2 checkpoints. Neither target nor source
validation labels set thresholds. Entropy is computed after mixing the label
distributions; averaging domain entropies, using `log C`, or taking entropy of a
sparse batch would produce a different constraint. Empirical risks use equal
domain weights even if batch sizes or source sizes differ. Inactive experts
contribute zero; the hinge is applied after computing each support risk and
summed over active experts, as proposed. Thus candidates with more active
experts can incur more total constraint violation; this is part of the stated
sum objective, not a normalized mean over experts.

Once an expert meets its estimated threshold, the hinge gradient is zero.
Mixture CE and MMD can still change that expert. The loss does not force expert
disagreement, forbid duplicate supports, or ensure all assigned experts are used.
The latter are observed through EDA, not enforced by another regularizer.
Minibatch risk is a noisy estimate: applying the hinge to it is not an unbiased
estimator of the hinge of population risk. Batch fluctuations can still activate
the penalty even when the population constraint is satisfied.

## Neural optimization and warm-up

After warm-up the objective is

```
L = CE(f_pi, Y) + lambda_pred * L_pred
    + lambda_adm(t) * L_adm + lambda_sub * L_sub.
```

Mixture CE provides sample-level supervision for choosing experts **within** the
admissible set. The representation normalization, shared bias-free bounded W,
and unnormalized linear mixture preserve the existing prediction algebra.

Warm-up trains mixture CE plus the predictive hinge with all experts evaluated
on the global source law. It applies no alignment or admissibility term; the
router now receives a mixture-prediction gradient. This all-domain warm-up is
not claimed to satisfy the post-warm-up pair budget. The first feasible support
is selected at the end of warm-up.

The default v2 configuration has delta=0.1 nats, lambda_pred=1, lambda_sub=1,
lambda_adm=1, and a 200-step linear ramp starting at the first structure step.
The coefficient is 1/200 on that step and reaches 1 at the 200th post-warm-up
neural update. This avoids reusing the conservative theoretical Lmax as a large
optimization weight by default. The ramp does not reset at every structure
refresh. The `hinge_matched_adm` arm retains Lmax and no ramp to isolate this
schedule change. These are prespecified starting settings, not target-tuned
hyperparameters.

## Theory and limits

Under a population constraint `R_m <= H_Qm(Y)-delta`, cross-entropy decomposition
implies `I_Qm(Z_m;Y) >= delta`. An empirical soft hinge does **not** guarantee that
population constraint or even that all training constraints are satisfied;
violations are logged explicitly. Constant features with the optimal label-prior
classifier have risk H_Qm(Y) and pay at least delta in the population hinge.
For low-entropy supports with H_m < delta the requested threshold is impossible;
the code retains and reports the violation rather than pretending it is feasible.

The old Jensen upper bound `local max CE + Lmax * admissibility` remains a valid
diagnostic for the same bounded predictor, but it is no longer the optimized
objective. A hinge value cannot replace local max CE in that bound. Structure
recoverability, if discussed, must use the new router-free population score and
its estimation error; neither an empirical gap nor a positive information-bound
estimate is a certificate. Since L_mix trains the shared backbone, the router
can still indirectly influence learned representations. Independence here is of
the **structure score at fixed expert predictions and features**, not of the
entire joint training history.

## Experiments

Tests and all-arm GPU smoke:

```bash
python -m unittest domainbed.test.test_support_revision \
  domainbed.test.test_support_eda domainbed.test.test_predictive_support \
  domainbed.test.test_sirm_res domainbed.test.test_subset_irm
python -m domainbed.scripts.run_support_revision_suite --smoke \
  --output-root predictive_support_outputs/pacs_revision_smoke
```

Full prespecified PACS suite:

```bash
python -m domainbed.scripts.run_support_revision_suite \
  --output-root predictive_support_outputs/pacs_revision_suite
```

The suite runs sequentially on one GPU, 5,000 steps per job:

| Arm | Neural predictive loss | Structure score | Admissibility | Seeds |
|---|---|---|---|---|
| v2_learned | mixture CE + information hinge | hinge + MMD | 1, ramp 200 | 0 |
| decoupled_max | old max CE | max CE + MMD | old Lmax, no ramp | 0 |
| hinge_matched_adm | mixture CE + information hinge | hinge + MMD | old Lmax, no ramp | 0 |
| v2_no_discrepancy | mixture CE + information hinge | hinge only | 1, ramp 200 | 0 |
| v2_fixed | mixture CE + information hinge | deterministic fixed feasible S | 1, ramp 200 | 0 |
| v2_random | mixture CE + information hinge | one random feasible S | 1, ramp 200 | 0 |

At the user's request this is now 24 seed-0 runs: four targets for each of the
six arms. The eight originally queued learned-v2 runs at seeds 1 and 2 were
cancelled before starting. Historical v1 seed-0 results are in
`pacs_eda_lodo_seed0`; they are not rerun or relabeled. All comparisons remain
single-seed and exploratory. The runner defaults to `--seeds 0`; extra learned
replications require explicit `--seeds 0 1 2`.

Every launch freezes configs and a Python source snapshot before any target
scores are observed. SHA-256 hashes identify snapshot files. Pending jobs run
from that snapshot, so further workspace edits cannot change the protocol.
Dataset paths remain external. All methods use the same domain splits, seed
protocol, backbone, expert count, pair budget and training steps. Checkpoint
selection uses source validation only; final target evaluation follows selection.
The suite stops on a failed job, leaving its log and queue state. It does not
silently retry failures, overwrite old runs, or resume a partial neural state.
`--resume --seeds 0` can replace a stopped queue coordinator, adopting its
existing worker without restarting training. It cancels queued excluded seeds.
For adopted processes the exit code is unavailable; completion is recorded only
after the process exits and full-step metrics plus final results are verified.

`summary.json` updates after each completed job. Across-seed mean/std are computed
from complete four-target seed means, never from partial seeds or by treating
domains as independent seeds. `queue.json` and `report.md` track all arms.

## EDA additions

Structure logs separate the runner-up gap into predictive, MMD and router
components; the last must be exactly zero in v2. They also show the support
selected without discrepancy on the **same probe**, tie counts, per-expert
support risk/entropy/threshold/violation, and the selected predictive score.
A counterfactual same-probe support is diagnostic; the separately retrained
no-discrepancy arm is the causal intervention.

Validation logs evaluate fixed training thresholds using held-out source risks.
The mixed-distribution quantity `H_train-R_val` is explicitly only an estimate,
not a population lower bound. The new `structure_information.png` figure shows
gap components and active-expert constraint violations. Existing disagreement,
feature variance, utilization and gradient diagnostics remain. New diagnostics
include assigned-but-unused experts at a 1% mean-routing cutoff, mixture vs best
expert/oracle gaps, uniform-logit mixture accuracy and per-class oracle accuracy.
The 1% cutoff is descriptive only. Target oracle diagnostics use labels and are
not deployable inference mechanisms.

Fixed-representation synthetic tests implement a shared bounded W and normalized
features with known overlapping supports, a single-global optimum and a no-gap
case. They verify exact support recovery, router interventions, and loss of
identifiability when discrepancy is removed. They test the structure solver's
mechanism, not an unconditional claim of end-to-end latent support recovery.

Local verification: 57 tests passed across the five test modules above. All six
GPU smoke arms completed four updates, source selection, checkpoint reload and
target evaluation; their structure logs had exactly zero router gap contribution
and their reports included `structure_information.png`. Smoke artifacts are in
`predictive_support_outputs/pacs_revision_smoke_v2`. The initial snapshot smoke
exposed an import dependency on the unrelated algorithm registry; the support
model now imports its transformer directly, and the transformer resolves its
helper through the package rather than requiring that registry's path mutation.
