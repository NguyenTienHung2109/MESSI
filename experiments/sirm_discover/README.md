# SIRM-Discover v2 prototype

Implements the four requested initial changes in an isolated experimental module. Legacy SIRM code and the stopped Q pilot are unchanged. This is a runnable architecture/objective hypothesis, not a claim of factor recovery or improved OOD accuracy.

## Prediction and discovery

The existing distilled DeiT-Small featurizer produces `z`. A linear global head produces `g(z)`. Six independent MLP residual experts produce class-centered corrections `delta_m`. Hidden layers are independently initialized; final residual layers are zero-initialized to preserve the global prediction initially. The router is softmax over six residual slots plus a NULL slot:

```
logits = global_logits + sum_m pi_m * centered_delta_m
```

Residual probabilities sum to at most one. NULL keeps the global predictor available. There is no Top-2 prediction truncation and no standalone expert CE.

Discovery uses CE(global + delta_m), combined with a normalized soft minimum (log-mean-exp). Its gradients match the proposed unnormalized soft minimum; the normalization adds only a constant. Balancing acts on soft multiple-choice responsibilities and is weighted by detached non-NULL teacher mass, so it does not force residual load when no useful correction is observed.

Router teachers use single-correction utility `CE(global) - CE(global + delta_m)`. The teacher includes NULL and suppresses nonpositive utility through a continuous effect gate. Auxiliary KL uses detached representation and detached teacher; main mixture CE retains its router/residual gradients.

## Utility Q and NULL

Q uses a different statistic from the router teacher: leave-one-out marginal utility of the *current composition*:

```
U_dm = mean_domain [CE(full_logits - pi_m*delta_m) - CE(full_logits)]
```

Dropping a term never renormalizes the other routing weights. EMA updates use source-in training minibatches only, on a slower configurable interval. Each source must supply at least `utility_min_samples` observations in an update. A missing domain/expert statistic leaves the whole Q in NULL rather than imputing an incomplete two-way interaction.

Let `V = U - row_mean(U) - column_mean(U) + grand_mean(U)`. Define `clip01(x) = min(1,max(0,x))`. The prototype uses:

```
p_d = entmax15([V_d / tau_Q, 0])
c_d = clip01(std(V_d)/(std(U_d)+eps)) * clip01(std(U_d)/tau_Q)
evidence_dm = clip01((U_dm-u_min)/tau_Q) * clip01((V_dm-v_min)/tau_Q)
q_dm = c_d * p_dm * evidence_dm
q_null_d = 1 - sum_m q_dm
```

The additional continuous gates make harmful/too-small raw contributions and absent interactions go to NULL without a hard eligible-logit switch. The variance ratio is clipped; it can otherwise exceed one. The final scale factor makes confidence go smoothly to zero for nearly flat rows. These are explicit prototype heuristics, not calibrated statistical confidence or proof that a subset exists. Thresholds and temperatures are fixed in config, not chosen from target results.

Entmax-1.5 uses the full-sort threshold equation, checked against the [authors' reference implementation](https://github.com/deep-spin/entmax/blob/master/entmax/activations.py). No new package is required. Entmax is used only for detached teachers/Q; the prediction router uses softmax.

## Marginal-component IRMv1

For each valid source/expert cell, scale only its routed residual contribution:

```
G_dm = d/ds mean_domain CE(stopgrad(logits_minus_m) + s*stopgrad(pi_m)*delta_m) at s=1
```

All other experts' context, the global predictor, routing and Q are detached in this loss. Risk is unweighted over domain examples; routing enters inside the component, not as a second outer CE weight. Routing ESS excludes unsupported cells. Pair weights are `stopgrad(q_im*q_jm)` for sources i<j:

```
L_sirm = sum_(i<j,m) q_im*q_jm*(G_im^2 + G_jm^2) / number_of_source_pairs
```

The divisor is fixed, not sum of pair weights: confidence/NULL attenuation therefore remains effective. Experts selected for only one source get no pair penalty. This tests scalar-scale stationarity of a contribution, not equality of full optimal classifier vectors or recovered latent factors.

## Three stages and matched controls

Default candidate config: 1000 global steps, 1000 discovery steps, 3001 invariance steps (5001 total), batch 32/source, LR 3e-5, residual hidden width 256. These are initial engineering defaults, not validated research settings.

1. Global: H2 augmentation, train only backbone/global head by CE. Experts, router and utility Q are dormant.
2. Discovery: freeze backbone/global head and keep their training mode off; train residuals/router with mixture CE + soft-min discovery + teacher KL + load balancing. No SIRM. Marginal utility EMA/Q are measured.
3. Invariance: retain the frozen global anchor and discovery losses. Add component SIRM, linearly ramped from zero over 500 steps. OFF changes only `lambda_sirm` to zero and still measures utility/Q.

The all-zero-residual model reduces to the global predictor. Stage 1 is CE training, so this alone is not a mathematical equivalence to global IRM.

The runner reuses the existing PACS 80/20 source splits, H2 augmentation, deterministic augmentation/batch schedules and equal-domain source-out selection. It selects checkpoints only during stage 3 with earliest ties. Target is not loaded by the train/smoke commands. Source-out diagnostics export held-out marginal utilities, interactions, Q/NULL and router mass; they never update Q. Source-out already selected the checkpoint and is not an untouched diagnostic test split.

Checkpoint, optimizer, stage, utility EMA/Q and RNG state are resumable. Resume checks source hashes, effective config, data inventory/splits, batch schedule and package freeze. Model initialization, split, schedule and capacity are audited across ON/OFF.

## Run and current validation

```bash
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/ntduc/miniconda3/envs/dg/bin/python -m unittest -q experiments.sirm_discover.test_discover
PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/sirm-mpl /home/ntduc/miniconda3/envs/dg/bin/python -u -m experiments.sirm_discover.run smoke --output results/sirm_discover/smoke_verified --target 1 --seed 0
```

Use a fresh output directory for a new smoke. The completed `smoke_verified` run is idempotent under its locked plan. A full candidate run can be started explicitly later:

```bash
/home/ntduc/miniconda3/envs/dg/bin/python -u -m experiments.sirm_discover.run train --arm off --target 1 --seed 0 --output results/sirm_discover/manual_off
/home/ntduc/miniconda3/envs/dg/bin/python -u -m experiments.sirm_discover.run train --arm on --target 1 --seed 0 --output results/sirm_discover/manual_on
```

Only tests and the requested paired smoke were run. No full pilot was launched. Existing factorial caches/evaluator assume standalone expert logits; they cannot be reused to evaluate this architecture without changing the composition formula.

12 numerical/gradient/stage/state tests passed. Tests cover an additive no-interaction NULL control, known pair-shared utilities, harmful corrections, exact leave-one-out removal, entmax sparsity/normalization, continuity at effect floors, scalar analytic derivatives, gradient isolation, confidence scaling without normalization cancellation, stage freezing, zero residual fallback and model/optimizer round-trip.

The two 20-step PACS smokes completed. Their initialization, splits, schedule, capacity and all recorded pre-penalty losses/Q/utility statistics matched exactly. ON exercised active pair penalties after the ramp; OFF loss/weight stayed zero. Smoke utility floors are zero and stages are compressed to 4/8/8 to check plumbing; defaults use positive floors. Q remained almost entirely NULL (only tiny residual mass), and both selected step-12 checkpoints were at the zero-ramp boundary because source accuracy tied. This is a pipeline check, not evidence that ON improves accuracy or that factors were learned. The separate synthetic gradient tests exercise nontrivial active penalties.

A first smoke attempt failed in logging before a checkpoint; it remains in `results/sirm_discover/smoke/`. The corrected complete artifacts are in `results/sirm_discover/smoke_verified/`.
