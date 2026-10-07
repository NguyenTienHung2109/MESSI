# Isolated PACS H1–H6 experiments

All new code/configuration is in this folder. Existing DomainBed/SIRM files and
`docs/sirm_pacs_h1_h6_runbook.md` are unchanged. Invoke these modules from the
repository root using the existing `dg` environment:

```bash
conda activate dg
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled
```

## Protocol

- Four leave-one-domain-out targets; IDs 0 art_painting, 1 cartoon, 2 photo,
  3 sketch. Seeds 0–2 pilot, 0–4 confirmation.
- **DomainBed 80/20 source split**, using `misc.seed_hash(seed, env)` and the
  same index permutation as `misc.split_dataset`. Train bank on all source-in;
  select by equal-domain mean source-out accuracy, earliest checkpoint on ties.
- Teacher/router/global weights fit on source-in; router temperature/epoch/
  fixed-pair selection use equal-domain source-out NLL. No target fitting or
  selection. All six experts are computed densely.
- For H6 linear probes only, partition source-in by domain/class into two
  deterministic halves: probe-fitting images and statistic-evaluation images.
  Both halves remain bank training data. This is **not** a replacement of the
  DomainBed bank split. It prevents Q-Regret probes from scoring their own fit
  images. Independent invariance diagnostics use source-out.
- Full distilled DeiT-Small pretrained, six experts and separate heads,
  Top-2, dense warmup 500, 5001 steps, batch 32/source, LR 3e-5, WD 1e-6.
  This replaces the old smoke's legacy DeiT variant only in this new pipeline.
- Auxiliary KL calls router on detached projected features. Main CE retains
  encoder/router/expert gradients; SIRM detaches Q and responsibilities while
  retaining encoder/projection/expert/head gradients.
- H6 Random/Risk/Grad/Regret enumerate the same feasible supports: two edges
  per source domain, each active expert assigned at least two domains,
  at most three active experts, uniform weights within the selected support.
  FixedPairs uses slots 0/1/2 for pairs (0,1)/(0,2)/(1,2). Global uses all six
  and is a separate structural branch. Architecture capacity remains six.
- Q-Grad uses analytic CE gradients of existing expert classifier weights and
  biases, averaged over supported classes. Q-Regret fits equally regularized,
  standardized linear probes with one common scale per expert and equal-domain
  fitting weights; convergence is checked. Raw negative gaps and image SE are
  retained. Joint statistics/features refresh every 500 steps; Random is fixed
  by seed. Frozen tuning has the same candidate/epoch budget for all criteria.
- H2 balanced_views and H3 border are separate retrained cohorts. H5 Top-1
  retrains from the same initialization/split/schedule as Top-2.

## Implemented controls

| Hypothesis | Controls and outputs |
|---|---|
| H1 | Uniform, learned global, native, teacher input router, CE/correctness single-expert oracles; temperature selection; standardized-feature noise 0/.25/.5/1; feature shuffle; expert-loss margin |
| H2 | Paired original/grayscale/blur caches; four locked mixture weights; accuracy/NLL/regret per view and worst-mixture accuracy; standard and balanced_views banks |
| H3 | Original, router-only intervention with original expert logits, full intervention; JS and support changes; border source correct/swapped/removed/random, target colors independent of class and stable per parent |
| H4 | Uniform-six, global, best of 15 source-selected fixed pairs, input Top-2/dense; global and domain/class route shuffles; paired deltas and per-class metrics |
| H5 | Top-1/Top-2/dense/uniform/best global expert, separately named oracles, 15-pair grid oracle; remove expert and refit router; disagreement; Top-1 joint retrain; measured compute |
| H6 | Six frozen Q controls and six joint train variants; independent source-validation classifier gap and gradient conflict; Q supports, churn, loads, ESS, SIRM activity, compute |

PACS has no ground-truth mechanism labels. These are proxy tests, not evidence
of causal mechanism recovery. Diagnostic target-label oracles and within-class
shuffles run only during final evaluation.

## Checks and quick smoke

```bash
python -m experiments.sirm_h1_h6.preflight \
  --output experiments/sirm_h1_h6/artifacts/preflight.json

python -m experiments.sirm_h1_h6.suite --mode smoke --targets 1 --seeds 1 \
  --variants native balanced_views border top1 \
    Q_Random Q_Risk Q_Grad Q_Regret Q_FixedPairs Q_Global \
  --output experiments/sirm_h1_h6/artifacts/my_smoke
```

Smoke uses four training steps, 56 images/domain for cache/validation, short
router fitting, and compressed warmup/update intervals. Its artifacts are
marked `smoke`; reports exclude them by default. A smoke validates execution,
not accuracy or scientific hypotheses. Every suite output must be new.

## Start experiments

Start with one native bank per target/seed: **12 full training runs**, with all
frozen H1/H4/H5/H6 controls, H2a and H3a evaluated on those same banks.

```bash
python -m experiments.sirm_h1_h6.suite --mode pilot --variants native \
  --output experiments/sirm_h1_h6/artifacts/pilot_native
```

To include all retrained cohorts and six joint H6 variants: **120 pilot runs**
(10 bank variants × 4 targets × 3 seeds). The frozen controls are not counted
as additional full training runs. Review the exact matrix without running:

```bash
python -m experiments.sirm_h1_h6.suite --mode pilot \
  --variants native balanced_views border top1 \
    Q_Random Q_Risk Q_Grad Q_Regret Q_FixedPairs Q_Global \
  --output experiments/sirm_h1_h6/artifacts/pilot_all --dry-run
```

Remove `--dry-run` to execute. `--mode confirm` uses seeds 0–4; restrict variants
with `--variants`. The launcher always includes native to select H6 joint
assignment weights from its **source-only** frozen stage, before any target
pixels for that target/seed are accessed. `plan.json` records every command;
source-file hashes are checked between stages. Changing code during a suite
stops the launcher.

Do not select methods from target pilot scores and describe confirmation on
those same targets as untouched. Freeze the scientific decision rule first.
No particular accuracy or hypothesis outcome is guaranteed by preflight.

## Individual stages and source intervention review

```bash
python -m experiments.sirm_h1_h6.run train --target-env 1 --seed 0 \
  --output experiments/sirm_h1_h6/artifacts/bank_env1_seed0

python -m experiments.sirm_h1_h6.run cache --source-only \
  --bank experiments/sirm_h1_h6/artifacts/bank_env1_seed0 \
  --output experiments/sirm_h1_h6/artifacts/bank_env1_seed0/cache

python -m experiments.sirm_h1_h6.run fit \
  --cache experiments/sirm_h1_h6/artifacts/bank_env1_seed0/cache \
  --output experiments/sirm_h1_h6/artifacts/bank_env1_seed0/routers.pt

python -m experiments.sirm_h1_h6.source_review \
  --cache experiments/sirm_h1_h6/artifacts/bank_env1_seed0/cache \
  --routers experiments/sirm_h1_h6/artifacts/bank_env1_seed0/routers.pt \
  --output experiments/sirm_h1_h6/artifacts/bank_env1_seed0/source_review.json
```

Inspect source recognizability and severity of grayscale/blur/style before
choosing an intervention configuration. `config.json` contains explicit mild
starting levels, locked before target evaluation. Source accuracy is a proxy;
inspect examples on source if the label-preservation assumption is doubtful.
If changing intervention settings, use a new locked config and new output.

After locking source-selected bank, routers and transform settings:

```bash
python -m experiments.sirm_h1_h6.run cache --target-only \
  --bank experiments/sirm_h1_h6/artifacts/bank_env1_seed0 \
  --routers experiments/sirm_h1_h6/artifacts/bank_env1_seed0/routers.pt \
  --output experiments/sirm_h1_h6/artifacts/bank_env1_seed0/cache

python -m experiments.sirm_h1_h6.run eval \
  --cache experiments/sirm_h1_h6/artifacts/bank_env1_seed0/cache \
  --routers experiments/sirm_h1_h6/artifacts/bank_env1_seed0/routers.pt \
  --output experiments/sirm_h1_h6/artifacts/bank_env1_seed0/evaluation

python -m experiments.sirm_h1_h6.profile \
  --bank experiments/sirm_h1_h6/artifacts/bank_env1_seed0 \
  --output experiments/sirm_h1_h6/artifacts/bank_env1_seed0/compute.json
```

Target cache exports the **entire** held-out domain for full runs. Source image
hashes, split IDs, checkpoint hash, router hash, view identity and paired parent
IDs are checked. A second fitting stage after target cache creation is rejected.
Do not use `--limit`, `--steps`, or `--smoke` for official runs.

## Resume and reporting

An atomic `resume.pt` is saved at every validation checkpoint with model,
optimizer and RNG states. Resume the interrupted **train** stage with the same
arguments plus `--resume`. Changing config, image bytes, split, steps, cohort,
seed, source code or package freeze is rejected. Augmentation is seeded by
sample occurrence/step, independent of worker scheduling. A completed run
cannot be extended silently. Other stages use fresh output and never overwrite
selected routers, caches or evaluations.

```bash
python -m experiments.sirm_h1_h6.summarize \
  --root experiments/sirm_h1_h6/artifacts/pilot_all \
  --output experiments/sirm_h1_h6/artifacts/pilot_all/hypotheses_report.json
```

Reports include per-target mean/SD, mean of all four targets per seed, paired
seed deltas and completeness flags. Cross-bank paired comparisons verify split,
initialization and sample-schedule hashes. Per-image paired bootstrap in each
run is over parent images, not transformed images; training uncertainty comes
from seeds. `profile.py` reports actual dense inference time/VRAM and profiler
matmul/conv FLOPs, explicitly excluding unsupported operations.

Artifacts also include full config, split manifest with image SHA256s, Git SHA,
pretrained SHA256, package freeze, executed source snapshot, checkpoints,
assignment logs, source/target metrics and per-parent predictions/routing.
