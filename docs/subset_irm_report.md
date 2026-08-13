# Subset-IRM for MESSI: implementation and PACS evaluation

All results in this report are from the requested single run: PACS, seed 0,
held-out environment index 1, and GPU training. Source environments were the
only environments used for training, risk estimation, assignment, coefficient
calibration, and checkpoint selection. The held-out environment was evaluated
once per run after reloading the source-selected checkpoint.

## A. Repository audit

The detailed pre-edit audit is in `docs/subset_irm_repository_audit.md`.

The current MESSI implementation is `domainbed.algorithms.MESSI`, an
inference-compatible subclass of `GMOE_InvMMD`. A shared DeiT featurizer emits
the backbone feature. `ExplicitMoEHead` then evaluates six independent MLP
experts, computes a dense softmax with a linear router, mixes the six expert
features, and applies one shared linear classifier. Before this work, all six
experts were evaluated and all normally received classification gradients.

The baseline objective in this worktree is

`L_cls + lambda_inv L_ssi + lambda_sp L_sp + lambda_bal L_bal + lambda_div L_div`,

where `L_cls` is shared-classifier CE, `L_ssi` is the worktree's
routing-weighted class-conditional MMD (despite some project terminology
calling it OT), `L_sp` is mean router entropy, `L_bal` penalizes deviation from
uniform mean load, and `L_div` penalizes cross-expert feature correlation.

The actual PACS implementation declares `PACS.ENVIRONMENTS = ["A", "C", "P",
"S"]`, mapping to art painting, cartoon, photo, and sketch. Thus environment
index 1 is **C (cartoon)**. Training used source indices 0, 2, and 3 only.

The generic training entry point evaluates target splits during training, so a
dedicated runner was added. It fixes the source sampling schedule, evaluates
only source validation splits at checkpoints, saves the best source mean,
reloads that checkpoint, and only then iterates over the target evaluation
view. Target metrics do not participate in stopping or selection.

## B. Files changed

| Path | Purpose | Baseline effect |
| --- | --- | --- |
| `docs/subset_irm_repository_audit.md` | Pre-implementation architecture, loss, data, selection, and gradient-path audit. | None. |
| `domainbed/subset_irm.py` | Testable risk, ESS, Top-k, Q, router KL, IRMv1 modules and the gated `MESSISubsetIRM` algorithm. | None when disabled; disabled `update` and `_forward` delegate to MESSI. |
| `domainbed/algorithms.py` | Registers and lazily resolves `MESSI_SubsetIRM`. | Adds a new algorithm name only. |
| `domainbed/hparams_registry.py` | Adds explicit Subset-IRM defaults, all disabled or zero. | Existing algorithm defaults unchanged. |
| `domainbed/test/test_subset_irm.py` | Sixteen focused module and integration tests. | Test-only. |
| `configs/subset_irm_pacs.json` | Fixed PACS protocol, smokes, calibration, and B0--B6 ladder. | None. |
| `domainbed/scripts/train_subset_irm_pacs.py` | GPU-only, deterministic, source-selected experimental runner with JSON diagnostics and heatmaps. | Separate entry point. |
| `scripts/run_subset_irm_smokes.sh` | Sequential smoke-test launcher. | None. |
| `scripts/run_subset_irm_pacs.sh` | Full-run launcher with CUDA verification and completed-run skipping. | None. |
| `subset_irm_outputs/full/summary.csv` | Consolidated machine-readable full-ladder results. | Artifact only. |
| `docs/subset_irm_report.md` | This report. | None. |

The worktree contained unrelated pre-existing modifications; they were
preserved. In particular, only the new registration/default blocks in the two
pre-existing Python files above belong to this change.

## C. Mathematical implementation

For sample `n`, expert `m`, source domain `e`, expert logits `l_nm`, class
target `y_n`, and per-sample Top-k responsibility `gamma_nm`, the implementation
uses

`R_em = sum[n:d_n=e] gamma_nm CE(l_nm,y_n) / (sum[n:d_n=e] gamma_nm + eps)`

and

`ESS_em = (sum gamma_nm)^2 / (sum gamma_nm^2 + eps)`.

Responsibilities are detached by default. Unsupported or ESS-below-2 cells
are invalid and safely produce zero risk rather than NaN.

The expert predictive loss is

`L_expert = mean_n sum_m stopgrad(gamma_nm) CE(l_nm,y_n)`.

Every 100 steps, all six expert heads are evaluated without backpropagation on
the current source-only minibatches. Their all-expert risks update an EMA:

`Rbar_em <- beta Rbar_em + (1-beta) Rhat_em`, with `beta=0.9`.

For each source row, only the two lowest-risk experts survive:

`q_e = Top2Softmax(-Rbar_e / tau_q)`, with `tau_q=1`.

`Q` is detached and has shape 3 x 6; there is no cartoon row. The debug
`routing_mass` assignment is also implemented but was not used in full runs.

Router distillation uses dense, pre-Top-k probabilities to avoid logs of zero:

`L_route = mean_n KL(stopgrad(q_{d_n}) || pi(x_n))`.

For SIRM, a scalar `s=1` multiplies each valid expert/domain cell's logits,
`g_em = d R_em(s) / ds`, computed with `create_graph=True`. Only experts with
at least two assigned, ESS-valid source domains contribute:

`L_SIRM = sum_em q_em I_valid g_em^2 / (sum_em q_em I_valid + eps)`.

`Q` and routing responsibilities are detached. SIRM is disabled for the first
500 steps and returns a device-correct zero if no expert is eligible.

For expert-logit prediction, exact per-sample Top-2 masks the dense router and
renormalizes the two retained weights, then computes

`l_mix = sum_m gamma_nm l_nm`.

During the first 500 of 5,001 steps, routing stays dense. The implementation is
gradient-sparse but compute-dense: every expert is evaluated, while exact
masking ensures non-selected expert paths receive zero loss gradient for that
sample. In auxiliary mode the original dense feature mixture and shared
classifier remain the final predictor. All expert classifier heads copy the
shared classifier's initial weights and bias.

The clean full objective is

`L_mix + 0.1 L_expert + 0.12 L_route + 0.25 L_SIRM + 0.02 L_sp + 0.02 L_bal`.

The two routing regularizers were retained for stability. `L_ssi` and `L_div`
are zero in B1--B5. B6 adds `0.01 L_ssi`; B0 retains the exact current MESSI
coefficients (`0.01, 0.02, 0.02, 0.02` for SSI, sparsity, balance, diversity).

## D. Test and calibration results

Final regression command:

```bash
conda activate gmoe
python -m unittest -q domainbed.test.test_subset_irm domainbed.test.test_moe_losses domainbed.test.test_rmessi_invot
python -m py_compile domainbed/subset_irm.py domainbed/scripts/train_subset_irm_pacs.py
python -m json.tool configs/subset_irm_pacs.json >/dev/null
bash -n scripts/run_subset_irm_pacs.sh scripts/run_subset_irm_smokes.sh
git diff --check
```

Result: **40 tests passed in 1.231 s**, and compilation, JSON validation,
shell syntax, and whitespace checks all passed.

The 16 new tests cover: six identical heads/shape/device; auxiliary equivalence;
expert-logit output shape; exact normalized deterministic per-sample Top-k;
per-sample selected/non-selected gradient masking; manual weighted risk;
one-hot and soft responsibility; mixed/unequal domains; zero support and ESS;
CPU FP32 and CUDA autocast; expert-loss gradient isolation; source-only EMA and
Top-r Q; invalid risk entries; router KL direction/gradients/zero teacher
entries; shared versus conflicting mechanisms; inactive SIRM cases; and finite
first/second-order gradients with no detached router gradient.

All mandatory GPU smokes passed on real PACS source batches:

| Smoke | Steps | Key outcome |
| --- | ---: | --- |
| 0 baseline | 30 | Finite baseline, 6.92 GB, source set excludes index 1. |
| 1 zero new coefficients | 30 | Same first-batch total and component losses as Smoke 0; integration test gives exact prediction equivalence. |
| 2 Top-2 | 100 | Exactly two active experts; CE decreased; no NaN. |
| 3 + expert CE | 200 | Expert CE decreased and source validation reached 73.46%. |
| 4 + router KL | 220 | KL fell from 1.232 to 0.239; four experts received traffic, not one. |
| 5 + SIRM | 500 | SIRM finite and active on 96% of steps under the short smoke anneal; source validation reached 83.42%. |

One infrastructure issue was found: 24 aggregate DataLoader workers stalled a
Smoke 3 attempt at step 101. No model failure occurred. The stalled artifact
was archived, worker count was reduced, and the smoke reran successfully. A
separate isolation issue was also fixed: when `lambda_sirm=0`, the code now
does not construct an unused second-order SIRM graph.

The 200-step calibration did not evaluate the target (`target_accuracy=null`).
Median raw backbone gradient norms were 5.872 (mix), 5.367 (expert), 4.920
(route), and 2.072 (SIRM). Median raw losses were 1.129, 1.242, 0.249, and
0.0447. Conservative frozen coefficients were 0.10, 0.12, and 0.25,
corresponding to approximately 8.8%, 9.8%, and 8.8% of the mixture backbone
gradient during calibration. Median raw expert-module norms were 1.189 (mix),
1.452 (expert), 0 (route, by design), and 0.601 (SIRM).

Training diagnostics contain isolated raw and coefficient-weighted gradient
norms for each new loss into backbone, router, experts, and expert heads. The
old baseline/SSI terms retain their scalar logs, but the completed runs do not
contain separately isolated old-loss gradient norms; this is the one logging
limitation relative to the requested exhaustive list.

## E. Exact training commands

The environment/GPU check used before training was:

```bash
conda activate gmoe
nvidia-smi
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required. Do not fall back to CPU training.")
print("GPU:", torch.cuda.get_device_name(0))
PY
```

The device was an NVIDIA GeForce RTX 5070 Ti with 16,303 MiB (driver 595.58.03);
PyTorch 2.8.0+cu128 reported one CUDA device. The exact full-run commands were:

```bash
conda activate gmoe
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python -m domainbed.scripts.train_subset_irm_pacs \
  --config configs/subset_irm_pacs.json \
  --data-dir ./domainbed/data \
  --run B0 --output-dir subset_irm_outputs/full/B0
```

```bash
conda activate gmoe
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python -m domainbed.scripts.train_subset_irm_pacs \
  --config configs/subset_irm_pacs.json \
  --data-dir ./domainbed/data \
  --run B1 --output-dir subset_irm_outputs/full/B1
```

```bash
conda activate gmoe
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python -m domainbed.scripts.train_subset_irm_pacs \
  --config configs/subset_irm_pacs.json \
  --data-dir ./domainbed/data \
  --run B2 --output-dir subset_irm_outputs/full/B2
```

```bash
conda activate gmoe
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python -m domainbed.scripts.train_subset_irm_pacs \
  --config configs/subset_irm_pacs.json \
  --data-dir ./domainbed/data \
  --run B3 --output-dir subset_irm_outputs/full/B3
```

```bash
conda activate gmoe
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python -m domainbed.scripts.train_subset_irm_pacs \
  --config configs/subset_irm_pacs.json \
  --data-dir ./domainbed/data \
  --run B4 --output-dir subset_irm_outputs/full/B4
```

```bash
conda activate gmoe
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python -m domainbed.scripts.train_subset_irm_pacs \
  --config configs/subset_irm_pacs.json \
  --data-dir ./domainbed/data \
  --run B5 --output-dir subset_irm_outputs/full/B5
```

```bash
conda activate gmoe
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python -m domainbed.scripts.train_subset_irm_pacs \
  --config configs/subset_irm_pacs.json \
  --data-dir ./domainbed/data \
  --run B6 --output-dir subset_irm_outputs/full/B6
```

The optional B6 command was executed only after B4 proved stable. The
configuration fixes seed 0, 5,001 steps, checkpoint frequency 300, batch size
32 per source environment, Adam at 3e-5 with weight decay 1e-6, standard PACS
augmentation, six experts, and a 500-step dense-routing/SIRM warmup.

## F. Experimental results

Accuracies are percentages. Active SIRM rate is cumulative at the selected
checkpoint, so its maximum is approximately 90% because of the 500-step
anneal. Peak memory is allocated GPU memory.

| Run | Prediction | Top-k | Expert | Route | SIRM | SSI | Source val | Env 1 OOD | Active SIRM | Peak GB | Selected step | Wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | shared feature | dense | 0 | 0 | 0 | 0.01 | 87.48 | 55.38 | 0.00% | 6.92 | 4800 | 28.87 min |
| B1 | expert logits | 2 | 0 | 0 | 0 | 0 | 89.77 | 55.28 | 0.00% | 7.23 | 4800 | 16.56 min |
| B2 | expert logits | 2 | 0.10 | 0 | 0 | 0 | 89.39 | 54.80 | 0.00% | 7.23 | 4800 | 16.57 min |
| B3 | expert logits | 2 | 0.10 | 0.12 | 0 | 0 | 89.20 | 56.40 | 0.00% | 7.23 | 4800 | 16.55 min |
| B4 | expert logits | 2 | 0.10 | 0.12 | 0.25 | 0 | 89.46 | **57.41** | 87.18% | 7.23 | 3900 | 16.81 min |
| B5 | shared feature | 2 | 0.10 | 0.12 | 0.25 | 0 | 89.63 | 54.32 | 89.59% | 7.23 | 4800 | 16.91 min |
| B6 | expert logits | 2 | 0.10 | 0.12 | 0.25 | 0.01 | 89.01 | **57.41** | 90.00% | 7.24 | 5000 | 31.43 min |

Per-run `train_log.jsonl`, detailed `diagnostics.jsonl`, manifests, summaries,
source-selected diagnostics, checkpoints, and R/Q heatmaps are under
`subset_irm_outputs/full/B*/`. B0, which deliberately uses the untouched
baseline class, has the baseline train log and summary rather than new-module
diagnostics.

## G. Mechanism analysis

The B4 source-selected EMA risk matrix is structured. For source A, experts 1
and 2 have risks 0.119 and 0.112; for P, experts 1 and 3 have 0.056 and 0.057;
for S, experts 0 and 3 have 0.391 and 0.397. The remaining cells, especially
experts 4 and 5, are markedly worse.

Accordingly, B4 learned these Top-2 Q rows:

| Source | e0 | e1 | e2 | e3 | e4 | e5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A (art) | 0 | 0.498 | 0.502 | 0 | 0 | 0 |
| P (photo) | 0 | 0.500 | 0 | 0.500 | 0 | 0 |
| S (sketch) | 0.502 | 0 | 0 | 0.498 | 0 | 0 |

This is an overlapping subset structure rather than one expert per domain:
expert 1 covers A/P and expert 3 covers P/S, while experts 2 and 0 specialize
more narrowly. Each SIRM-eligible shared expert therefore spans two source
domains. There is no target row.

The assigned B4 source-validation expert accuracies support the risk structure:
expert 1 scores 89.49%/97.01% on A/P; expert 3 scores 97.31%/81.78% on P/S;
expert 2 scores 89.00% on A; and expert 0 scores 81.66% on S. Expert CE sharply
improved the all-expert risk/accuracy profiles relative to B1, so routed experts
became independently useful. It did not make all six experts useful: Top-2 plus
Q consistently routed through four, leaving experts 4 and 5 inactive. This is
a limited four-expert specialization, not a single-expert collapse.

After warmup, every logged sample has exactly two nonzero routing weights.
Router KL decreases during training and B3 improves OOD by 1.60 points over
B2, showing that the source-only assignment teacher was useful. B4's SIRM is
active on every eligible post-warmup step and improves OOD another 1.01 points
over B3. Its selected cumulative active rate is 87.18%, high enough to matter.

B5 obtains slightly higher source validation than B4 but loses 3.09 OOD points.
Thus expert-specific classifiers are useful at inference here; auxiliary-only
heads are not merely equivalent training supervision. B6 exactly ties B4 OOD,
uses nearly identical memory, and takes 31.43 versus 16.81 minutes because the
existing MMD alignment is expensive. It provides no observed complementary
gain at this seed.

## H. Final conclusion

1. **Numerical stability:** yes. All tests and staged smokes passed; B4 and B6
   completed with finite first/second-order gradients, no NaNs, bounded memory,
   and no one-expert collapse.
2. **First harm/improvement:** adding expert CE alone first harmed OOD by 0.48
   points (B2 versus B1), despite improving independent expert competence.
   Router distillation was the first new loss to improve OOD (+1.60 points over
   B2). SIRM then added +1.01 points over B3.
3. **Independent experts:** the four routed experts became independently
   predictive on their assigned domains. The claim does not extend to experts
   4 and 5, which remained inactive.
4. **Interpretable Q:** yes. It contains overlapping A/P and P/S expert subsets,
   with domain-specific companions and no target row.
5. **OOD improvement:** B4 reaches 57.41%, +2.03 points over current MESSI B0,
   +2.13 over the B1 predictive architecture alone, and +1.01 over the full
   predictive/router architecture B3 immediately before SIRM.
6. **Old SSI usefulness:** no gain was observed. The repository's current SSI
   is MMD rather than OT; B6 tied B4 at 57.41% and increased wall time by 87%.
7. **Next implementation:** add capacity-aware assignment/load regularization
   that preserves Q's source-risk meaning while recruiting experts 4 and 5,
   then test accumulated multi-minibatch ESS/SIRM episodes. This should be done
   under the same source-only selection protocol before spending on additional
   seeds or datasets.

These are single-seed results and should not be interpreted as uncertainty
estimates. No additional seeds or datasets were run.
