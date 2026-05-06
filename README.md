# MESSI — Anonymous Supplementary Code

Anonymous supplementary code for the submission. **Author and institutional
information has been removed for double-blind review.**

This repository contains the implementation of **MESSI**, a unified
full-objective Mixture-of-Experts variant for domain generalization.
MESSI extends a sparse-MoE backbone with a single hyperparameter
`inv_type ∈ {A, B, MMD, OT, ED}` that selects between five subset-aware
invariance losses, optionally combined with sparsity, load-balance, and
diversity regularizers.

The implementation is built on the public DomainBed benchmark and reuses
parts of Tutel-MoE, timm/DeiT, and the original GMoE codebase. License
headers in individual files preserve upstream attribution.

## Installation

```sh
# Python 3.9+ recommended.
pip install torch torchvision torchaudio
pip install -r requirements.txt
# Tutel MoE (optional; required only for the Tutel-backed routing path).
pip install --upgrade git+https://github.com/microsoft/tutel@main
```

## Datasets

Download the DomainBed datasets to a directory of your choice:

```sh
python -m domainbed.scripts.download --data_dir /path/to/datasets
```

All scripts in this repo accept a `--data_dir` flag or read the
`DATA_DIR` environment variable; defaults assume `./domainbed/data`.

## Training

```sh
python -m domainbed.scripts.train \
    --algorithm MESSI \
    --dataset PACS \
    --test_envs 0 \
    --data_dir /path/to/datasets/PACS \
    --output_dir /path/to/output \
    --hparams '{"inv_type": "A", "lambda_inv": 0.1}'
```

`--algorithm MESSI` supports `inv_type ∈ {A, B, MMD, OT, ED}`. Other
DomainBed baselines (`ERM`, `IRM`, `GroupDRO`, `Mixup`, `MLDG`, `CORAL`,
`MMD`, `DANN`, `CDANN`, `MTL`, `SagNet`, `ARM`, `VREx`, `RSC`, `SD`,
`ANDMask`, `SANDMask`, `IGA`, `SelfReg`, `Fishr`, `TRM`, `IB_ERM`,
`IB_IRM`, `CAD`, `CondCAD`, `Fish`) and the GMoE backbone (`GMOE`) are
also available.

## Logging

Weights & Biases logging is **off by default**. To enable it, set the
following environment variables before running any script:

```sh
export WANDB_API_KEY=<your_key>     # or rely on `wandb login`
export WANDB_PROJECT=<your_project>
export WANDB_ENTITY=<your_entity>   # optional; falls back to your default entity
```

If `WANDB_PROJECT` is unset, the code skips W&B initialization entirely.
To force-disable even when set, use `WANDB_DISABLED=1`.

## Hyperparameters

Default hyperparameters are defined in `domainbed/hparams_registry.py`
and selected automatically based on `--algorithm` and `--dataset`. You
can override individual values via `--hparams '{"key": value}'`.

For MESSI specifically, the relevant entries are:

| key | default | description |
|---|---|---|
| `inv_type` | `'A'` | invariance variant (A, B, MMD, OT, ED) |
| `lambda_inv` | `0.01` | weight on invariance loss |
| `lambda_sp` | `0.01` | weight on sparsity (routing entropy) |
| `lambda_bal` | `0.01` | weight on load-balance loss |
| `lambda_div` | `0.01` | weight on expert-diversity loss |
| `alpha_cov` | `0.1` | covariance term weight (B only) |
| `alpha` | `4.0` | routing-weight temperature (MMD/OT/ED) |
| `mmd_sigmas` | `(1,2,4,8,16)` | RBF bandwidths (MMD only) |
| `ot_epsilon` | `0.1` | Sinkhorn entropy reg (OT only) |
| `sinkhorn_iters` | `50` | Sinkhorn iterations (OT only) |

## Repository layout

```
domainbed/         # algorithms, datasets, training loop, MoE layer
  algorithms.py    # MESSI + DomainBed baselines
  hparams_registry.py
  scripts/train.py # entry point for single runs
  losses/          # MoE / invariance loss helpers
configs/           # YAML configs (relative paths only)
```

## Reproducibility

- All training runs are deterministic given `--seed`, `--trial_seed`,
  `--hparams_seed`, dataset, algorithm, test environment, and pinned
  versions in `requirements.txt`.
- Pretrained ImageNet-1k DeiT weights are downloaded automatically by
  `timm` on first use; cache location is controlled by
  `TORCH_HOME` / `HF_HOME` env vars.
- A short smoke test:
  ```sh
  WANDB_DISABLED=1 python -m domainbed.scripts.train \
      --algorithm MESSI --dataset PACS --test_envs 0 \
      --steps 2 --output_dir /tmp/messi_smoke \
      --hparams '{"batch_size": 8, "inv_type": "A", "lambda_inv": 0.01}'
  ```

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). Individual
files derived from upstream projects (DomainBed, Tutel, timm, DeiT, GMoE)
preserve their original license headers in place.
