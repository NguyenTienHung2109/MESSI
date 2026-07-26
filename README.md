# MESSI

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
    --output_dir /path/to/output
```

## Evaluation

Evaluate a checkpoint on one complete environment:

```sh
python -m domainbed.scripts.eval \
    --env 0 \
    --dir_dataset /path/to/datasets/PACS \
    --dir_ckpt /path/to/model.pt
```
