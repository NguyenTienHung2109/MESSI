# Welcome to Generalizable Mixture-of-Experts for Domain Generalization

🔥 Our paper [Sparse Mixture-of-Experts are Domain Generalizable Learners](https://openreview.net/forum?id=RecZ9nB9Q4) has officially been accepted as ICLR 2023 for Oral presentation. 

🔥 GMoE-S/16 model currently [ranks top place](https://paperswithcode.com/sota/domain-generalization-on-domainnet) among multiple DG datasets without extra pre-training data. (Our GMoE-S/16 is initilized from [DeiT-S/16](https://github.com/facebookresearch/deit/blob/main/README_deit.md), which was only pretrained on ImageNet-1K 2012)

Wondering why GMoEs have astonishing performance? 🤯 Let's investigate the generalization ability of model architecture itself and see the great potentials of Sparse Mixture-of-Experts (MoE) architecture.

## Predictive binary subset supports

The router-independent revision with predictive-information thresholds is
documented in [the v2 guide](docs/predictive_support_v2.md), including isolated
ablations and the seed-0 PACS protocol.

The new `MESSI_Support` method learns overlapping domain supports with local
expert supervision, routing admissibility and conditional MMD. See the
[implementation and running guide](docs/predictive_support.md) for PACS /
TerraIncognita configurations, smoke tests and matched interventions.

## Quickstart: K-domain sweeps on iWildCam (WILDS) and MetaShift

### 0. Clone & install

```sh
git clone https://github.com/NguyenTienHung2109/messi.git
cd messi

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade git+https://github.com/microsoft/tutel@main
pip install -r requirements.txt
pip install wilds          # required for iWildCam
```

### 1. iWildCam K-sweep (WILDS)

```sh
# (a) Download iWildCam (~12 GB) → domainbed/data/iwildcam_v2.0/
python -c "from wilds.datasets.iwildcam_dataset import IWildCamDataset; \
           IWildCamDataset(root_dir='domainbed/data', download=True)"

# (b) Pick balanced source/test domains; emits k_sweep_setup.json
python scripts/iwildcam_pick_domains.py

# (c) Run sweep over K ∈ {4, 8, 10, 12, 14, 16}, seed 0
bash scripts/run_iwildcam_k_sweep.sh CORAL              # baseline
bash scripts/run_iwildcam_k_sweep.sh GMOE_InvMMD        # MESSI variant
# Optional 2nd arg = parallel jobs, e.g. `bash ... CORAL 2`
# Outputs: multi_dataset/test_<ALGO>_iwildcam_k/K<K>_seed0/
```

### 2. MetaShift K-sweep

MetaShift needs splits **before** the targeted image extraction (extractor reads split CSVs to know which IDs to pull from the GQA zip).

```sh
# (a) Fetch the metadata pickle (~15 MB) only
python -c "import os, urllib.request; \
           os.makedirs('data/metashift/meta_data', exist_ok=True); \
           urllib.request.urlretrieve( \
             'https://github.com/Weixin-Liang/MetaShift/raw/main/dataset/meta_data/full-candidate-subsets.pkl', \
             'data/metashift/meta_data/full-candidate-subsets.pkl')"

# (b) Build splits for K ∈ {4,6,8} × seeds {0,1,2} (5-class, N=240 per class)
for K in 4 6 8; do for S in 0 1 2; do
  python scripts/metashift_build_splits.py \
    --class-set cat_dog_horse_elephant_bird --K $K --seed $S \
    --total-per-class 240 --single-test
done; done

# (c) Download GQA images.zip (~21.8 GB) and extract only IDs referenced by the splits
python -c "from domainbed.scripts.download import download_metashift; download_metashift('data')"
# (set METASHIFT_KEEP_ZIP=1 in env to retain the 21.8 GB zip after extraction)

# (d) Run sweep (K ∈ {4,6,8}, seeds {0,1,2})
bash scripts/run_metashift_k_sweep.sh ERM
bash scripts/run_metashift_k_sweep.sh GMOE_InvMMD
# Outputs: multi_dataset/test_<ALGO>_metashift_k/K<K>_seed<S>/
```

Per-run logs land in `multi_dataset/logs/`. A run is marked complete when its output dir contains a `done` file; re-running the sweep skips completed cells.



### Preparation

```sh
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

python3 -m pip uninstall tutel -y
python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@main

pip3 install -r requirements.txt
```

### Datasets

```sh
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data
```

### Environments

Environment details used in paper for the main experiments on Nvidia V100 GPU.

```shell
Environment:
	Python: 3.9.12
	PyTorch: 1.12.0+cu116
	Torchvision: 0.13.0+cu116
	CUDA: 11.6
	CUDNN: 8302
	NumPy: 1.19.5
	PIL: 9.2.0
```

## Start Training

Train a model:

```sh
python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/OfficeHome/\
       --algorithm GMOE\
       --dataset OfficeHome\
       --test_env 2
```

## Hyper-params

We put hparams for each dataset into
```sh
./domainbed/hparams_registry.py
```

Basically, you just need to choose `--algorithm` and `--dataset`. The optimal hparams will be loaded accordingly. 

## License

This source code is released under the MIT license, included [here](LICENSE).

## Acknowledgement

The MoE module is built on [Tutel MoE](https://github.com/microsoft/tutel).
