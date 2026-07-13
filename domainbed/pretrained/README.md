# Pretrained backbones

Place the DeiT-distilled checkpoints here for offline / no-outbound-internet
servers. `DeiTFeaturizer` (in `domainbed/deit_transformer.py`) will detect them
automatically and skip the URL download.

Required filenames:
- `deit_tiny_distilled_patch16_224-b40b3cf7.pth`  (~22 MB)
- `deit_small_distilled_patch16_224-649709d9.pth` (~86 MB)
- `deit_base_distilled_patch16_224-df68dfff.pth`  (~318 MB, only needed for deit_base)

Original URLs (download on a machine with internet, then transfer):
- https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth
- https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth
- https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth
