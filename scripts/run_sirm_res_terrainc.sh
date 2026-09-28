#!/usr/bin/env bash
set -euo pipefail

python -u -m domainbed.scripts.train_subset_irm_pacs \
  --config configs/sirm_res_terrainc.json \
  --data-dir ./domainbed/data \
  --run S-IRM-res_Terra_env0_seed0 \
  --output-dir subset_irm_outputs/sirm_res_terrainc/S-IRM-res_Terra_env0_seed0 \
  --target-env 0
