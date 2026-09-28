#!/usr/bin/env bash
set -euo pipefail

python -m domainbed.scripts.train_subset_irm_pacs \
  --config configs/sirm_res_pacs.json \
  --run S-IRM-res_PACS_smoke \
  --data-dir domainbed/data \
  --output-dir subset_irm_outputs/sirm_res_pacs_smoke
