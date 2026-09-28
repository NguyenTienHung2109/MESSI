#!/usr/bin/env bash
set -euo pipefail

for target_env in 1 2 3; do
  run_name="S-IRM-res_Terra_env${target_env}_seed0"
  output_dir="subset_irm_outputs/sirm_res_terrainc/${run_name}"
  log_file="subset_irm_outputs/tmux_logs/${run_name}.log"
  python -u -m domainbed.scripts.train_subset_irm_pacs \
    --config configs/sirm_res_terrainc.json \
    --data-dir ./domainbed/data \
    --run "$run_name" \
    --output-dir "$output_dir" \
    --target-env "$target_env" > "$log_file" 2>&1
done
