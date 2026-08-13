#!/usr/bin/env bash
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gmoe
set -u
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0

CONFIG="configs/subset_irm_pacs.json"
DATA_DIR="./domainbed/data"
ROOT="subset_irm_outputs/smoke"
runs=(Smoke0 Smoke1 Smoke2 Smoke3 Smoke4 Smoke5)
steps=(30 30 100 200 220 500)

for index in "${!runs[@]}"; do
  run="${runs[$index]}"
  out="${ROOT}/${run}"
  if [[ -f "${out}/summary.json" ]]; then
    echo "[skip] ${run} already complete"
    continue
  fi
  if [[ -e "$out" ]]; then
    echo "refusing incomplete smoke output: $out" >&2
    exit 1
  fi
  python -m domainbed.scripts.train_subset_irm_pacs \
    --config "$CONFIG" --data-dir "$DATA_DIR" --run "$run" \
    --steps "${steps[$index]}" --checkpoint-freq "${steps[$index]}" \
    --output-dir "$out"
done
