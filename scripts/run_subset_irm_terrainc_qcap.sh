#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-dg}"

CONFIG="${CONFIG:-configs/subset_irm_terrainc_qcap.json}"
DATA_DIR="${DATA_DIR:-./domainbed/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-subset_irm_outputs/terrainc_qcap_tune}"
LOG_DIR="${LOG_DIR:-subset_irm_outputs/tmux_logs}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

runs=(
  TerraQCapLambda2
  TerraQCapLambda1
  TerraQCapLambda0p5
  TerraQCapLambda0p1
)

for target_env in 0 1 2 3; do
  for run in "${runs[@]}"; do
    name="${run}_env${target_env}_seed0"
    out="${OUTPUT_ROOT}/${name}"
    log="${LOG_DIR}/${name}.log"
    if [[ -f "${out}/summary.json" ]]; then
      echo "[skip] ${name} already complete"
      continue
    fi
    if [[ -e "$out" ]]; then
      echo "[stop] incomplete output already exists: $out" >&2
      exit 1
    fi
    echo "[start] ${name} -> ${log}"
    if CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
        python -u -m domainbed.scripts.train_subset_irm_pacs \
          --config "$CONFIG" \
          --data-dir "$DATA_DIR" \
          --run "$run" \
          --target-env "$target_env" \
          --output-dir "$out" \
          >"$log" 2>&1; then
      echo "[done] ${name}"
    elif [[ -f "${out}/q_collapse.json" ]]; then
      collapsed_out="${out}_collapsed"
      collapsed_log="${log%.log}_collapsed.log"
      mv "$out" "$collapsed_out"
      mv "$log" "$collapsed_log"
      echo "[collapse] ${name} -> ${collapsed_out}; continuing"
    else
      echo "[fail] ${name}; see ${log}" >&2
      exit 1
    fi
  done
done
