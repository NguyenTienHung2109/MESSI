#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-dg}"

ROOT="${ROOT:-subset_irm_outputs/terrainc_global_expert}"
LOG_ROOT="${LOG_ROOT:-subset_irm_outputs/tmux_logs}"
CONFIG="${CONFIG:-configs/subset_irm_terrainc_global_expert.json}"
DATA_DIR="${DATA_DIR:-./domainbed/data}"
RUN="TerraGlobalInvariantExpert"
mkdir -p "$ROOT" "$LOG_ROOT"

for env in 0 1 2 3; do
  output="$ROOT/${RUN}_env${env}_seed0"
  log="$LOG_ROOT/${RUN}_env${env}_seed0.log"
  if [[ -f "$output/summary.json" ]]; then
    echo "[skip] completed $output"
    continue
  fi
  if [[ -d "$output" ]] && [[ -n "$(find "$output" -mindepth 1 -print -quit)" ]]; then
    echo "[stop] non-empty incomplete output exists: $output" >&2
    exit 1
  fi
  echo "[run] held-out env $env -> $output"
  python -u -m domainbed.scripts.train_subset_irm_pacs \
    --config "$CONFIG" \
    --data-dir "$DATA_DIR" \
    --run "$RUN" \
    --output-dir "$output" \
    --target-env "$env" 2>&1 | tee "$log"
done
