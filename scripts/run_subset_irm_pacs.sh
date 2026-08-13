#!/usr/bin/env bash
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gmoe
set -u

CONFIG="${CONFIG:-configs/subset_irm_pacs.json}"
DATA_DIR="${DATA_DIR:-./domainbed/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-subset_irm_outputs/full}"
RUNS=("$@")
if [[ ${#RUNS[@]} -eq 0 ]]; then
  RUNS=(B0 B1 B2 B3 B4 B5 B6)
fi

nvidia-smi
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required. Do not fall back to CPU training.")
print("GPU:", torch.cuda.get_device_name(0))
PY

for run in "${RUNS[@]}"; do
  out="${OUTPUT_ROOT}/${run}"
  if [[ -f "${out}/summary.json" ]]; then
    echo "[skip] ${run} already complete"
    continue
  fi
  if [[ -e "${out}" ]]; then
    echo "refusing non-empty/incomplete output: ${out}" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
    python -m domainbed.scripts.train_subset_irm_pacs \
      --config "$CONFIG" --data-dir "$DATA_DIR" \
      --run "$run" --output-dir "$out"
done
