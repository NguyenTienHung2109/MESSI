#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the physical GPU ID, e.g. 1}"
PYTHON=${PYTHON:-python}
NUM_CPU_THREADS=${NUM_CPU_THREADS:-1}
if ! [[ "$NUM_CPU_THREADS" =~ ^[1-8]$ ]]; then
  echo 'NUM_CPU_THREADS must be 1..8, within the aggregate user limit.' >&2
  exit 2
fi
export OMP_NUM_THREADS="$NUM_CPU_THREADS" MKL_NUM_THREADS="$NUM_CPU_THREADS" OPENBLAS_NUM_THREADS="$NUM_CPU_THREADS"
DATASETS=(PACS VLCS OfficeHome TerraIncognita DomainNet)
case "${1:-}" in
  prepare)
    : "${DATA_DIR:?Set DATA_DIR to the directory containing dataset folders}"
    : "${CLIP_CHECKPOINT:?Set CLIP_CHECKPOINT to the official ViT-B-16.pt}"
    shift
    dataset=${1:-all}
    if [[ $# -gt 0 ]]; then shift; fi
    if [[ "$dataset" == all ]]; then
      for name in "${DATASETS[@]}"; do
        "$PYTHON" scripts/prepare_clip_fewshot.py --dataset "$name" --data-dir "$DATA_DIR" --checkpoint "$CLIP_CHECKPOINT" --device "${PREP_DEVICE:-cuda:0}" "$@"
      done
    else
      exec "$PYTHON" scripts/prepare_clip_fewshot.py --dataset "$dataset" --data-dir "$DATA_DIR" --checkpoint "$CLIP_CHECKPOINT" --device "${PREP_DEVICE:-cuda:0}" "$@"
    fi
    ;;
  run)
    shift
    exec "$PYTHON" scripts/run_fewshot_matrix.py --datasets "${DATASETS[@]}" --backbones clip_vit_b16 \
      --methods clip_zeroshot tip_adapter_f_imagenet_fixed gda_clip lp_pp_frozen clap_frozen proker_imagenet_fixed timo_fixed \
      --settings single_source --scopes total --shots 1 2 4 8 16 --seeds 42 43 44 45 46 \
      --training-device "${TRAINING_DEVICE:-cpu}" --rff-device "${RFF_DEVICE:-cpu}" --proker-rff-on-large "$@"
    ;;
  *)
    echo "Usage: $0 prepare {all|PACS|VLCS|OfficeHome|TerraIncognita|DomainNet} [options] | run [options]" >&2
    exit 2
    ;;
esac
