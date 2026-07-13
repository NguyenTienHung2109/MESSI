#!/bin/bash
# iWildCam runner — WILDS official 5-way split (post-2026-05 layout).
#
# Env layout (see WILDSIWildCam / WILDSIWildCamERM in domainbed/datasets.py):
#   env_0 = train pool   (243 locs, ~130k imgs)  — sole source
#   env_1 = val (OOD)    (32 locs,  ~15k imgs)   — model selection
#   env_2 = test (OOD)   (48 locs,  ~43k imgs)   — final report
#   env_3 = id_val       (~7k imgs)              — ID validation diagnostic
#   env_4 = id_test      (~8k imgs)              — ID test diagnostic
#
# Usage:
#   bash scripts/run_iwildcam_paper.sh ERM
#   bash scripts/run_iwildcam_paper.sh GMOE_InvMMD
#   bash scripts/run_iwildcam_paper.sh CORAL 2     # 2 parallel seeds
#
# Reports `env2_out_f1` (OOD test) as the headline metric. Use
# `env1_out_f1` (OOD val) for model selection.

set -u

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 ALGO [PARALLEL]"
    echo "  ALGO ∈ {ERM, IRM, MMD, CORAL, GroupDRO, VREx, DANN, Mixup,"
    echo "          GMOE_InvMMD, GMOE_InvOT, GMOE_InvED, CORAL_CNN, ...}"
    exit 1
fi

ALGO=$1
PARALLEL=${2:-1}
SEEDS=(0 1 2)

DATASET=WILDSIWildCam
if [ "$ALGO" = "ERM" ]; then
    DATASET=WILDSIWildCamERM
fi
TEST_ENVS="1 2 3 4"

# Per-algo settings. With a single source env (env_0), per-step batch is just
# `batch_size` images; pick what fits 16GB VRAM comfortably.
case "$ALGO" in
    ERM|IRM|MMD|GroupDRO|VREx|DANN|Mixup|CORAL)
        STEPS=10000
        BATCH_SIZE=32
        EXTRA_HPARAMS=''
        ;;
    CORAL_CNN)
        STEPS=25000
        BATCH_SIZE=32
        EXTRA_HPARAMS=',"model":"cnn"'
        ;;
    GMOE_InvMMD)
        STEPS=10000
        BATCH_SIZE=64
        EXTRA_HPARAMS=',"model":"deit_tiny_patch16_224","lambda_inv":0.01,"lambda_sp":0,"lambda_bal":0,"lambda_div":0.02,"alpha":4.0'
        ;;
    GMOE_InvOT)
        STEPS=10000
        BATCH_SIZE=64
        EXTRA_HPARAMS=',"model":"deit_tiny_patch16_224","lambda_inv":0.01,"lambda_sp":0,"lambda_bal":0,"lambda_div":0.02,"alpha":4.0,"ot_epsilon":0.1,"sinkhorn_iters":50'
        ;;
    GMOE_InvED)
        STEPS=10000
        BATCH_SIZE=64
        EXTRA_HPARAMS=',"model":"deit_tiny_patch16_224","lambda_inv":0.01,"lambda_sp":0,"lambda_bal":0,"lambda_div":0.02,"alpha":4.0'
        ;;
    *)
        echo "[WARN] Unknown algo '$ALGO' — using ERM defaults. Edit script if needed."
        STEPS=10000
        BATCH_SIZE=32
        EXTRA_HPARAMS=''
        ;;
esac

CHECKPOINT_FREQ=500

LOG_DIR=multi_dataset/logs
OUT_ROOT=multi_dataset/iwildcam_paper_${DATASET}_${ALGO}
mkdir -p "$LOG_DIR" "$OUT_ROOT"

run_one() {
    local seed=$1
    local hparams="{\"data_augmentation\":true,\"batch_size\":${BATCH_SIZE}${EXTRA_HPARAMS}}"

    local out_dir="${OUT_ROOT}/seed${seed}"
    local log_file="$LOG_DIR/${DATASET}_${ALGO}_iwildcam_paper_seed${seed}.log"

    if [ -f "$out_dir/done" ]; then
        echo "[SKIP ] $ALGO seed$seed — already done"
        return 0
    fi

    mkdir -p "$out_dir"
    echo "[START] $DATASET/$ALGO seed$seed bs=$BATCH_SIZE steps=$STEPS"

    python -m domainbed.scripts.train \
        --dataset "$DATASET" --algorithm "$ALGO" \
        --test_envs $TEST_ENVS \
        --output_dir "$out_dir" \
        --hparams "$hparams" \
        --steps "$STEPS" --checkpoint_freq "$CHECKPOINT_FREQ" \
        --seed "$seed" --trial_seed "$seed" \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        touch "$out_dir/done"
        echo "[DONE ] $ALGO seed$seed"
    else
        echo "[FAIL ] $ALGO seed$seed — see $log_file"
    fi
}

export ALGO DATASET STEPS CHECKPOINT_FREQ EXTRA_HPARAMS BATCH_SIZE TEST_ENVS LOG_DIR OUT_ROOT
export -f run_one

echo "=== iWildCam paper-split: $DATASET/$ALGO ==="
echo "    dataset: $DATASET"
echo "    seeds: ${SEEDS[*]}"
echo "    parallel: $PARALLEL"
echo "    out_root: $OUT_ROOT"

if [ "$PARALLEL" -gt 1 ]; then
    printf '%s\n' "${SEEDS[@]}" | xargs -n1 -P"$PARALLEL" -I{} bash -c 'run_one "$@"' _ {}
else
    for s in "${SEEDS[@]}"; do run_one "$s"; done
fi

done_count=$(find "$OUT_ROOT" -name "done" 2>/dev/null | wc -l)
echo "=== Finished: $done_count / ${#SEEDS[@]} done ==="
