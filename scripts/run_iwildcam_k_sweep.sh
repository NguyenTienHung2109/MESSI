#!/bin/bash
# Generic iWildCam K-sweep runner.
# Sweeps K ∈ {4, 8, 10, 12, 14, 16} sources × 3 seeds, with FIXED total budget=16k samples.
# Algorithm and hparams are parameterized so the same script works for CORAL, GMOE_InvMMD, etc.
#
# Usage:
#   bash scripts/run_iwildcam_k_sweep.sh CORAL                   # default 1 parallel
#   bash scripts/run_iwildcam_k_sweep.sh CORAL 2                 # 2 parallel jobs
#   bash scripts/run_iwildcam_k_sweep.sh GMOE_InvMMD             # InvMMD variant
#   bash scripts/run_iwildcam_k_sweep.sh GMOE_InvMMD 2
#
# Test domains: locations {93, 309, 117} — fixed for all K.
# Source pool: top 16 locations by sample count (nested progression).

set -u

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 ALGO [PARALLEL]"
    echo "  ALGO ∈ {CORAL, GMOE_InvMMD, GMOE_InvOT, GMOE_InvED, ERM, IRM, ...}"
    exit 1
fi

ALGO=$1
PARALLEL=${2:-1}

DATASET=WILDSIWildCam
CHECKPOINT_FREQ=500

# Per-algorithm hparams + per-algorithm steps.
# Effective batch per step = K × batch_size; must fit 16GB VRAM (RTX 5070 Ti).
# CORAL/ResNet50 is heavier than DeiT-tiny, so we drop bs at high K. The
# pick_batch_size helper below scales by K.
pick_batch_size() {
    local algo=$1
    local k=$2
    case "$algo" in
        CORAL|ERM|IRM|MMD|GroupDRO|VREx|DANN|Mixup)
            # ResNet50 ~25M params + activations. Empirically: bs=8 ok up to K=10,
            # bs=4 for K=12-14, bs=2 for K=16 on 16GB.
            if   [ "$k" -le 10 ]; then echo 8
            elif [ "$k" -le 14 ]; then echo 4
            else                       echo 2
            fi ;;
        GMOE_InvMMD|GMOE_InvOT|GMOE_InvED)
            # DeiT-tiny ~5M params, much lighter. bs=32 fine up to K=10, then taper.
            if   [ "$k" -le 10 ]; then echo 32
            elif [ "$k" -le 14 ]; then echo 16
            else                       echo 8
            fi ;;
        *)
            echo 8 ;;
    esac
}

case "$ALGO" in
    CORAL)
        STEPS=5000
        EXTRA_HPARAMS=''
        ;;
    GMOE_InvMMD)
        STEPS=10000
        EXTRA_HPARAMS=',"model":"deit_tiny_patch16_224","lambda_inv":0.01,"lambda_sp":0,"lambda_bal":0,"lambda_div":0.02,"alpha":4.0'
        ;;
    GMOE_InvOT)
        STEPS=7500
        EXTRA_HPARAMS=',"model":"deit_tiny_patch16_224","lambda_inv":0.01,"lambda_sp":0,"lambda_bal":0,"lambda_div":0.02,"alpha":4.0,"ot_epsilon":0.1,"sinkhorn_iters":50'
        ;;
    GMOE_InvED)
        STEPS=7500
        EXTRA_HPARAMS=',"model":"deit_tiny_patch16_224","lambda_inv":0.01,"lambda_sp":0,"lambda_bal":0,"lambda_div":0.02,"alpha":4.0'
        ;;
    ERM|IRM|MMD|GroupDRO|VREx|DANN|Mixup)
        STEPS=5000
        EXTRA_HPARAMS=''
        ;;
    *)
        echo "[WARN] Unknown algo '$ALGO' — using empty hparams. Edit script if you need custom hparams."
        STEPS=5000
        EXTRA_HPARAMS=''
        ;;
esac

LOG_DIR=multi_dataset/logs
OUT_ROOT=multi_dataset/test_${ALGO}_iwildcam_k
mkdir -p "$LOG_DIR"

K_VALUES=(4 8 10 12 14 16)
SEEDS=(0)   # only seed 0 — multi-seed runs OOM/unstable on 16GB GPU

run_one() {
    local K=$1
    local seed=$2

    local source_envs="$(python scripts/iwildcam_k_config.py --K $K --emit source_envs)"
    local test_envs="$(python scripts/iwildcam_k_config.py --K $K --emit test_envs)"
    local max_samples="$(python scripts/iwildcam_k_config.py --K $K --emit max_samples)"
    local bs="$(pick_batch_size "$ALGO" "$K")"
    local hparams="{\"batch_size\":${bs}${EXTRA_HPARAMS}}"

    local out_dir="${OUT_ROOT}/K${K}_seed${seed}"
    local log_file="$LOG_DIR/${ALGO}_iwildcam_K${K}_seed${seed}.log"

    if [ -f "$out_dir/done" ]; then
        echo "[SKIP ] $ALGO K=$K seed$seed — already done"
        return 0
    fi

    mkdir -p "$out_dir"
    echo "[START] $ALGO K=$K seed$seed bs=$bs (sources=[$source_envs], tests=[$test_envs], max_samples=$max_samples)"

    python -m domainbed.scripts.train \
        --dataset "$DATASET" --algorithm "$ALGO" \
        --test_envs $test_envs \
        --source_envs $source_envs \
        --max_samples_per_env "$max_samples" \
        --output_dir "$out_dir" \
        --hparams "$hparams" \
        --steps "$STEPS" --checkpoint_freq "$CHECKPOINT_FREQ" \
        --seed "$seed" --trial_seed "$seed" \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        touch "$out_dir/done"
        echo "[DONE ] $ALGO K=$K seed$seed"
    else
        echo "[FAIL ] $ALGO K=$K seed$seed — see $log_file"
    fi
}

export ALGO DATASET STEPS CHECKPOINT_FREQ EXTRA_HPARAMS LOG_DIR OUT_ROOT
export -f run_one pick_batch_size

echo "=== iWildCam K-sweep: $ALGO ==="
echo "    K values: ${K_VALUES[*]}"
echo "    seeds: ${SEEDS[*]}"
echo "    total runs: $((${#K_VALUES[@]} * ${#SEEDS[@]}))"
echo "    parallel: $PARALLEL"
echo "    out_root: $OUT_ROOT"
echo ""

{
    for seed in "${SEEDS[@]}"; do
        for K in "${K_VALUES[@]}"; do
            echo "$K $seed"
        done
    done
} | xargs -n 2 -P "$PARALLEL" bash -c 'run_one "$0" "$1"'

echo ""
echo "=== Summary ==="
done_count=$(find "$OUT_ROOT" -name "done" 2>/dev/null | wc -l)
total=$((${#K_VALUES[@]} * ${#SEEDS[@]}))
echo "Completed: $done_count / $total"
