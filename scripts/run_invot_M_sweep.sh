#!/bin/bash
# Sweep số experts M ∈ {2, 4, 6, 8, 16} cho GMOE_InvOT (deit_small).
# Local machine, không SLURM.
#
# Layout output:
#   multi_dataset/test_InvOT_small_M_sweep/M<M>/<DATASET>_env<E>_seed<S>/
#
# Parallel strategy:
#   - M ∈ {2,4,6,8}: 2 parallel (mỗi run ~5-7 GB, 2 fit 16 GB)
#   - M = 16: sequential (12 GB / run, không parallel được)
#
# Usage:
#   bash scripts/run_invot_M_sweep.sh                # default 2-parallel for small M
#   bash scripts/run_invot_M_sweep.sh 3              # custom parallelism (cẩn thận OOM)
#   bash scripts/run_invot_M_sweep.sh 1              # all sequential
#
# Tip: chạy trong tmux/screen để không sợ SSH disconnect.

set -u

PARALLEL_SMALL=${1:-2}    # parallelism cho M ≤ 8
PARALLEL_M16=1            # M=16 luôn sequential

ALGO=GMOE_InvOT
DATASETS=( "PACS" "OfficeHome" )
SEEDS=( 0 1 2 )
ENVS=( 0 1 2 3 )
STEPS=5000
CHECKPOINT_FREQ=500
SWEEP_ROOT="multi_dataset/test_InvOT_small_M_sweep"
LOG_DIR="multi_dataset/logs"
mkdir -p "$LOG_DIR"

# Shared hparams (lambdas, alpha, model)
HP_BASE='"lambda_inv":0.01,"lambda_sp":0.02,"lambda_bal":0.02,"lambda_div":0.02,"alpha":4.0,"model":"deit_small_patch16_224"'

# ===========================================================================
# Worker: chạy 1 (M, dataset, seed, env)
# ===========================================================================
run_one() {
    local M=$1
    local DATASET=$2
    local SEED=$3
    local ENV=$4

    local OUT_DIR="${SWEEP_ROOT}/M${M}/${DATASET}_env${ENV}_seed${SEED}"
    local LOG_FILE="${LOG_DIR}/invot_M${M}_${DATASET}_env${ENV}_seed${SEED}.log"

    if [ -f "$OUT_DIR/done" ]; then
        echo "[SKIP ] M=$M $DATASET env=$ENV seed=$SEED — already done"
        return 0
    fi
    mkdir -p "$OUT_DIR"
    echo "[START] M=$M $DATASET env=$ENV seed=$SEED → $LOG_FILE"

    WANDB_MODE=disabled python -m domainbed.scripts.train \
        --dataset "$DATASET" --algorithm "$ALGO" --test_envs "$ENV" \
        --output_dir "$OUT_DIR" \
        --hparams "{${HP_BASE},\"num_experts\":${M}}" \
        --steps "$STEPS" --checkpoint_freq "$CHECKPOINT_FREQ" \
        --seed "$SEED" --trial_seed "$SEED" \
        > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        touch "$OUT_DIR/done"
        echo "[DONE ] M=$M $DATASET env=$ENV seed=$SEED"
    else
        echo "[FAIL ] M=$M $DATASET env=$ENV seed=$SEED — see $LOG_FILE"
    fi
}

export ALGO STEPS CHECKPOINT_FREQ SWEEP_ROOT LOG_DIR HP_BASE
export -f run_one

# ===========================================================================
# Loop: cho mỗi M, generate task list (DATASET × SEED × ENV) rồi xargs -P
# ===========================================================================
M_VALUES=(2 4 6 8 16)
TOTAL_START_TIME=$(date +%s)

for M in "${M_VALUES[@]}"; do
    if [ "$M" -eq 16 ]; then
        P=$PARALLEL_M16
    else
        P=$PARALLEL_SMALL
    fi

    echo
    echo "=== M=$M ($P parallel, $(date)) ==="

    # Generate "M DATASET SEED ENV" lines and dispatch via xargs -P.
    {
        for DATASET in "${DATASETS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                for ENV in "${ENVS[@]}"; do
                    echo "$M $DATASET $SEED $ENV"
                done
            done
        done
    } | xargs -n 4 -P "$P" bash -c 'run_one "$0" "$1" "$2" "$3"'

    DONE_COUNT=$(find "${SWEEP_ROOT}/M${M}" -name "done" 2>/dev/null | wc -l)
    echo "=== M=$M finished: $DONE_COUNT / 24 done ==="
done

# ===========================================================================
# Summary
# ===========================================================================
TOTAL_END_TIME=$(date +%s)
ELAPSED=$((TOTAL_END_TIME - TOTAL_START_TIME))
echo
echo "================================================================"
echo "=== Sweep summary ==="
TOTAL_DONE=0
for M in "${M_VALUES[@]}"; do
    DONE_COUNT=$(find "${SWEEP_ROOT}/M${M}" -name "done" 2>/dev/null | wc -l)
    echo "  M=$M: $DONE_COUNT / 24 done"
    TOTAL_DONE=$((TOTAL_DONE + DONE_COUNT))
done
echo "  -----"
echo "  TOTAL: $TOTAL_DONE / 120 done"
echo "  Wall time: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
echo "================================================================"
