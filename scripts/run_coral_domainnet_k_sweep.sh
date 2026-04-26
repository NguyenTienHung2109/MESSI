#!/bin/bash
# Run CORAL on DomainNet with K-sweep (varying number of source domains).
# Total training budget is FIXED at TOTAL_BUDGET samples — split equally across K sources.
#
# Test domain: env 5 (sketch) — fixed for all K.
# Source progression (added closest → farthest from sketch in visual style):
#   K=2: {clipart, quickdraw}
#   K=3: + painting
#   K=4: + real
#   K=5: + infograph
#
# Usage:
#   bash scripts/run_coral_domainnet_k_sweep.sh            # default 2 parallel
#   bash scripts/run_coral_domainnet_k_sweep.sh 1          # sequential

set -u

PARALLEL=${1:-2}
ALGO=CORAL
DATASET=DomainNet
STEPS=7500
CHECKPOINT_FREQ=500
TOTAL_BUDGET=50000
HPARAMS='{}'   # default: ResNet50, lr=5e-5, mmd_gamma=1.0

LOG_DIR=multi_dataset/logs
mkdir -p "$LOG_DIR"

# Map K → "test_envs" string. Test domain (sketch=5) always FIRST; the rest are
# sources excluded from training to enforce K source domains.
#   K=5: train = {clipart=0, painting=2, quickdraw=3, real=4}, exclude {infograph=1}
#   K=4: train = {clipart=0, painting=2, quickdraw=3},          exclude {1, real=4}
#   K=3: train = {clipart=0, quickdraw=3},                       exclude {1, 4, painting=2}
#   K=2: train = {clipart=0, quickdraw=3},                       (Wait — that's still K=2; for K=3 we need to add painting back)
# Re-derived correctly below: progression closest → farthest is
#   K=2 sources = {clipart=0, quickdraw=3}
#   K=3 sources = +painting=2  → {0,2,3}
#   K=4 sources = +real=4      → {0,2,3,4}
#   K=5 sources = +infograph=1 → {0,1,2,3,4}
# Excluded sources (not in K) are appended to test_envs after the test (sketch=5).
declare -A K_TO_TESTENVS=(
    [2]="5 1 4 2"     # exclude infograph(1), real(4), painting(2)
    [3]="5 1 4"       # exclude infograph(1), real(4)
    [4]="5 1"         # exclude infograph(1)
    [5]="5"           # all 5 remaining envs are sources
)

run_one() {
    local K=$1
    local seed=$2

    local test_envs="${K_TO_TESTENVS[$K]}"
    local max_samples=$((TOTAL_BUDGET / K))
    local out_dir="multi_dataset/test_CORAL_domainnet_k/K${K}_seed${seed}"
    local log_file="$LOG_DIR/${ALGO}_${DATASET}_K${K}_seed${seed}.log"

    if [ -f "$out_dir/done" ]; then
        echo "[SKIP ] K=$K seed$seed — already done"
        return 0
    fi

    mkdir -p "$out_dir"
    echo "[START] K=$K seed$seed (test_envs=[$test_envs], max_samples=$max_samples)"

    python -m domainbed.scripts.train \
        --dataset "$DATASET" --algorithm "$ALGO" --test_envs $test_envs \
        --max_samples_per_env "$max_samples" \
        --output_dir "$out_dir" \
        --hparams "$HPARAMS" \
        --steps "$STEPS" --checkpoint_freq "$CHECKPOINT_FREQ" \
        --seed "$seed" --trial_seed "$seed" \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        touch "$out_dir/done"
        echo "[DONE ] K=$K seed$seed"
    else
        echo "[FAIL ] K=$K seed$seed — see $log_file"
    fi
}

export ALGO DATASET STEPS CHECKPOINT_FREQ TOTAL_BUDGET HPARAMS LOG_DIR
export -f run_one
declare -p K_TO_TESTENVS > /tmp/k_to_testenvs_coral.sh
export _K_MAP_FILE=/tmp/k_to_testenvs_coral.sh

echo "=== Running $ALGO on $DATASET (K=2..5, 3 seeds = 12 runs, parallel=$PARALLEL) ==="
echo "    Test domain: env 5 (sketch)"
echo "    Total training budget: $TOTAL_BUDGET samples (split = budget/K per source)"
echo ""

{
    for seed in 0 1 2; do
        for K in 2 3 4 5; do
            echo "$K $seed"
        done
    done
} | xargs -n 2 -P "$PARALLEL" bash -c 'source $_K_MAP_FILE; run_one "$0" "$1"'

echo ""
echo "=== Summary ==="
done_count=$(find multi_dataset/test_CORAL_domainnet_k -name "done" 2>/dev/null | wc -l)
echo "Completed: $done_count / 12"
