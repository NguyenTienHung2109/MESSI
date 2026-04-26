#!/bin/bash
# Sweep ColoredMNIST_K over K=2..10 source domains × ALGOS × seeds.
#
# Usage:
#   bash scripts/run_coloredmnist_k_sweep.sh                # default 4 parallel
#   bash scripts/run_coloredmnist_k_sweep.sh 2              # parallelism=2
#   WANDB_MODE=disabled bash scripts/run_coloredmnist_k_sweep.sh
#
# Each run: K source envs (60k samples split equally) + 1 fixed test env
# (10k, p=0.5). Test env index = K → --test_envs $K.

set -u

PARALLEL=${1:-4}
ALGOS=(ERM CORAL GMOE_InvMMD)
KS=(2 3 4 5 6 7 8 9 10)
SEEDS=(0 1 2)
STEPS=5001
CHECKPOINT_FREQ=200

LOG_DIR=multi_dataset/coloredmnist_k/logs
mkdir -p "$LOG_DIR"

run_one() {
    local algo=$1
    local k=$2
    local seed=$3

    local out_dir="multi_dataset/coloredmnist_k/${algo}/K${k}_seed${seed}"
    local log_file="$LOG_DIR/${algo}_K${k}_seed${seed}.log"

    if [ -f "$out_dir/done" ]; then
        echo "[SKIP ] $algo K=$k seed=$seed — already done"
        return 0
    fi

    mkdir -p "$out_dir"
    echo "[START] $algo K=$k seed=$seed → $log_file"

    python -m domainbed.scripts.train \
        --dataset ColoredMNIST_K --algorithm "$algo" \
        --hparams "{\"num_source_domains\": $k}" \
        --test_envs "$k" \
        --output_dir "$out_dir" \
        --steps "$STEPS" --checkpoint_freq "$CHECKPOINT_FREQ" \
        --seed "$seed" --trial_seed "$seed" \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        touch "$out_dir/done"
        echo "[DONE ] $algo K=$k seed=$seed"
    else
        echo "[FAIL ] $algo K=$k seed=$seed — see $log_file"
    fi
}

export STEPS CHECKPOINT_FREQ LOG_DIR
export -f run_one

total=$(( ${#ALGOS[@]} * ${#KS[@]} * ${#SEEDS[@]} ))
echo "=== ColoredMNIST_K sweep: ${#ALGOS[@]} algos × ${#KS[@]} K × ${#SEEDS[@]} seeds = $total runs (parallel=$PARALLEL) ==="

{
    for algo in "${ALGOS[@]}"; do
        for k in "${KS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                echo "$algo $k $seed"
            done
        done
    done
} | xargs -n 3 -P "$PARALLEL" bash -c 'run_one "$0" "$1" "$2"'

echo ""
echo "=== Summary ==="
done_count=$(find multi_dataset/coloredmnist_k -name "done" 2>/dev/null | wc -l)
echo "Completed: $done_count / $total"
