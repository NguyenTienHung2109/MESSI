#!/bin/bash
# Run GMOE baseline with DeiT-Tiny backbone on DomainNet: 6 envs × 1 seed, sequential.
#
# Usage:
#   bash scripts/run_gmoe_tiny_domainnet.sh            # sequential through 6 envs
#   bash scripts/run_gmoe_tiny_domainnet.sh 2          # 2 parallel (DeiT-Ti small enough)
#
# DomainNet is the heaviest dataset (15k steps, 345 classes) — default sequential
# to avoid CPU/data-loader thrashing. DeiT-Ti uses ~3-4GB/run → could parallel 2-3.

set -u

PARALLEL=${1:-1}
ALGO=GMOE
DATASET=DomainNet
STEPS=15000
CHECKPOINT_FREQ=500
HPARAMS='{"model":"deit_tiny_patch16_224","num_experts":6,"gate_k":2,"mlp_ratio":4,"expert_depth":2}'

LOG_DIR=multi_dataset/logs
mkdir -p "$LOG_DIR"

run_one() {
    local env=$1
    local seed=$2

    local out_dir="multi_dataset/test_GMOE_tiny/${DATASET}_env${env}_seed${seed}"
    local log_file="$LOG_DIR/${ALGO}_tiny_${DATASET}_env${env}_seed${seed}.log"

    if [ -f "$out_dir/done" ]; then
        echo "[SKIP ] env$env seed$seed — already done"
        return 0
    fi

    mkdir -p "$out_dir"
    echo "[START] env$env seed$seed → $log_file"

    python -m domainbed.scripts.train \
        --dataset "$DATASET" --algorithm "$ALGO" --test_envs "$env" \
        --output_dir "$out_dir" \
        --hparams "$HPARAMS" \
        --steps "$STEPS" --checkpoint_freq "$CHECKPOINT_FREQ" \
        --seed "$seed" --trial_seed "$seed" \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        touch "$out_dir/done"
        echo "[DONE ] env$env seed$seed"
    else
        echo "[FAIL ] env$env seed$seed — see $log_file"
    fi
}

export ALGO DATASET STEPS CHECKPOINT_FREQ HPARAMS LOG_DIR
export -f run_one

echo "=== Running $ALGO (DeiT-Ti) on $DATASET (6 envs × 1 seed = 6 runs, parallel=$PARALLEL) ==="

{
    for seed in 0; do
        for env in 0 1 2 3 4 5; do
            echo "$env $seed"
        done
    done
} | xargs -n 2 -P "$PARALLEL" bash -c 'run_one "$0" "$1"'

echo ""
echo "=== Summary ==="
done_count=$(find multi_dataset/test_GMOE_tiny -name "done" -path "*${DATASET}_*" 2>/dev/null | wc -l)
echo "Completed: $done_count / 6"
