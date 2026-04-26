#!/bin/bash
# Run GMOE_InvOT on OfficeHome: 4 envs × 3 seeds = 12 runs, parallel on single GPU.
#
# Usage:
#   bash scripts/run_invot_officehome.sh            # default 4 parallel jobs
#   bash scripts/run_invot_officehome.sh 3          # custom parallelism

set -u

PARALLEL=${1:-4}
ALGO=GMOE_InvOT
DATASET=OfficeHome
STEPS=5000
CHECKPOINT_FREQ=500
HPARAMS='{"model":"deit_tiny_patch16_224","lambda_inv":0.01,"lambda_sp":0,"lambda_bal":0,"lambda_div":0.02,"alpha":4.0,"ot_epsilon":0.1,"sinkhorn_iters":50}'

LOG_DIR=multi_dataset/logs
mkdir -p "$LOG_DIR"

run_one() {
    local env=$1
    local seed=$2

    local out_dir="multi_dataset/test_${ALGO#GMOE_}/${DATASET}_env${env}_seed${seed}"
    local log_file="$LOG_DIR/${ALGO}_${DATASET}_env${env}_seed${seed}.log"

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

echo "=== Running $ALGO on $DATASET (4 envs × 3 seeds = 12 runs, parallel=$PARALLEL) ==="

{
    for seed in 0 1 2; do
        for env in 0 1 2 3; do
            echo "$env $seed"
        done
    done
} | xargs -n 2 -P "$PARALLEL" bash -c 'run_one "$0" "$1"'

echo ""
echo "=== Summary ==="
done_count=$(find multi_dataset/test_${ALGO#GMOE_} -name "done" -path "*${DATASET}_*" 2>/dev/null | wc -l)
echo "Completed: $done_count / 12"
