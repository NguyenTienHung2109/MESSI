#!/bin/bash
# MetaShift_K sweep: 5-class (cat,dog,horse,elephant,bird), N=240 per class,
# K ∈ {4, 6, 8}, single test context, 3 seeds.
#
# Splits must already exist under data/metashift/splits/cat_dog_horse_elephant_bird/
# (run scripts/metashift_build_splits.py if not).
#
# Images must already exist under data/metashift/raw/images/<id>.jpg
# (run python -m domainbed.scripts.download --data_dir data if not).
#
# Usage:
#   bash scripts/run_metashift_k_sweep.sh ERM
#   bash scripts/run_metashift_k_sweep.sh GMOE_InvMMD 2

set -u

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 ALGO [PARALLEL]"
    echo "  ALGO ∈ {ERM, CORAL, IRM, Mixup, GroupDRO, GMOE_InvMMD, ...}"
    exit 1
fi

ALGO=$1
PARALLEL=${2:-1}

DATASET=MetaShift_K
CLASS_SET=cat_dog_horse_elephant_bird
TOTAL=240
DATA_DIR=data
CHECKPOINT_FREQ=200

case "$ALGO" in
    GMOE_InvMMD)
        STEPS=5000
        BASE_HP='{"model":"deit_tiny_patch16_224","batch_size":32,"lambda_inv":0.01,"lambda_sp":0,"lambda_bal":0,"lambda_div":0.02,"alpha":4.0}'
        ;;
    GMOE_InvOT)
        STEPS=5000
        BASE_HP='{"model":"deit_tiny_patch16_224","batch_size":16,"lambda_inv":0.01,"lambda_sp":0,"lambda_bal":0,"lambda_div":0.02,"alpha":4.0,"ot_epsilon":0.1,"sinkhorn_iters":50}'
        ;;
    ERM|IRM|MMD|CORAL|GroupDRO|VREx|DANN|Mixup)
        STEPS=5000
        BASE_HP='{"model":"deit_tiny_patch16_224","batch_size":32}'
        ;;
    *)
        echo "[WARN] Unknown algo '$ALGO' — using deit_tiny + batch_size=32."
        STEPS=5000
        BASE_HP='{"model":"deit_tiny_patch16_224","batch_size":32}'
        ;;
esac

LOG_DIR=multi_dataset/logs
OUT_ROOT=multi_dataset/test_${ALGO}_metashift_k
mkdir -p "$LOG_DIR"

K_VALUES=(4 6 8)
SEEDS=(0 1 2)

run_one() {
    local K=$1
    local seed=$2

    # MetaShift_K exposes K training envs at indices 0..K-1, then 1 test env at index K (single_test).
    local test_envs=$K

    # Build the per-run hparams: BASE_HP merged with split selectors.
    local hp=$(python -c "
import json, sys
base=json.loads('''$BASE_HP''')
base.update({'class_set':'$CLASS_SET','K':$K,'split_seed':$seed,'total_per_class':$TOTAL,'single_test':True})
print(json.dumps(base))")

    local out_dir="${OUT_ROOT}/K${K}_seed${seed}"
    local log_file="$LOG_DIR/${ALGO}_metashift_K${K}_seed${seed}.log"

    if [ -f "$out_dir/done" ]; then
        echo "[SKIP ] $ALGO K=$K seed$seed — already done"
        return 0
    fi

    mkdir -p "$out_dir"
    echo "[START] $ALGO K=$K seed$seed (test_env=$test_envs)"

    python -m domainbed.scripts.train \
        --dataset "$DATASET" --algorithm "$ALGO" \
        --data_dir "$DATA_DIR" \
        --test_envs $test_envs \
        --output_dir "$out_dir" \
        --hparams "$hp" \
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

export ALGO DATASET CLASS_SET TOTAL DATA_DIR STEPS CHECKPOINT_FREQ BASE_HP LOG_DIR OUT_ROOT
export -f run_one

echo "=== MetaShift_K sweep: $ALGO ==="
echo "    class_set: $CLASS_SET (5 classes)"
echo "    K values:  ${K_VALUES[*]}"
echo "    seeds:     ${SEEDS[*]}"
echo "    total runs: $((${#K_VALUES[@]} * ${#SEEDS[@]}))"
echo "    parallel:  $PARALLEL"
echo "    out_root:  $OUT_ROOT"
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
