#!/bin/bash
#SBATCH --job-name=invot_sweep_sensitivity
#SBATCH --output=/mnt/data/s26hungnt/DG-OMOE/slurm_logs/invot_sweep_%A_%a.out
#SBATCH --error=/mnt/data/s26hungnt/DG-OMOE/slurm_logs/invot_sweep_%A_%a.err

#SBATCH --partition=main
#SBATCH --time=150:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

echo "Sensitivity sweep started at: $(date)"
echo "Running on node: $(hostname)"

source /mnt/data/s26hungnt/miniconda3/etc/profile.d/conda.sh
conda activate dg

# ===== YOUR CODE =====
cd /mnt/data/s26hungnt/DG-OMOE
export PYTHONPATH=$PYTHONPATH:.
export WANDB_MODE=disabled

mkdir -p slurm_logs multi_dataset/logs

# ===========================================================================
# Sensitivity sweep for GMOE_InvOT (deit_small):
#   one-at-a-time over lambda_sp / lambda_bal / lambda_div
#   default cho 2 weight kia: 0.02
#   lambda_inv = 0.01, alpha = 4.0
#   row value=0.02 chung cho cả 3 sub-sweep ⇒ 1 baseline + 12 = 13 unique configs
# 13 configs × 2 datasets × 4 envs × 3 seeds = 312 runs
# ===========================================================================

# Parallel arrays: 13 unique configs. Index i ⇔ một hparam combination.
CONFIG_NAMES=(
    "default"
    "lambda_sp_0.000"  "lambda_sp_0.005"  "lambda_sp_0.010"  "lambda_sp_0.050"
    "lambda_bal_0.000" "lambda_bal_0.005" "lambda_bal_0.010" "lambda_bal_0.050"
    "lambda_div_0.000" "lambda_div_0.005" "lambda_div_0.010" "lambda_div_0.050"
)
SP_VALS=(
    0.02
    0.000  0.005  0.010  0.050
    0.02   0.02   0.02   0.02
    0.02   0.02   0.02   0.02
)
BAL_VALS=(
    0.02
    0.02   0.02   0.02   0.02
    0.000  0.005  0.010  0.050
    0.02   0.02   0.02   0.02
)
DIV_VALS=(
    0.02
    0.02   0.02   0.02   0.02
    0.02   0.02   0.02   0.02
    0.000  0.005  0.010  0.050
)

DATASETS=( "PACS" "OfficeHome" )
SEEDS=( 0 1 2 )
ENVS=( 0 1 2 3 )

ALGO="GMOE_InvOT"
MODEL="deit_small_patch16_224"
LAMBDA_INV=0.01
ALPHA=4.0
STEPS=5000
CHECKPOINT_FREQ=500
SWEEP_ROOT="multi_dataset/test_InvOT_small_sweep"

# ===========================================================================
# Sequential loop over (config, dataset, seed, env)
# ===========================================================================
for i in "${!CONFIG_NAMES[@]}"; do
    NAME="${CONFIG_NAMES[$i]}"
    SP="${SP_VALS[$i]}"
    BAL="${BAL_VALS[$i]}"
    DIV="${DIV_VALS[$i]}"

    HPARAMS="{\"model\":\"$MODEL\",\"lambda_inv\":$LAMBDA_INV,\"lambda_sp\":$SP,\"lambda_bal\":$BAL,\"lambda_div\":$DIV,\"alpha\":$ALPHA}"

    for DATASET in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for ENV in "${ENVS[@]}"; do
                OUT_DIR="${SWEEP_ROOT}/${NAME}/${DATASET}_env${ENV}_seed${SEED}"
                LOG_FILE="multi_dataset/logs/invot_sweep_${NAME}_${DATASET}_env${ENV}_seed${SEED}.log"

                if [ -f "$OUT_DIR/done" ]; then
                    echo "[SKIP ] $NAME | $DATASET env=$ENV seed=$SEED — already done"
                    continue
                fi
                mkdir -p "$OUT_DIR"

                echo "----------------------------------------------------------------"
                echo "RUNNING: config=$NAME"
                echo "         hparams=lambda_sp=$SP lambda_bal=$BAL lambda_div=$DIV (lambda_inv=$LAMBDA_INV alpha=$ALPHA)"
                echo "         dataset=$DATASET env=$ENV seed=$SEED | $(date)"
                echo "         out_dir=$OUT_DIR"
                echo "         log=$LOG_FILE"
                echo "----------------------------------------------------------------"

                python -m domainbed.scripts.train \
                    --dataset "$DATASET" \
                    --algorithm "$ALGO" \
                    --test_envs "$ENV" \
                    --output_dir "$OUT_DIR" \
                    --hparams "$HPARAMS" \
                    --steps "$STEPS" \
                    --checkpoint_freq "$CHECKPOINT_FREQ" \
                    --seed "$SEED" \
                    --trial_seed "$SEED" \
                    > "$LOG_FILE" 2>&1

                if [ $? -eq 0 ]; then
                    touch "$OUT_DIR/done"
                    echo "Finished: $NAME | $DATASET env=$ENV seed=$SEED | $(date)"
                else
                    echo "FAILED:   $NAME | $DATASET env=$ENV seed=$SEED — see $LOG_FILE"
                fi
            done
        done
    done
done

# ===========================================================================
# Summary
# ===========================================================================
echo
echo "=== Summary ==="
TOTAL_DONE=0
for NAME in "${CONFIG_NAMES[@]}"; do
    DONE_COUNT=$(find "${SWEEP_ROOT}/${NAME}" -name "done" 2>/dev/null | wc -l)
    echo "  $NAME: $DONE_COUNT / 24 done"
    TOTAL_DONE=$((TOTAL_DONE + DONE_COUNT))
done
echo "  -------"
echo "  TOTAL:  $TOTAL_DONE / 312 done"

echo
echo "Sensitivity sweep finished at: $(date)"
