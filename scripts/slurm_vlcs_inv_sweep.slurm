#!/bin/bash
#SBATCH --job-name=vlcs_inv_sweep
#SBATCH --output=/mnt/data/s26hungnt/DG-OMOE/slurm_logs/vlcs_inv_%A_%a.out
#SBATCH --error=/mnt/data/s26hungnt/DG-OMOE/slurm_logs/vlcs_inv_%A_%a.err

#SBATCH --partition=main
#SBATCH --time=150:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

echo "Sequential Job started at: $(date)"
echo "Running on node: $(hostname)"

source /mnt/data/s26hungnt/miniconda3/etc/profile.d/conda.sh
conda activate dg

# ===== YOUR CODE =====
cd /mnt/data/s26hungnt/DG-OMOE
export PYTHONPATH=$PYTHONPATH:.
export WANDB_MODE=disabled

# Tạo thư mục log slurm nếu chưa có (cho lần submit tiếp theo)
mkdir -p slurm_logs multi_dataset/logs

# ===== Định nghĩa 3 cấu hình bằng parallel arrays =====
# Index i tương ứng cấu hình thứ i.
TAGS=(         "invmmd_small"            "invot_small"             "invot_tiny"            )
OUT_PREFIXES=( "test_InvMMD_small"       "test_InvOT_small"        "test_InvOT_tiny"       )
ALGOS=(        "GMOE_InvMMD"             "GMOE_InvOT"              "GMOE_InvOT"            )
MODELS=(       "deit_small_patch16_224"  "deit_small_patch16_224"  "deit_tiny_patch16_224" )

SEEDS=(0 1 2)
ENVS=(0 1 2 3)
DATASET="VLCS"
STEPS=5000
CHECKPOINT_FREQ=500

# Shared MESSI-style hparams.
HPARAMS_BASE='"lambda_inv":0.01,"lambda_sp":0.02,"lambda_bal":0.02,"lambda_div":0.02,"alpha":4.0'

# ===== Chạy tuần tự tất cả cấu hình =====
for i in "${!TAGS[@]}"; do
    TAG="${TAGS[$i]}"
    OUT_PREFIX="${OUT_PREFIXES[$i]}"
    ALGO="${ALGOS[$i]}"
    MODEL="${MODELS[$i]}"

    for SEED in "${SEEDS[@]}"; do
        for ENV in "${ENVS[@]}"; do
            OUT_DIR="multi_dataset/${OUT_PREFIX}/${DATASET}_env${ENV}_seed${SEED}"
            LOG_FILE="multi_dataset/logs/${TAG}_${DATASET}_env${ENV}_seed${SEED}.log"

            if [ -f "$OUT_DIR/done" ]; then
                echo "[SKIP ] $TAG env=$ENV seed=$SEED — already done"
                continue
            fi
            mkdir -p "$OUT_DIR"

            echo "----------------------------------------------------------------"
            echo "RUNNING: tag=$TAG | algo=$ALGO | model=$MODEL | env=$ENV | seed=$SEED | $(date)"
            echo "         out_dir=$OUT_DIR"
            echo "         log=$LOG_FILE"
            echo "----------------------------------------------------------------"

            python -m domainbed.scripts.train \
                --dataset "$DATASET" \
                --algorithm "$ALGO" \
                --test_envs "$ENV" \
                --output_dir "$OUT_DIR" \
                --hparams "{\"model\":\"$MODEL\",${HPARAMS_BASE}}" \
                --steps "$STEPS" \
                --checkpoint_freq "$CHECKPOINT_FREQ" \
                --seed "$SEED" \
                --trial_seed "$SEED" \
                > "$LOG_FILE" 2>&1

            if [ $? -eq 0 ]; then
                touch "$OUT_DIR/done"
                echo "Finished: $TAG env=$ENV seed=$SEED | $(date)"
            else
                echo "FAILED:   $TAG env=$ENV seed=$SEED — see $LOG_FILE"
            fi
        done
    done
done

echo
echo "=== Summary ==="
for i in "${!TAGS[@]}"; do
    TAG="${TAGS[$i]}"
    OUT_PREFIX="${OUT_PREFIXES[$i]}"
    DONE_COUNT=$(find "multi_dataset/${OUT_PREFIX}" -name "done" -path "*${DATASET}_*" 2>/dev/null | wc -l)
    echo "  $TAG: $DONE_COUNT / 12 done"
done

echo
echo "All sequential experiments finished at: $(date)"
