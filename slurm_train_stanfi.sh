#!/bin/bash
#SBATCH --job-name=pruned-gru-train-stanfi
#SBATCH --output=logs/pruned-gru-train-stanfi-%j.out
#SBATCH --error=logs/pruned-gru-train-stanfi-%j.err
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_DIR="/gpfs/workdir/balazsk/prunedAttentionGRU"
RESULTS_DIR="${REPO_DIR}/results/training_jobs_stanfi"

mkdir -p "${REPO_DIR}/logs"
mkdir -p "${RESULTS_DIR}"
cd "${REPO_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

python -u train_with_metrics.py \
    --datasets stanfi \
    --batchsize 16 \
    --learningrate 1e-3 \
    --epochs 100 \
    --hidden-size 128 \
    --attention-dim 32 \
    --seed 42 \
    --results-dir "${RESULTS_DIR}"
