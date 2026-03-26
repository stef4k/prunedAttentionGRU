#!/bin/bash
#SBATCH --job-name=pruned-gru-train-all
#SBATCH --output=logs/pruned-gru-train-all-%j.out
#SBATCH --error=logs/pruned-gru-train-all-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_DIR="/gpfs/workdir/balazsk/prunedAttentionGRU"

mkdir -p "${REPO_DIR}/logs"
mkdir -p "${REPO_DIR}/results/training_jobs"
cd "${REPO_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

python -u train_with_metrics.py \
    --datasets aril har-1 har-3 signfi stanfi \
    --batchsize 128 \
    --learningrate 1e-3 \
    --epochs 100 \
    --hidden-size 128 \
    --attention-dim 32 \
    --seed 42 \
    --results-dir results/training_jobs
