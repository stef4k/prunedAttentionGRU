#!/bin/bash
#SBATCH --job-name=pruned-gru-har1-smoke
#SBATCH --output=logs/pruned-gru-har1-smoke-%j.out
#SBATCH --error=logs/pruned-gru-har1-smoke-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_DIR="/gpfs/workdir/balazsk/prunedAttentionGRU"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

python -u smoke_test.py \
    --datasets har-1 \
    --batchsize 4 \
    --learningrate 1e-3 \
    --max-train-samples 8 \
    --max-test-samples 4
