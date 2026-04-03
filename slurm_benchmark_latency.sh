#!/bin/bash
#SBATCH --job-name=pruned-gru-benchmark
#SBATCH --output=logs/pruned-gru-benchmark-%j.out
#SBATCH --error=logs/pruned-gru-benchmark-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_DIR="/gpfs/workdir/balazsk/prunedAttentionGRU"
RESULTS_DIR="${REPO_DIR}/results/benchmark_latency"

mkdir -p "${REPO_DIR}/logs"
mkdir -p "${RESULTS_DIR}"
cd "${REPO_DIR}"

echo "Results will be saved to: ${RESULTS_DIR}"

# ── verify the expected pruned checkpoints exist before starting ──────────────
CKPT_ARIL="${REPO_DIR}/results/training_jobs/20260327T142902Z_aril/model.pt"
CKPT_HAR1="${REPO_DIR}/results/training_jobs/20260327T154618Z_har-1/model.pt"
CKPT_HAR3="${REPO_DIR}/results/training_jobs/20260327T210654Z_har-3/model.pt"
CKPT_SIGNFI="${REPO_DIR}/results/training_jobs/20260328T000619Z_signfi/model.pt"
CKPT_STANFI="${REPO_DIR}/results/training_jobs_stanfi/20260331T102238Z_stanfi/model.pt"

echo "Verifying checkpoints..."
for ckpt in "$CKPT_ARIL" "$CKPT_HAR1" "$CKPT_HAR3" "$CKPT_SIGNFI" "$CKPT_STANFI"; do
    if [[ ! -f "$ckpt" ]]; then
        echo "ERROR: checkpoint not found: $ckpt"
        exit 1
    fi
    echo "  OK: $ckpt"
done

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

python -u benchmark_single_sample.py
