#!/bin/bash
# Lightweight sbatch wrapper for bulk submissions. Assumes the venv is already
# prepared (run cluster/prepare.sh once before using this in a loop).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh
export PATH="$(pwd)/cluster/bin:$PATH"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <config/experiments/foo.yaml>" >&2
  exit 1
fi
EXP="$1"

SCRIPT=$(yq -r '.script' "$EXP")
QOS=$(yq -r '.slurm.qos' "$EXP")
TIME=$(yq -r '.slurm.time' "$EXP")
GPUS=$(yq -r '.slurm.gpus' "$EXP")
CPUS_PER_GPU=$(yq -r '.slurm.cpus_per_gpu // 20' "$EXP")
CPUS=$((GPUS * CPUS_PER_GPU))

mkdir -p logs
JOB_NAME=$(basename "$EXP" .yaml)
sbatch --parsable \
  --account="$ACCOUNT" --qos="$QOS" --time="$TIME" \
  --gres="gpu:$GPUS" --cpus-per-task="$CPUS" --ntasks=1 --nodes=1 \
  --job-name="$JOB_NAME" \
  --output="logs/%x-%j.out" --error="logs/%x-%j.err" \
  --export=ALL,EXP="$EXP",SCRIPT="$SCRIPT",REMOTE_DIR="$REMOTE_DIR",HF_CACHE="$HF_CACHE" \
  cluster/run.sbatch
