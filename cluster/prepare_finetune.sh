#!/bin/bash
# Runs on a login node. Installs ms-swift into its own .venv-finetune from
# local wheels (no internet needed), verifies the base HF model is already
# in $HF_CACHE (already there from prior eval runs of qwen3-vl-2b), then
# sbatch-submits a LoRA training job.
#
# Mirrors cluster/prepare.sh, but kept separate since training isn't
# YAML-driven like the eval experiment matrix.
#
# Usage: prepare_finetune.sh [sbatch-script] [time-limit] [job-name]
#   Defaults to the full pilot run; pass the overfit sbatch + a short time
#   limit for the quick sanity check.

set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh

SBATCH_SCRIPT="${1:-cluster/finetune_swap_detect.sbatch}"
TIME_LIMIT="${2:-04:00:00}"
JOB_NAME="${3:-$(basename "$SBATCH_SCRIPT" .sbatch)}"

module purge
module load oneapi hdf5 python

if [ ! -d .venv-finetune ]; then
  echo "Creating .venv-finetune..."
  python -m venv .venv-finetune
fi
source .venv-finetune/bin/activate
unset PYTHONHOME PYTHONPATH
pip install --no-index --find-links=cluster/wheels-finetune/ -r cluster/requirements-finetune.txt

MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
HF_MODEL_DIR="models--$(echo "$MODEL_PATH" | sed 's|/|--|g')"
if [ ! -d "$HF_CACHE/hub/$HF_MODEL_DIR" ]; then
  echo "ERROR: model '$MODEL_PATH' not found at $HF_CACHE/hub/$HF_MODEL_DIR" >&2
  echo "Run ./run.sh against any qwen3-vl-2b experiment first — it ships the model via rsync." >&2
  exit 1
fi

if [ ! -f datasets/CLEVRER/sft_swap_detect_train.jsonl ]; then
  echo "ERROR: datasets/CLEVRER/sft_swap_detect_train.jsonl not found." >&2
  echo "Re-run ./finetune.sh from your laptop — it ships the training data via rsync." >&2
  exit 1
fi

mkdir -p logs
echo "Submitting $SBATCH_SCRIPT → account=$ACCOUNT qos=acc_ehpc time=$TIME_LIMIT gpus=1"
sbatch \
  --account="$ACCOUNT" \
  --qos=acc_ehpc \
  --time="$TIME_LIMIT" \
  --gres=gpu:1 \
  --cpus-per-task=20 \
  --ntasks=1 \
  --nodes=1 \
  --job-name="$JOB_NAME" \
  --output="logs/%x-%j.out" \
  --error="logs/%x-%j.err" \
  --export=ALL,REMOTE_DIR="$REMOTE_DIR",HF_CACHE="$HF_CACHE" \
  "$SBATCH_SCRIPT"
