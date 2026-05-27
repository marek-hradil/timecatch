#!/bin/bash
# Runs on a login node. Installs deps from local wheels (no internet needed),
# verifies the HF model is already in $HF_CACHE, then sbatch-submits run.sbatch.

set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh

if [ $# -lt 1 ]; then
  echo "Usage: $0 <config/experiments/foo.yaml>" >&2
  exit 1
fi
EXP="$1"

# yq is shipped by run.sh in cluster/bin/yq — no internet needed.
export PATH="$(pwd)/cluster/bin:$PATH"

SCRIPT=$(yq -r '.script' "$EXP")
MODEL_KEY=$(yq -r '.model' "$EXP")
MODEL_PATH=$(yq -r '.path' "config/models/$MODEL_KEY.yaml")
QOS=$(yq -r '.slurm.qos' "$EXP")
TIME=$(yq -r '.slurm.time' "$EXP")
GPUS=$(yq -r '.slurm.gpus' "$EXP")
CPUS_PER_GPU=$(yq -r '.slurm.cpus_per_gpu // 20' "$EXP")
CPUS=$((GPUS * CPUS_PER_GPU))

module purge
module load oneapi hdf5 python

if [ ! -d .venv ]; then
  echo "Creating .venv..."
  python -m venv .venv
fi
source .venv/bin/activate
unset PYTHONHOME PYTHONPATH
# Idempotent — no-op when everything is already installed, recovers from a partial install otherwise.
pip install --no-index --find-links=cluster/wheels/ -r cluster/requirements.txt

# Sanity-check the model is in HF_CACHE (rsync'd by run.sh on laptop).
HF_MODEL_DIR="models--$(echo "$MODEL_PATH" | sed 's|/|--|g')"
if [ ! -d "$HF_CACHE/hub/$HF_MODEL_DIR" ]; then
  echo "ERROR: model '$MODEL_PATH' not found at $HF_CACHE/hub/$HF_MODEL_DIR" >&2
  echo "Re-run ./run.sh from your laptop — it ships the model via rsync." >&2
  exit 1
fi

mkdir -p logs
JOB_NAME=$(basename "$EXP" .yaml)
echo "Submitting → account=$ACCOUNT qos=$QOS time=$TIME gpus=$GPUS cpus=$CPUS"
sbatch \
  --account="$ACCOUNT" \
  --qos="$QOS" \
  --time="$TIME" \
  --gres="gpu:$GPUS" \
  --cpus-per-task="$CPUS" \
  --ntasks=1 \
  --nodes=1 \
  --job-name="$JOB_NAME" \
  --output="logs/%x-%j.out" \
  --error="logs/%x-%j.err" \
  --export=ALL,EXP="$EXP",SCRIPT="$SCRIPT",REMOTE_DIR="$REMOTE_DIR",HF_CACHE="$HF_CACHE" \
  cluster/run.sbatch
