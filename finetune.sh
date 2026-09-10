#!/bin/bash
# Laptop entry point for the swap-detect LoRA pilot (Qwen3-VL-2B).
# Builds ms-swift wheels (Docker, linux/amd64), rsyncs code + the fine-tuning
# manifests/JSONL under the given dataset dir + wheels, then SSHes in to
# prepare_finetune.sh + sbatch. Mirrors run.sh; kept separate since training
# isn't YAML-driven.
#
# Usage: ./finetune.sh [dataset_dir] [sbatch_script] [time_limit] [job_name]
#   Defaults to the CLEVRER pilot run; pass a different dataset_dir (e.g.
#   datasets/drive_lm) + sbatch_script to train on another dataset.

set -euo pipefail
cd "$(dirname "$0")"

DATASET_DIR="${1:-datasets/CLEVRER}"

if [ ! -f cluster/env.sh ]; then
  echo "Missing cluster/env.sh — copy cluster/env.sh.example and fill it in." >&2
  exit 1
fi
source cluster/env.sh

PY_VER="${CLUSTER_PYTHON_VERSION:-3.12}"
if [ ! -f cluster/wheels-finetune/.built ] || [ cluster/requirements-finetune.txt -nt cluster/wheels-finetune/.built ]; then
  echo "[wheels] downloading Linux wheels (Python $PY_VER) → cluster/wheels-finetune/"
  if ! command -v docker &>/dev/null; then
    echo "ERROR: docker required to cross-build Linux wheels from macOS." >&2
    exit 1
  fi
  rm -rf cluster/wheels-finetune
  mkdir -p cluster/wheels-finetune "$HOME/.cache/pip"
  docker run --rm --platform linux/amd64 \
    -v "$PWD:/work" \
    -v "$HOME/.cache/pip:/root/.cache/pip" \
    -w /work \
    "python:${PY_VER}-slim" \
    pip download -d /work/cluster/wheels-finetune/ -r cluster/requirements-finetune.txt
  touch cluster/wheels-finetune/.built
fi

echo "[1/4] rsync code → $CLUSTER_USER@$CLUSTER_TRANSFER_HOST:$REMOTE_DIR"
rsync -avz --mkpath --info=progress2 \
  --exclude .venv --exclude .venv-finetune --exclude .git --exclude __pycache__ --exclude '.DS_Store' \
  --exclude /datasets --exclude /results --exclude /outputs --exclude /logs \
  --exclude /failures --exclude /models --exclude /checkpoints \
  --exclude /cluster/wheels --exclude /cluster/wheels-finetune --exclude /cluster/hf_cache \
  ./ "$CLUSTER_USER@$CLUSTER_TRANSFER_HOST:$REMOTE_DIR/"

echo "[2/4] rsync $DATASET_DIR (new fine-tuning manifests/JSONL — images already present from prior eval runs)"
rsync -avz --mkpath --info=progress2 \
  "$DATASET_DIR/" \
  "$CLUSTER_USER@$CLUSTER_TRANSFER_HOST:$REMOTE_DIR/$DATASET_DIR/"

echo "[3/4] rsync finetune wheels → $REMOTE_DIR/cluster/wheels-finetune/  (--delete: prune stale wheels)"
rsync -avz --mkpath --delete --info=progress2 \
  cluster/wheels-finetune/ \
  "$CLUSTER_USER@$CLUSTER_TRANSFER_HOST:$REMOTE_DIR/cluster/wheels-finetune/"

echo "[4/4] ssh $CLUSTER_INTERNET_HOST → prepare_finetune + sbatch"
ssh "$CLUSTER_USER@$CLUSTER_INTERNET_HOST" \
  "cd '$REMOTE_DIR' && bash cluster/prepare_finetune.sh ${2:+'$2'} ${3:+'$3'} ${4:+'$4'}"
