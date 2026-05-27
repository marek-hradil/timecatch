#!/bin/bash
# Laptop entry point. Reads an experiment YAML, builds the cluster bundle
# (Linux wheels + HF model cache), rsyncs everything, then SSHes in to sbatch.
#
# Usage: ./run.sh config/experiments/corrupt-detect-smoke.yaml

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f cluster/env.sh ]; then
  echo "Missing cluster/env.sh — copy cluster/env.sh.example and fill it in." >&2
  exit 1
fi
source cluster/env.sh

if [ $# -lt 1 ]; then
  echo "Usage: $0 <config/experiments/foo.yaml>" >&2
  exit 1
fi
EXP="$1"

if ! command -v yq &>/dev/null; then
  echo "yq required on laptop. Install with: brew install yq" >&2
  exit 1
fi

# Ship a Linux yq binary to the cluster (compute + login nodes have no internet).
if [ ! -f cluster/bin/yq ]; then
  echo "Fetching yq linux binary for cluster..."
  mkdir -p cluster/bin
  curl -fsSL -o cluster/bin/yq \
    https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
  chmod +x cluster/bin/yq
fi

DATASET_KEY=$(yq -r '.dataset' "$EXP")
DATASET_PATH=$(yq -r '.path' "config/datasets/$DATASET_KEY.yaml")
MODEL_KEY=$(yq -r '.model' "$EXP")
MODEL_PATH=$(yq -r '.path' "config/models/$MODEL_KEY.yaml")

# Build Linux wheels for cluster Python via Docker — native marker evaluation
# (pip download --platform on Mac silently skips deps gated by sys_platform/platform_machine).
PY_VER="${CLUSTER_PYTHON_VERSION:-3.12}"
if [ ! -f cluster/wheels/.built ] || [ cluster/requirements.txt -nt cluster/wheels/.built ]; then
  echo "[wheels] downloading Linux wheels (Python $PY_VER) → cluster/wheels/"
  if ! command -v docker &>/dev/null; then
    echo "ERROR: docker required to cross-build Linux wheels from macOS." >&2
    echo "Install Docker Desktop, then re-run." >&2
    exit 1
  fi
  rm -rf cluster/wheels
  mkdir -p cluster/wheels "$HOME/.cache/pip"
  docker run --rm --platform linux/amd64 \
    -v "$PWD:/work" \
    -v "$HOME/.cache/pip:/root/.cache/pip" \
    -w /work \
    "python:${PY_VER}-slim" \
    pip download -d /work/cluster/wheels/ -r cluster/requirements.txt
  touch cluster/wheels/.built
fi

# Ensure HF model is cached locally (laptop).
LOCAL_HF_CACHE="cluster/hf_cache"
HF_MODEL_DIR="models--$(echo "$MODEL_PATH" | sed 's|/|--|g')"
if [ ! -d "$LOCAL_HF_CACHE/hub/$HF_MODEL_DIR" ]; then
  echo "[hf] downloading model '$MODEL_PATH' → $LOCAL_HF_CACHE/"
  if command -v hf &>/dev/null; then
    HF_DL=(hf)
  elif command -v uv &>/dev/null; then
    HF_DL=(uv run --with huggingface_hub --no-project -- hf)
  else
    echo "Need the 'hf' CLI. Install via:  pip install --user huggingface_hub" >&2
    exit 1
  fi
  HF_HUB_CACHE="$PWD/$LOCAL_HF_CACHE/hub" "${HF_DL[@]}" download "$MODEL_PATH"
fi

echo "[1/5] rsync code → $CLUSTER_USER@$CLUSTER_TRANSFER_HOST:$REMOTE_DIR"
rsync -avz --mkpath --info=progress2 \
  --exclude .venv --exclude .git --exclude __pycache__ --exclude '.DS_Store' \
  --exclude /datasets --exclude /results --exclude /outputs --exclude /logs \
  --exclude /failures --exclude /models \
  --exclude /cluster/wheels --exclude /cluster/hf_cache \
  ./ "$CLUSTER_USER@$CLUSTER_TRANSFER_HOST:$REMOTE_DIR/"

echo "[2/5] rsync dataset '$DATASET_KEY' from $DATASET_PATH/"
rsync -avz --mkpath --info=progress2 \
  "$DATASET_PATH/" \
  "$CLUSTER_USER@$CLUSTER_TRANSFER_HOST:$REMOTE_DIR/$DATASET_PATH/"

echo "[3/5] rsync wheels → $REMOTE_DIR/cluster/wheels/  (--delete: prune stale wheels)"
rsync -avz --mkpath --delete --info=progress2 \
  cluster/wheels/ \
  "$CLUSTER_USER@$CLUSTER_TRANSFER_HOST:$REMOTE_DIR/cluster/wheels/"

echo "[4/5] rsync HF model '$MODEL_PATH' → $HF_CACHE/"
rsync -avz --mkpath --info=progress2 \
  "$LOCAL_HF_CACHE/hub/$HF_MODEL_DIR/" \
  "$CLUSTER_USER@$CLUSTER_TRANSFER_HOST:$HF_CACHE/hub/$HF_MODEL_DIR/"

echo "[5/5] ssh $CLUSTER_INTERNET_HOST → prepare + sbatch"
ssh "$CLUSTER_USER@$CLUSTER_INTERNET_HOST" \
  "cd '$REMOTE_DIR' && bash cluster/prepare.sh '$EXP'"
