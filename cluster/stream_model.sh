#!/bin/bash
# Stream-submit a model: download locally → rsync to cluster → submit 20 jobs → delete local copy.
# Usage:  ./cluster/stream_model.sh <model_key>
# where <model_key> matches config/models/<model_key>.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh

if [ $# -lt 1 ]; then
  echo "Usage: $0 <model_key>" >&2
  exit 1
fi
MODEL_KEY="$1"
MODEL_YAML="config/models/${MODEL_KEY}.yaml"
if [ ! -f "$MODEL_YAML" ]; then
  echo "Missing $MODEL_YAML" >&2
  exit 1
fi

MODEL_PATH=$(yq -r '.path' "$MODEL_YAML")
HF_MODEL_DIR="models--$(echo "$MODEL_PATH" | sed 's|/|--|g')"
LOCAL_HF_CACHE="cluster/hf_cache"

# 1. Generate 20 experiment YAMLs for this model, reusing the system prompts from the qwen2-5-vl-7b set.
echo "[1/5] generating 20 YAMLs for $MODEL_KEY"
DETECT_PROMPT=$(yq -r '.system_prompt' config/experiments/corrupt-detect-mtl-aqa.yaml)
CLOC_PROMPT=$(yq -r '.system_prompt' config/experiments/corrupt-localize-mtl-aqa.yaml)
SDET_PROMPT=$(yq -r '.system_prompt' config/experiments/swap-detect-mtl-aqa.yaml)
SLOC_PROMPT=$(yq -r '.system_prompt' config/experiments/swap-localize-mtl-aqa.yaml)
generate() {
  local script_kebab="$1" script_snake="$2" prompt="$3" dataset="$4"
  cat > "config/experiments/${script_kebab}-${dataset}-${MODEL_KEY}.yaml" <<YAML
model: ${MODEL_KEY}
dataset: ${dataset}
seed: 42
script: ${script_snake}
batch_size: 64

slurm:
  qos: acc_ehpc
  time: "01:00:00"
  gpus: 1
  cpus_per_gpu: 20

system_prompt: >
  ${prompt}
YAML
}
for ds in clevrer craft craft-long drive-lm mtl-aqa; do
  generate corrupt-detect    corrupt_detect    "$DETECT_PROMPT" "$ds"
  generate corrupt-localize  corrupt_localize  "$CLOC_PROMPT"   "$ds"
  generate swap-detect       swap_detect       "$SDET_PROMPT"   "$ds"
  generate swap-localize     swap_localize     "$SLOC_PROMPT"   "$ds"
done

# 2. Download model locally if missing.
if [ ! -d "$LOCAL_HF_CACHE/hub/$HF_MODEL_DIR" ]; then
  echo "[2/5] downloading $MODEL_PATH → $LOCAL_HF_CACHE/"
  mkdir -p "$LOCAL_HF_CACHE/hub"
  if command -v hf &>/dev/null; then
    HF_DL=(hf)
  elif command -v uv &>/dev/null; then
    HF_DL=(uv run --with huggingface_hub --no-project -- hf)
  else
    echo "Need hf CLI: pip install --user huggingface_hub" >&2
    exit 1
  fi
  HF_HUB_CACHE="$PWD/$LOCAL_HF_CACHE/hub" "${HF_DL[@]}" download "$MODEL_PATH"
else
  echo "[2/5] model already cached locally"
fi

# 3. Rsync code (incl. updated scripts + new YAMLs) to cluster.
echo "[3/5] rsync code + new YAMLs to cluster"
rsync -avz --mkpath \
  --exclude .venv --exclude .git --exclude __pycache__ --exclude '.DS_Store' \
  --exclude /datasets --exclude /results --exclude /outputs --exclude /logs \
  --exclude /failures --exclude /models \
  --exclude /cluster/wheels --exclude /cluster/hf_cache \
  ./ "$CLUSTER_USER@$CLUSTER_TRANSFER_HOST:$REMOTE_DIR/"

# 4. Rsync model weights to cluster.
echo "[4/5] rsync $MODEL_PATH → $HF_CACHE/"
rsync -avz --mkpath --info=progress2 \
  "$LOCAL_HF_CACHE/hub/$HF_MODEL_DIR/" \
  "$CLUSTER_USER@$CLUSTER_TRANSFER_HOST:$HF_CACHE/hub/$HF_MODEL_DIR/"

# 5. Submit 20 jobs + free local space.
echo "[5/5] submitting 20 jobs"
ssh "$CLUSTER_USER@$CLUSTER_INTERNET_HOST" "
set -uo pipefail
cd '$REMOTE_DIR'
for ds in clevrer craft craft-long drive-lm mtl-aqa; do
  for script in corrupt-detect corrupt-localize swap-detect swap-localize; do
    yaml=\"config/experiments/\${script}-\${ds}-${MODEL_KEY}.yaml\"
    jobid=\$(bash cluster/submit.sh \"\$yaml\" 2>&1)
    echo \"  \${script}-\${ds}-${MODEL_KEY} → \$jobid\"
  done
done
"

echo ""
echo "freeing local $MODEL_KEY copy + xet chunk cache..."
rm -rf "$LOCAL_HF_CACHE/hub/$HF_MODEL_DIR"
rm -rf "$LOCAL_HF_CACHE/hub/.locks"
rm -rf "$HOME/.cache/huggingface"
df -h /Users/marekhradil | awk 'NR==2'

echo ""
echo "Done with $MODEL_KEY. Jobs are queued on the cluster."
