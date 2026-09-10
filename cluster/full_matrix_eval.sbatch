#!/bin/bash
# Runs scripts/full_matrix_eval.py for one trained checkpoint: own held-out
# swap-detect, own held-out swap-localize, transfer to the other 3 datasets.
# Uses the vLLM eval venv (.venv), not .venv-finetune.
# LORA_PATH and HOME_DATASET must be passed via --export.

set -euo pipefail
cd "$REMOTE_DIR"

module purge
module load oneapi hdf5 python sqlite3/3.45.2-gcc nvidia-hpc-sdk/25.3
source .venv/bin/activate
unset PYTHONHOME PYTHONPATH

# Only prepend the CUDA forward-compat shim if the node's native driver
# doesn't already support cu128 -- on newer nodes (H100, driver 595.x,
# natively CUDA 13.2) this old compat stub (bundled driver 570.124.06)
# actively breaks CUDA init (torch.cuda.is_available() -> False, silent
# CPU fallback) instead of helping, since it doesn't match the loaded
# kernel driver module. Detected and fixed 2026-08-03 after MTL-AQA runs
# repeatedly landed on H100 nodes and trained at ~250s/it instead of ~5s/it.
NATIVE_CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
if [ -n "$NATIVE_CUDA_VER" ] && awk "BEGIN{exit !($NATIVE_CUDA_VER < 12.8)}"; then
  echo "Native driver reports CUDA $NATIVE_CUDA_VER < 12.8 -- using forward-compat shim"
  export LD_LIBRARY_PATH="/apps/ACC/NVIDIA-HPC-SDK/25.3/Linux_x86_64/25.3/cuda/12.8/compat:${LD_LIBRARY_PATH:-}"
else
  echo "Native driver reports CUDA ${NATIVE_CUDA_VER:-unknown} >= 12.8 -- skipping forward-compat shim"
fi
export CC=gcc
export CXX=g++
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [ -n "${HELDOUT_MANIFEST:-}" ]; then
  srun python scripts/full_matrix_eval.py "$LORA_PATH" "$HOME_DATASET" "$HELDOUT_MANIFEST"
else
  srun python scripts/full_matrix_eval.py "$LORA_PATH" "$HOME_DATASET"
fi
