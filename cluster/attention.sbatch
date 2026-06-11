#!/bin/bash
# Run attention rollout analysis on a GPU node using transformers (not vllm).
# Much lower throughput than vllm but exposes attention weights.
set -euo pipefail
cd "$REMOTE_DIR"

module purge
module load oneapi hdf5 python sqlite3/3.45.2-gcc nvidia-hpc-sdk/25.3
source .venv/bin/activate
unset PYTHONHOME PYTHONPATH

export LD_LIBRARY_PATH="/apps/ACC/NVIDIA-HPC-SDK/25.3/Linux_x86_64/25.3/cuda/12.8/compat:${LD_LIBRARY_PATH:-}"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CC=gcc
export CXX=g++

srun python scripts/analyze_attention.py \
    config/experiments/${ATTN_TASK:-swap}-detect-mtl-aqa-qwen3-vl-8b.yaml \
    scene_14_68 scene_04_66 scene_06_54 scene_02_10 scene_03_51 \
    --task "${ATTN_TASK:-swap}"
