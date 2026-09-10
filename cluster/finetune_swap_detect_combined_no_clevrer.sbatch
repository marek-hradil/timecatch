#!/bin/bash
# LoRA SFT job: Qwen3-VL-2B-Instruct, swap detection, CRAFT + MTL-AQA +
# DriveLM combined -- CLEVRER excluded entirely. Tests whether fine-tuning
# on the other three datasets generalizes to CLEVRER's held-out set (never
# seen in any form during this run) vs. the zero-shot baseline and vs. the
# all-four-combined run (cluster/finetune_swap_detect_combined.sbatch).
#
# Same standardization as the all-four combined run: each dataset capped to
# 557 scenes (DriveLM's own count, the smallest of the four, used as the
# common n) via scripts/cap_finetune_train.py; max_length=12288 (DriveLM's
# requirement); per_device_train_batch_size=1 / gradient_accumulation_steps=16
# (DriveLM's memory-safe config, applied globally).
#
# Combined train set: 1,671 examples (557 CRAFT + 557 MTL-AQA + 557 DriveLM).
# ~100 steps/epoch (1,671 * 0.95 / effective-batch-16) * 3 = ~300 steps;
# eval/save every ~1/5 epoch = 20.

set -euo pipefail
cd "$REMOTE_DIR"

module purge
module load oneapi hdf5 python sqlite3/3.45.2-gcc nvidia-hpc-sdk/25.3
source .venv-finetune/bin/activate
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

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NPROC_PER_NODE=1
export USE_HF=1
# Distinct rendezvous port -- 29500-29508 already used by earlier runs this
# project.
export MASTER_PORT=29509

mkdir -p checkpoints logs

srun swift sft \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --dataset datasets/combined/sft_swap_detect_train_no_clevrer.jsonl \
    --split_dataset_ratio 0.05 \
    --tuner_type lora \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 12288 \
    --eval_strategy steps \
    --eval_steps 20 \
    --save_steps 20 \
    --logging_steps 5 \
    --seed 42 \
    --output_dir checkpoints/qwen3-vl-2b-lora-swap-detect-combined-no-clevrer
