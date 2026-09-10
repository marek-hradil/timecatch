#!/bin/bash
# LoRA SFT job: Qwen3-VL-2B-Instruct, swap detection, CRAFT + MTL-AQA +
# DriveLM combined, UNCAPPED (full recovered pools, not the 557/dataset cap
# used in cluster/finetune_swap_detect_combined_no_clevrer.sbatch). CLEVRER
# still excluded entirely.
#
# Tests whether that earlier run's near-chance CLEVRER transfer (51.6%
# detect vs. 49.6% zero-shot) was an artifact of the deliberately small
# 557-scene-per-dataset cap, or a genuine "this doesn't transfer to
# CLEVRER" finding: same three datasets, ~5.7x more training data (9,509
# vs. 1,671 examples: 7,855 CRAFT-full80 + 1,097 MTL-AQA-full80 + 557
# DriveLM, imbalanced -- no cap this time, since the whole point is testing
# whether more data (even skewed toward CRAFT) changes the transfer result).
#
# Same max_length/batch config as the capped run (12288 / batch=1 /
# grad_accum=16 -- DriveLM's requirement, applied globally).
#
# ~565 steps/epoch (9,509 * 0.95 / effective-batch-16) * 3 = ~1,695 steps;
# eval/save every ~1/5 epoch = 113. Expect ~7-8h given the capped run's
# pace (300 steps / 1h18m).

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
# Distinct rendezvous port -- 29500-29510 already used by earlier runs this
# project.
export MASTER_PORT=29511

mkdir -p checkpoints logs

srun swift sft \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --dataset datasets/combined/sft_swap_detect_train_no_clevrer_full.jsonl \
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
    --eval_steps 113 \
    --save_steps 113 \
    --logging_steps 20 \
    --seed 42 \
    --output_dir checkpoints/qwen3-vl-2b-lora-swap-detect-combined-no-clevrer-full
