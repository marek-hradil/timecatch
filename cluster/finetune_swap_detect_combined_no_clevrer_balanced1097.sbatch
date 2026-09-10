#!/bin/bash
# LoRA SFT job: Qwen3-VL-2B-Instruct, swap detection, CRAFT + MTL-AQA +
# DriveLM combined, BALANCED at n=1097/dataset (2x the 557-cap run's size),
# CLEVRER excluded entirely. Supersedes the uncapped
# cluster/finetune_swap_detect_combined_no_clevrer_full.sbatch attempt,
# which used the full recovered pools (7,855 CRAFT / 1,097 MTL-AQA / 557
# DriveLM) -- ~14:2:1 imbalanced, so any change in CLEVRER's transfer score
# would have been confounded with "the model just leans into CRAFT's
# statistics" rather than isolating "more data helps".
#
# DriveLM's pool (696 scenes total) caps out at 557 train examples at
# 80/20, so a perfectly balanced cap can't go higher than that with unique
# data alone. Instead: CRAFT capped down from 7,855 to MTL-AQA's own
# ceiling (1,097, via scripts/cap_finetune_train.py), MTL-AQA kept at its
# full 1,097, and DriveLM's 557 duplicated 2x (1,114) to match -- a real 2x
# increase in unique data for two of three datasets, DriveLM's caveat being
# duplication rather than new data (unavoidable, already at its ceiling).
#
# Combined train set: 3,308 examples (1,097 CRAFT + 1,097 MTL-AQA + 1,114
# DriveLM-duplicated). Same max_length/batch config as prior combined runs
# (12288 / batch=1 / grad_accum=16 -- DriveLM's requirement, applied
# globally).
#
# ~197 steps/epoch (3,308 * 0.95 / effective-batch-16) * 3 = ~591 steps;
# eval/save every ~1/5 epoch = 39. Expect ~2.5h given the 557-cap run's
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
# Distinct rendezvous port -- 29500-29511 already used by earlier runs this
# project.
export MASTER_PORT=29512

mkdir -p checkpoints logs

srun swift sft \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --dataset datasets/combined/sft_swap_detect_train_no_clevrer_balanced1097.jsonl \
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
    --eval_steps 39 \
    --save_steps 39 \
    --logging_steps 10 \
    --seed 42 \
    --output_dir checkpoints/qwen3-vl-2b-lora-swap-detect-combined-no-clevrer-balanced1097
