#!/bin/bash
# LoRA SFT job: Qwen3-VL-2B-Instruct, swap detection, ALL FOUR datasets
# combined (CLEVRER + CRAFT + MTL-AQA + DriveLM) into one joint training run.
#
# Standardized before combining, per the three-rule scheme:
#   1. Cap at a common n -- 557 scenes/dataset, DriveLM's own 80/20 train
#      count (the smallest of the four), via scripts/cap_finetune_train.py.
#      CLEVRER (3998), CRAFT-full80 (7855), and MTL-AQA-full80 (1097) train
#      sets were all downsampled to 557, stratified by frame-length bucket,
#      so no single dataset dominates the combined gradient. DriveLM itself
#      needed no capping.
#   2. max_length = max of everything needed = 12288, DriveLM's requirement
#      for its real nuScenes photos (vs. 4096-8192 for the other three's
#      synthetic renders) -- see cluster/finetune_swap_detect_drivelm.sbatch
#      for why. Applying the largest max_length (and DriveLM's memory-safe
#      per_device_train_batch_size=1 / gradient_accumulation_steps=16) to
#      the whole combined run, since every example shares one global config.
#   3. Always a fresh 80/20 split of the full recovered pool, seed 42,
#      stratified by frame-length bucket -- already true for all four
#      datasets independently (CLEVRER/DriveLM's original splits and
#      CRAFT/MTL-AQA's "full80" splits all use the identical method), so no
#      new splitting was needed here, only re-using the existing train
#      manifests as the capping source.
#
# Combined train set: 2,227 examples (556 CLEVRER + 557 CRAFT + 557 MTL-AQA
# + 557 DriveLM), plain-concatenated into one jsonl -- image paths are
# already relative to the repo root in every export script, so no path
# rewriting needed. Held-out eval stays per-dataset (each dataset's own
# existing 80/20 held-out set, uncapped) via scripts/full_matrix_eval.py.
#
# ~133 steps/epoch (2,227 * 0.95 / effective-batch-16) * 3 = ~399 steps;
# eval/save every ~1/5 epoch = 27.

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
# Distinct rendezvous port -- 29500-29507 already used by earlier runs this
# project.
export MASTER_PORT=29508

mkdir -p checkpoints logs

srun swift sft \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --dataset datasets/combined/sft_swap_detect_train_combined.jsonl \
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
    --eval_steps 27 \
    --save_steps 27 \
    --logging_steps 5 \
    --seed 42 \
    --output_dir checkpoints/qwen3-vl-2b-lora-swap-detect-combined
