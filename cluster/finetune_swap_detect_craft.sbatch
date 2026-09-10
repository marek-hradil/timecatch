#!/bin/bash
# LoRA SFT job: Qwen3-VL-2B-Instruct, swap detection, CRAFT train half.
# Identical hyperparameters to cluster/finetune_swap_detect.sbatch (CLEVRER) —
# CRAFT's images are small synthetic renders (256x256, like CLEVRER) with
# short descriptions, so no max_length/batch adjustment needed unlike DriveLM.
#
# CRAFT's train half (686 scenes) is smaller than CLEVRER's (3998), so total
# steps and eval/save cadence scale down: ~41 steps/epoch * 3 = ~123 steps;
# eval/save every ~1/5 epoch = 8.
#
# No save_total_limit / load_best_model_at_end — keep every checkpoint,
# select via scripts/checkpoint_sweep.py against the true held-out set.

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
# Distinct rendezvous port so future concurrent training jobs on the same
# physical node don't collide on torchrun's default port 29500 (this
# happened between this job and a concurrent MTL-AQA submission).
export MASTER_PORT=29500

mkdir -p checkpoints logs

srun swift sft \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --dataset datasets/CRAFT/sft_swap_detect_train.jsonl \
    --split_dataset_ratio 0.05 \
    --tuner_type lora \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 4096 \
    --eval_strategy steps \
    --eval_steps 8 \
    --save_steps 8 \
    --logging_steps 2 \
    --seed 42 \
    --output_dir checkpoints/qwen3-vl-2b-lora-swap-detect-craft
