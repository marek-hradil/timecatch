#!/bin/bash
# LoRA SFT job: Qwen3-VL-2B-Instruct, swap detection, CLEVRER subsampled to
# ~1000 train scenes (from the full 3998-scene train pool, same held-out set
# unchanged). Identical hyperparameters to cluster/finetune_swap_detect.sbatch
# (the full-CLEVRER run) — this tests whether CLEVRER's train-set SIZE (4-15x
# larger than the other three datasets, which all failed to learn anything)
# is what let fine-tuning find its shortcut, or whether it's something about
# CLEVRER's data specifically regardless of amount. CRAFT (686 train scenes,
# closest in kind to CLEVRER) already failed at a similar scale to this run's
# ~1000, which is suggestive but not conclusive on its own.
#
# ~60 steps/epoch * 3 = ~180 steps; eval/save every ~1/5 epoch = 12.
#
# No save_total_limit / load_best_model_at_end — keep every checkpoint,
# select via scripts/checkpoint_sweep.py against the SAME held-out set used
# for the full-CLEVRER run, for a direct comparison.

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
# Distinct rendezvous port — avoids the same collision hit between concurrent
# CRAFT/MTL-AQA training jobs earlier in this project.
export MASTER_PORT=29502

mkdir -p checkpoints logs

srun swift sft \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --dataset datasets/CLEVRER/sft_swap_detect_train_1000.jsonl \
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
    --eval_steps 12 \
    --save_steps 12 \
    --logging_steps 3 \
    --seed 42 \
    --output_dir checkpoints/qwen3-vl-2b-lora-swap-detect-clevrer-1000
