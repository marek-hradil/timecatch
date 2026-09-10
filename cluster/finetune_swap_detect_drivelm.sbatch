#!/bin/bash
# LoRA SFT job: Qwen3-VL-2B-Instruct, swap detection, DriveLM train half.
# Identical hyperparameters to cluster/finetune_swap_detect.sbatch (CLEVRER) —
# only the dataset changes, to test whether the CLEVRER run's shortcut
# behavior is dataset-specific or a general property of this training setup.
#
# DriveLM's train half is much smaller than CLEVRER's (557 vs 3998 scenes),
# so total steps and eval/save cadence are scaled down accordingly:
# ~34 steps/epoch * 3 epochs = ~102 steps; eval/save every ~1/5 epoch = 7.
# No save_total_limit / load_best_model_at_end — same reasoning as the
# CLEVRER retrain: keep every checkpoint, select the real one via
# scripts/checkpoint_sweep.py against the true held-out set, not ms-swift's
# internal (and previously misleading) eval_loss.
#
# max_length raised to 12288 (vs. 4096 for CLEVRER): DriveLM's real nuScenes
# photos need far more vision tokens than CLEVRER's simple synthetic renders,
# and its scene descriptions are full paragraphs rather than short templated
# sentences. 4096 crashed on the first example; 8192 got further (25/102
# steps) before crashing on a longer outlier; 16384 avoided that crash but
# OOM'd instead — the lm_head logits tensor scales with seq_len * vocab_size
# (151936), so doubling max_length nearly doubled peak memory. Splitting the
# difference at 12288, and per_device_train_batch_size dropped 2->1 (with
# gradient_accumulation_steps doubled 8->16 to keep the same effective batch
# of 16) to directly cut peak memory rather than fight it via max_length alone.

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

mkdir -p checkpoints logs

srun swift sft \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --dataset datasets/drive_lm/sft_swap_detect_train.jsonl \
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
    --eval_steps 7 \
    --save_steps 7 \
    --logging_steps 2 \
    --seed 42 \
    --output_dir checkpoints/qwen3-vl-2b-lora-swap-detect-drivelm
