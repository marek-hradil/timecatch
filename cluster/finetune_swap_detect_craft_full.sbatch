#!/bin/bash
# LoRA SFT job: Qwen3-VL-2B-Instruct, swap detection, CRAFT train half
# EXPANDED to include CRAFT-long (9-16 frame sequences), which the original
# CRAFT fine-tune (cluster/finetune_swap_detect_craft.sbatch) excluded only
# to keep CRAFT-long clean for the paper's separate sequence-length ablation
# (measured on the zero-shot base model — unaffected by reuse here). Train
# set: 1,787 scenes (original 686 at 4-8 frames + all 1,101 CRAFT-long at
# 9-16 frames) vs. the held-out set, UNCHANGED at 172 scenes (4-8 frames
# only), for direct comparability with the original 686-scene CRAFT result
# (45.3% final swap-detect accuracy, flat/no shortcut found).
#
# max_length bumped 4096 -> 8192 vs. the original CRAFT run: CRAFT-long
# sequences have up to 16 frames (2x the original 8-frame cap), so image
# token count roughly doubles. Images are still small synthetic renders
# (256x256, like CLEVRER) so no batch-size/memory adjustment expected to be
# needed unlike DriveLM's real photos.
#
# ~107 steps/epoch * 3 = ~321 steps; eval/save every ~1/5 epoch = 21.
#
# No save_total_limit / load_best_model_at_end — keep every checkpoint,
# select via scripts/checkpoint_sweep.py against the SAME held-out set used
# for the original CRAFT run, for a direct comparison.

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
# Distinct rendezvous port — 29500/29501/29502/29503 already used by the
# original CRAFT, MTL-AQA, CLEVRER-1000, and CLEVRER-2000 runs.
export MASTER_PORT=29504

mkdir -p checkpoints logs

srun swift sft \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --dataset datasets/CRAFT/sft_swap_detect_train_full.jsonl \
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
    --max_length 8192 \
    --eval_strategy steps \
    --eval_steps 21 \
    --save_steps 21 \
    --logging_steps 3 \
    --seed 42 \
    --output_dir checkpoints/qwen3-vl-2b-lora-swap-detect-craft-full
