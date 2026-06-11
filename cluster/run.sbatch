#!/bin/bash
# Actual job body. Runs on a compute node with no internet access.
# All SBATCH directives are passed via command-line flags from prepare.sh.

set -euo pipefail
cd "$REMOTE_DIR"

module purge
# sqlite3 provides a newer libsqlite3.so.0 with sqlite3_deserialize symbol;
# the cluster's system libsqlite3 is too old for the Python module's
# precompiled _sqlite3.so, and vllm 0.11 imports it eagerly via diskcache.
module load oneapi hdf5 python sqlite3/3.45.2-gcc nvidia-hpc-sdk/25.3
source .venv/bin/activate
# Module-loaded PYTHONHOME/PYTHONPATH point at the cluster's system site-packages
# and would shadow the venv on sys.path. Drop them so the venv is properly isolated.
unset PYTHONHOME PYTHONPATH

# Cluster kernel driver natively supports CUDA 12.2. nvidia-hpc-sdk/25.3 ships
# a forward-compat libcuda (570.124.06) that lets torch+cu128 (vllm 0.11) run.
export LD_LIBRARY_PATH="/apps/ACC/NVIDIA-HPC-SDK/25.3/Linux_x86_64/25.3/cuda/12.8/compat:${LD_LIBRARY_PATH:-}"

# The nvhpc module also puts `nvc` first on PATH; torch.inductor would pick it
# up for JIT C compilation and fail on GCC-specific flags (-Wno-psabi).
# Force the system GCC instead.
export CC=gcc
export CXX=g++
# Allow Molmo's 4096 position embedding limit to be overridden.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# Force the V0 engine: Gemma4's dummy-input generation is broken in vLLM 0.19.1's V1 engine.
export VLLM_USE_V1=0

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

srun python "scripts/${SCRIPT}.py" "$EXP"
