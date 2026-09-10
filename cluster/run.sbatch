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

# The nvhpc module also puts `nvc` first on PATH; torch.inductor would pick it
# up for JIT C compilation and fail on GCC-specific flags (-Wno-psabi).
# Force the system GCC instead.
export CC=gcc
export CXX=g++
# Allow Molmo's 4096 position embedding limit to be overridden.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# VLLM_USE_V1=0 is unrecognized in vLLM 0.19.1 (V1 is always on). Disabling the
# EngineCore subprocess avoids an IPC deadlock on InternVL3.5 initialization.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

srun python "${SCRIPT}.py" "$EXP"
