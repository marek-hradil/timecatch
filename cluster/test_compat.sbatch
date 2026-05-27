#!/bin/bash
# Forward-compat smoke test:
# Verifies torch+cu124 can talk to the cluster's H100 via NVIDIA's
# forward-compatibility libcuda from NVHPC SDK 25.1.
set -euo pipefail
cd "$REMOTE_DIR"

module purge
module load oneapi hdf5 python nvidia-hpc-sdk/25.1
source .venv-cu124-test/bin/activate
unset PYTHONHOME PYTHONPATH

# Prepend the forward-compat libcuda directory so cu124 torch sees a newer userspace driver.
export LD_LIBRARY_PATH="/apps/ACC/NVIDIA-HPC-SDK/25.1/Linux_x86_64/25.1/cuda/12.6/compat:${LD_LIBRARY_PATH:-}"

python - <<'EOF'
import torch
print(f"torch.__version__   = {torch.__version__}")
print(f"torch.version.cuda  = {torch.version.cuda}")
print(f"is_available        = {torch.cuda.is_available()}")
print(f"device_count        = {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"device_name(0)      = {torch.cuda.get_device_name(0)}")
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x.t()
    torch.cuda.synchronize()
    print(f"matmul on GPU OK    | result.shape={tuple(y.shape)} | device={y.device}")
else:
    raise SystemExit("torch.cuda.is_available() returned False")
EOF
