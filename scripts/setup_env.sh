#!/usr/bin/env bash
# One-command environment setup for ABSA-QUAD
set -e

echo "=== Creating conda environment ==="
mamba env remove -n absa -y 2>/dev/null || true
mamba env create -f environment.yml

echo "=== Activating environment ==="
eval "$(mamba shell hook --shell bash)"
mamba activate absa

echo "=== Installing PyTorch with CUDA ==="
# Detect CUDA version and select matching wheel
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
if python -c "import subprocess; v=subprocess.check_output(['nvidia-smi'],text=True); print(v)" 2>/dev/null | grep -q "12\.[89]"; then
    TORCH_INDEX="cu128"
elif python -c "import subprocess; v=subprocess.check_output(['nvidia-smi'],text=True); print(v)" 2>/dev/null | grep -q "12\.[4567]"; then
    TORCH_INDEX="cu124"
else
    TORCH_INDEX="cu124"
fi
echo "Using PyTorch index: ${TORCH_INDEX}"
pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

echo "=== Configuring libstdc++ compatibility ==="
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' \
    > "${CONDA_PREFIX}/etc/conda/activate.d/libstdcxx.sh"

echo ""
echo "=== Setup complete! Run: mamba activate absa ==="
