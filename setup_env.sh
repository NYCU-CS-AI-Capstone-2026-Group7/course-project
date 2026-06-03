#!/bin/bash
set -e

echo "=========================================="
echo "   Isaac Sim & IsaacLab uv Setup Script   "
echo "=========================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 1. Initialize submodules if IsaacLab is empty
if [ ! -f "dependencies/IsaacLab/isaaclab.sh" ]; then
    echo "[INFO] Initializing git submodules for IsaacLab..."
    git submodule update --init --recursive dependencies/IsaacLab
fi

# 2. Setup uv virtual environment
VENV_DIR="/tmp/.venv"
echo "[1/5] Creating uv virtual environment at $VENV_DIR..."
uv venv $VENV_DIR --python 3.11
source $VENV_DIR/bin/activate

# 3. Install PyTorch & Isaac Sim
echo "[2/5] Installing PyTorch and Isaac Sim..."
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
uv pip install flatdict==4.0.0 huggingface-hub==0.35.3 transformers==4.57.6

# 4. Prepare IsaacLab build dependencies
echo "[3/5] Preparing IsaacLab build dependencies (numpy==1.26.0)..."
uv pip install setuptools==65 wheel==0.45.1 toml==0.10.2 packaging==23.0 poetry-core==2.2.1 numpy==1.26.0

# 5. Compile and install IsaacLab
echo "[4/5] Compiling and installing IsaacLab extensions..."
# Temporarily patch the script to use 'uv pip' for faster installation
sed -i 's/python -m pip/uv pip/g' dependencies/IsaacLab/isaaclab.sh
pushd dependencies/IsaacLab > /dev/null
./isaaclab.sh --install
popd > /dev/null

# 6. Install Simulator
echo "[5/5] Installing Simulator without build-isolation..."
uv pip install --no-build-isolation -e packages/simulator
uv pip install numpy==1.26.0

echo "=========================================="
echo "✅ Environment setup complete!"
echo "To activate the environment, run:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "Don't forget to export your Vulkan/GUI variables if running locally, e.g.:"
echo "    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json"
echo "    export OMNI_KIT_ARGS=\"--/renderer/multiGpu/enabled=false --/app/renderer/activeGpu=0 --/rtx/verifyDriverVersion/enabled=false\""
echo "    export OMNI_KIT_DISABLE_DRIVER_VERSION_CHECK=1"
echo "    export OMNI_DISABLE_DRIVER_CHECK=1"
echo "=========================================="
