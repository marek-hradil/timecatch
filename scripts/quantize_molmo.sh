#!/bin/bash
set -euo pipefail

QUANT="${1:-Q4_K_M}"
MODEL_ID="allenai/Molmo-7B-D-0924"
MODEL_DIR="$HOME/Molmo-7B-D-0924"
LLAMA_DIR="$HOME/llama.cpp"
OUTPUT_F16="$HOME/molmo-7b-f16.gguf"
OUTPUT_QUANT="$HOME/molmo-7b-${QUANT}.gguf"

echo "==> Installing system dependencies"
sudo apt-get update -qq
sudo apt-get install -y -qq git cmake build-essential python3 python3-pip python3-venv

echo "==> Cloning llama.cpp"
if [ ! -d "$LLAMA_DIR" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

echo "==> Building llama.cpp"
cmake -B "$LLAMA_DIR/build" "$LLAMA_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)"

echo "==> Setting up Python venv"
python3 -m venv "$HOME/venv"
source "$HOME/venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet huggingface_hub hf_transfer
pip install --quiet -r "$LLAMA_DIR/requirements.txt"

echo "==> Downloading model from HuggingFace"
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$MODEL_ID', local_dir='$MODEL_DIR')
"

echo "==> Converting to GGUF (F16)"
python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" --outfile "$OUTPUT_F16"

echo "==> Quantizing to $QUANT"
"$LLAMA_DIR/build/bin/llama-quantize" "$OUTPUT_F16" "$OUTPUT_QUANT" "$QUANT"

echo "==> Cleaning up F16 file to save disk space"
rm "$OUTPUT_F16"

echo ""
echo "Done! Quantized model: $OUTPUT_QUANT"
echo "Size: $(du -sh "$OUTPUT_QUANT" | cut -f1)"
