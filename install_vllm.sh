#!/bin/bash
# MMRB2 环境安装 (vLLM backend)
set -e

VENV_DIR=~/mmrb2_env

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Existing venv found at $VENV_DIR, activating..."
    source "$VENV_DIR/bin/activate"
else
    echo "Creating venv at $VENV_DIR ..."
    uv venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

echo "Installing dependencies..."

# vLLM from nightly (Qwen3.5 official recommendation)
uv pip install torch==2.9.1 \
    'vllm @ git+https://github.com/vllm-project/vllm.git@main' \
    openai json-repair Pillow tqdm datasets

uv pip install -U huggingface_hub \
    google-generativeai \
    google-genai

# flash-attn: --no-build-isolation so it can find torch already installed above
uv pip install flash-attn --no-build-isolation

uv pip install --no-deps 'transformers @ git+https://github.com/huggingface/transformers.git@main'

uv pip install sympy httpx==0.23.3

echo ""
echo "Done. Versions:"
python -c "
import torch, transformers, vllm
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  vllm:         {vllm.__version__}')
"

echo "To activate: source $VENV_DIR/bin/activate"
