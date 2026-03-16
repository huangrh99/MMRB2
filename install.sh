#!/bin/bash
# MMRB2 环境安装
set -e

if [ -d ~/mmrb2_env ] && [ -f ~/mmrb2_env/bin/activate ]; then
    echo "Existing venv found at ~/mmrb2_env, activating..."
    source ~/mmrb2_env/bin/activate
else
    echo "Creating venv at ~/mmrb2_env ..."
    uv venv ~/mmrb2_env
    source ~/mmrb2_env/bin/activate
fi

echo "Installing dependencies..."

uv pip install torch==2.9.1 \
    'sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python' \
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
import torch, transformers, sglang
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  sglang:       {sglang.__version__}')
"

echo "To activate: source ~/mmrb2_env/bin/activate"
