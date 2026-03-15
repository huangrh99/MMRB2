#!/bin/bash
# MMRB2 环境安装
set -e

echo "Creating venv at ~/mmrb2_env ..."
uv venv ~/mmrb2_env
source ~/mmrb2_env/bin/activate

echo "Installing dependencies..."

uv pip install torch==2.9.1 \
    'sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python' \
    openai json-repair Pillow tqdm datasets huggingface_hub

uv pip install --no-deps 'transformers @ git+https://github.com/huggingface/transformers.git@main'

echo ""
echo "Done. Versions:"
python -c "
import torch, transformers, sglang
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  sglang:       {sglang.__version__}')
"
echo ""
echo "To activate: source ~/mmrb2_env/bin/activate"
