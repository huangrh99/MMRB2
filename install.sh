#!/bin/bash
# MMRB2 环境安装脚本
#
# Usage:
#   bash install.sh          # 使用当前 Python 环境
#   bash install.sh venv     # 创建新的 venv（推荐，避免系统包冲突）

set -e

if [[ "$1" == "venv" ]]; then
    echo "Creating venv at ~/mmrb2_env ..."
    python3 -m venv ~/mmrb2_env
    source ~/mmrb2_env/bin/activate
    echo "Activated: $(which python)"
fi

echo "Installing dependencies..."

# Per Qwen3.5 official model card: both sglang and transformers must be installed from git main.
# sglang main pins transformers==4.57.1, so we install them in order and override.

# Step 1: sglang from main (brings transformers==4.57.1 as dep)
pip install torch==2.9.1 \
    'sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python' \
    openai json-repair Pillow tqdm datasets huggingface_hub

# Step 2: override transformers with latest main (--no-deps to avoid conflict)
pip install --no-deps 'transformers @ git+https://github.com/huggingface/transformers.git@main'

echo ""
echo "Done. Versions:"
python -c "
import torch, transformers, sglang
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  sglang:       {sglang.__version__}')
"
