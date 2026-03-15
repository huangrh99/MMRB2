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

# Step 1: Install torch and other deps first
pip install torch==2.9.1 openai json-repair Pillow tqdm datasets huggingface_hub

# Step 2: Install sglang from transformers-v5 compatible branch
#   PyPI sglang 0.5.9 pins transformers==4.57.1 which conflicts with Qwen3.5 (needs >=5.2)
#   This branch removes the pin and adds Qwen3.5 support
pip install -e 'git+https://github.com/joninco/sglang.git@feat/transformers-v5-qwen35-nvfp4#subdirectory=python&egg=sglang[all]'

# Step 3: Install transformers v5 (after sglang, to override any pinned version)
pip install --no-deps 'transformers @ git+https://github.com/huggingface/transformers.git@main'

echo ""
echo "Done. Versions:"
python -c "
import torch, transformers, sglang
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  sglang:       {sglang.__version__}')
"
