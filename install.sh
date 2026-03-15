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

pip install torch==2.9.1 'transformers>=5.2.0' 'sglang[all]>=0.5.9' openai json-repair Pillow tqdm datasets huggingface_hub

echo ""
echo "Done. Versions:"
python -c "
import torch, transformers, sglang
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  sglang:       {sglang.__version__}')
"
