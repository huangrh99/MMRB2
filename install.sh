#!/bin/bash
# MMRB2 环境安装脚本
#
# Usage:
#   bash install.sh          # pip 安装
#   bash install.sh uv       # uv 安装（更快）
#   bash install.sh venv     # 创建 venv + pip 安装
#   bash install.sh uv-venv  # 创建 uv venv + uv 安装（推荐）

set -e

MODE="${1:-pip}"

# Setup venv if requested
case "$MODE" in
    venv)
        echo "Creating venv at ~/mmrb2_env ..."
        python3 -m venv ~/mmrb2_env
        source ~/mmrb2_env/bin/activate
        echo "Activated: $(which python)"
        MODE="pip"
        ;;
    uv-venv)
        echo "Creating uv venv at ~/mmrb2_env ..."
        uv venv ~/mmrb2_env
        source ~/mmrb2_env/bin/activate
        echo "Activated: $(which python)"
        MODE="uv"
        ;;
esac

echo "Installing dependencies (${MODE})..."

# Per Qwen3.5 official model card: both sglang and transformers from git main.
# sglang main pins transformers==4.57.1, so install in order then override.

if [[ "$MODE" == "uv" ]]; then
    # uv: use --override to force transformers from main despite sglang's pin
    uv pip install torch==2.9.1 \
        'sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python' \
        openai json-repair Pillow tqdm datasets huggingface_hub
    uv pip install --no-deps 'transformers @ git+https://github.com/huggingface/transformers.git@main'
else
    pip install torch==2.9.1 \
        'sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python' \
        openai json-repair Pillow tqdm datasets huggingface_hub
    pip install --no-deps 'transformers @ git+https://github.com/huggingface/transformers.git@main'
fi

echo ""
echo "Done. Versions:"
python -c "
import torch, transformers, sglang
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  sglang:       {sglang.__version__}')
"
