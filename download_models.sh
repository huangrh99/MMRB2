#!/bin/bash
# 下载 MMRB2 评测所需模型
set -e

export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,.byteintl.net,.tiktok-row.net,.tiktok-row.org,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8"

DOWNLOAD_DIR=/opt/tiger/code
MODEL_DIR=/mnt/hdfs/user/huangrunhui/huggingface_models

mkdir -p "$DOWNLOAD_DIR" "$MODEL_DIR"

MODELS=(
    # Qwen3.5 (all sizes)
    Qwen/Qwen3.5-0.8B
    Qwen/Qwen3.5-2B
    Qwen/Qwen3.5-4B
    Qwen/Qwen3.5-9B
    Qwen/Qwen3.5-27B
    Qwen/Qwen3.5-35B-A3B
    Qwen/Qwen3.5-122B-A10B
    Qwen/Qwen3.5-397B-A17B
    # InternVL3.5 -HF (all sizes)
    OpenGVLab/InternVL3_5-1B-HF
    OpenGVLab/InternVL3_5-2B-HF
    OpenGVLab/InternVL3_5-4B-HF
    OpenGVLab/InternVL3_5-8B-HF
    OpenGVLab/InternVL3_5-14B-HF
    OpenGVLab/InternVL3_5-38B-HF
)

cd "$DOWNLOAD_DIR"

for MODEL in "${MODELS[@]}"; do
    SHORT_NAME=$(basename "$MODEL")

    if [ -d "$MODEL_DIR/$SHORT_NAME" ]; then
        echo "=== $SHORT_NAME already exists, skipping ==="
        continue
    fi

    echo "=== Downloading $MODEL ==="
    huggingface-cli download "$MODEL" --local-dir "$SHORT_NAME"
    mv "$SHORT_NAME" "$MODEL_DIR/"
    echo "=== $SHORT_NAME done ==="
    echo ""
done

echo "All models saved to: $MODEL_DIR"
ls -1 "$MODEL_DIR"
