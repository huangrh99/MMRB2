#!/bin/bash
# OpenGVLab/InternVL3_5-241B-A28B-HF — vLLM backend, TP=8
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_vllm.sh OpenGVLab/InternVL3_5-241B-A28B-HF 16 8000 8
