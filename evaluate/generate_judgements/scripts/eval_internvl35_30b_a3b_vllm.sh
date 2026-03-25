#!/bin/bash
# OpenGVLab/InternVL3_5-30B-A3B-HF — vLLM backend, TP=1
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_vllm.sh OpenGVLab/InternVL3_5-30B-A3B-HF 16 8000 1
