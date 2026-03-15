#!/bin/bash
# Qwen3.5-27B — 8 GPU (TP=2, DP=4), SGLang backend
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_sglang.sh Qwen/Qwen3.5-27B 16 8000 2 4
