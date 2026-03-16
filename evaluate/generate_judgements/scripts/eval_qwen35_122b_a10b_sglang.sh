#!/bin/bash
# Qwen3.5-122B-A10B (MoE) — 8 GPU (TP=4, DP=2), SGLang backend
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_sglang.sh Qwen/Qwen3.5-122B-A10B 16 8000 4 2
