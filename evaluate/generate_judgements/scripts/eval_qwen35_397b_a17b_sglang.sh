#!/bin/bash
# Qwen3.5-397B-A17B (MoE) — 8 GPU (TP=8, DP=1), SGLang backend
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_sglang.sh Qwen/Qwen3.5-397B-A17B 16 8000 8 1
