#!/bin/bash
# Qwen3.5-35B-A3B (MoE) — 8 GPU DP, SGLang backend
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_sglang.sh Qwen/Qwen3.5-35B-A3B 16 8000 1 8
