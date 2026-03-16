#!/bin/bash
# Qwen3.5-0.8B — 8 GPU DP, SGLang backend, nothink mode
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_sglang.sh Qwen/Qwen3.5-0.8B 16 8000 1 8 nothink
