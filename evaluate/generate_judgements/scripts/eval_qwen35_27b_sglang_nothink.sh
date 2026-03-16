#!/bin/bash
# Qwen3.5-27B — 8 GPU (TP=2, DP=4), SGLang backend, nothink mode
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_sglang.sh Qwen/Qwen3.5-27B 16 8000 8 1 nothink
