#!/bin/bash
# Qwen/Qwen3.5-27B — vLLM backend, TP=8, nothink
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_vllm.sh Qwen/Qwen3.5-27B 16 8000 8 nothink
