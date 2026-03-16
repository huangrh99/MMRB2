#!/bin/bash
# Qwen/Qwen3.5-4B — vLLM backend, TP=1, nothink
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_vllm.sh Qwen/Qwen3.5-4B 16 8000 1 nothink
