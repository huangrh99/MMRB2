#!/bin/bash
# Qwen/Qwen3.5-2B — vLLM backend, TP=1
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_vllm.sh Qwen/Qwen3.5-2B 16 8000 1
