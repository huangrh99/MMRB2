#!/bin/bash
# Qwen/Qwen3.5-397B-A17B — vLLM backend, TP=8, nothink
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_vllm.sh Qwen/Qwen3.5-397B-A17B 16 8000 8 nothink
