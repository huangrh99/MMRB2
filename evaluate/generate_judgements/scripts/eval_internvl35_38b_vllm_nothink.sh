#!/bin/bash
# OpenGVLab/InternVL3_5-38B-HF — vLLM backend, TP=8, nothink
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_vllm.sh OpenGVLab/InternVL3_5-38B-HF 16 8000 8 nothink
