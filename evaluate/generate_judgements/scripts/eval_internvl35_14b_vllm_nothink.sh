#!/bin/bash
# OpenGVLab/InternVL3_5-14B-HF — vLLM backend, TP=1, nothink
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_vllm.sh OpenGVLab/InternVL3_5-14B-HF 16 8000 1 nothink
