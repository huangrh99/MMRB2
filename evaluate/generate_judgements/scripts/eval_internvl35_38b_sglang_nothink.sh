#!/bin/bash
# InternVL3.5-38B — 8 GPU (TP=2, DP=4), SGLang backend, nothink mode
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_sglang.sh OpenGVLab/InternVL3_5-38B-HF 16 8000 8 1 nothink
