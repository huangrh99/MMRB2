#!/bin/bash
# OpenGVLab/InternVL3_5-241B-A28B-HF — SGLang backend, TP=8
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_sglang.sh OpenGVLab/InternVL3_5-241B-A28B-HF 16 8000 8 1
