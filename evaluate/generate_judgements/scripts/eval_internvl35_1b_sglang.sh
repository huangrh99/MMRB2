#!/bin/bash
# InternVL3.5-1B — 8 GPU DP, SGLang backend
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_sglang.sh OpenGVLab/InternVL3_5-1B-HF 16 8000 1 8
