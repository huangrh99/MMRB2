#!/bin/bash
# InternVL3.5-8B — 8 GPU DP, SGLang backend
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_sglang.sh OpenGVLab/InternVL3_5-8B-HF 16 8000 1 8
