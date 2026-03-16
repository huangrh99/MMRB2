#!/bin/bash
# InternVL3.5-14B — 8 GPU DP, SGLang backend
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_sglang.sh OpenGVLab/InternVL3_5-14B-HF 16 8000 1 8
