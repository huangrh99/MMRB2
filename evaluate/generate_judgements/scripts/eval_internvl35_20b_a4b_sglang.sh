#!/bin/bash
# OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF — SGLang backend, TP=1
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_sglang.sh OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF 16 8000 1 1
