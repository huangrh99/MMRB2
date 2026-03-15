#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# MMRB2 Benchmark - Qwen3.5 Evaluation Script
# Runs all tasks sequentially with multi-GPU parallelism
#
# Usage:
#   bash run_qwen35.sh [MODEL_SIZE] [N_GPU]
#
# MODEL_SIZE options: 0.8b, 2b, 4b, 9b (default), 27b, 35b-a3b, 122b-a10b, 397b-a17b
# N_GPU: number of GPUs (default: 1, use higher for 27b+ models)
#
# Examples:
#   bash run_qwen35.sh              # 9B on 1 GPU
#   bash run_qwen35.sh 27b 4        # 27B on 4 GPUs
#   bash run_qwen35.sh 122b-a10b 8  # 122B MoE on 8 GPUs

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Parse arguments
MODEL_SIZE="${1:-9b}"
N_GPU="${2:-1}"

# Map model size to evaluator name
case "$MODEL_SIZE" in
    0.8b)       EVALUATOR="qwen35vl08b-pairwise" ;;
    2b)         EVALUATOR="qwen35vl2b-pairwise" ;;
    4b)         EVALUATOR="qwen35vl4b-pairwise" ;;
    9b)         EVALUATOR="qwen35vl9b-pairwise" ;;
    27b)        EVALUATOR="qwen35vl27b-pairwise" ;;
    35b-a3b)    EVALUATOR="qwen35vl35ba3b-pairwise" ;;
    122b-a10b)  EVALUATOR="qwen35vl122ba10b-pairwise" ;;
    397b-a17b)  EVALUATOR="qwen35vl397ba17b-pairwise" ;;
    *)
        echo "Error: Unknown model size '$MODEL_SIZE'"
        echo "Available: 0.8b, 2b, 4b, 9b, 27b, 35b-a3b, 122b-a10b, 397b-a17b"
        exit 1
        ;;
esac

# Data paths
BASE_DATA_PATH="${MMRB2_DATA_PATH:-$SCRIPT_DIR/../../benchmark}"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "MMRB2 Benchmark - Qwen3.5-${MODEL_SIZE} Pairwise Evaluation"
echo "=========================================="
echo "Evaluator: $EVALUATOR"
echo "Output directory: $OUTPUT_DIR"
echo "Data path: $BASE_DATA_PATH"
echo "Number of GPUs: $N_GPU"
echo ""

# Task 1: Text-to-Image Generation
echo "[Task 1/4] Text-to-Image Generation (image)"
python multi_gpu_evaluate.py \
    --evaluator_name "$EVALUATOR" \
    --task_type image \
    --pairs_path "$BASE_DATA_PATH/t2i.json" \
    --output_path "$OUTPUT_DIR/task1_${EVALUATOR}.json" \
    --n_gpu $N_GPU \
    --n 1
echo "Task 1 complete!"
echo ""

# Task 2: Image Editing
echo "[Task 2/4] Image Editing (edit)"
python multi_gpu_evaluate.py \
    --evaluator_name "$EVALUATOR" \
    --task_type edit \
    --pairs_path "$BASE_DATA_PATH/edit.json" \
    --output_path "$OUTPUT_DIR/task2_${EVALUATOR}.json" \
    --n_gpu $N_GPU \
    --n 1
echo "Task 2 complete!"
echo ""

# Task 3: Interleaved Generation
echo "[Task 3/4] Interleaved Generation (interleaved)"
python multi_gpu_evaluate.py \
    --evaluator_name "$EVALUATOR" \
    --task_type interleaved \
    --pairs_path "$BASE_DATA_PATH/interleaved.json" \
    --output_path "$OUTPUT_DIR/task3_${EVALUATOR}.json" \
    --n_gpu $N_GPU \
    --n 1
echo "Task 3 complete!"
echo ""

# Task 4: Visual Reasoning
echo "[Task 4/4] Visual Reasoning (reasoning)"
python multi_gpu_evaluate.py \
    --evaluator_name "$EVALUATOR" \
    --task_type reasoning \
    --pairs_path "$BASE_DATA_PATH/reasoning.json" \
    --output_path "$OUTPUT_DIR/task4_${EVALUATOR}.json" \
    --n_gpu $N_GPU \
    --n 1
echo "Task 4 complete!"
echo ""

echo "=========================================="
echo "All tasks completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
