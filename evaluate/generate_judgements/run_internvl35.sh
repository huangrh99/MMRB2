#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# MMRB2 Benchmark - InternVL3.5 Evaluation Script
#
# Usage:
#   bash run_internvl35.sh [MODEL_SIZE] [N_GPU]
#
# MODEL_SIZE options: 1b, 2b, 4b, 8b (default), 14b, 38b
# N_GPU: number of GPUs (default: 1, use higher for 38b or data parallelism)
#
# Examples:
#   bash run_internvl35.sh              # 8B on 1 GPU
#   bash run_internvl35.sh 8b 8         # 8B on 8 GPUs (data parallel)
#   bash run_internvl35.sh 38b 4        # 38B on 4 GPUs (tensor parallel)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Parse arguments
MODEL_SIZE="${1:-8b}"
N_GPU="${2:-1}"

# Map model size to evaluator name
case "$MODEL_SIZE" in
    1b)   EVALUATOR="internvl35-1b-pairwise" ;;
    2b)   EVALUATOR="internvl35-2b-pairwise" ;;
    4b)   EVALUATOR="internvl35-4b-pairwise" ;;
    8b)   EVALUATOR="internvl35-8b-pairwise" ;;
    14b)  EVALUATOR="internvl35-14b-pairwise" ;;
    38b)  EVALUATOR="internvl35-38b-pairwise" ;;
    *)
        echo "Error: Unknown model size '$MODEL_SIZE'"
        echo "Available: 1b, 2b, 4b, 8b, 14b, 38b"
        exit 1
        ;;
esac

# Data paths
BASE_DATA_PATH="${MMRB2_DATA_PATH:-$SCRIPT_DIR/../../benchmark}"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "MMRB2 Benchmark - InternVL3.5-${MODEL_SIZE} Pairwise Evaluation"
echo "=========================================="
echo "Evaluator: $EVALUATOR"
echo "Output directory: $OUTPUT_DIR"
echo "Data path: $BASE_DATA_PATH"
echo "Number of GPUs: $N_GPU"
echo ""

for task_num_type in "1 image t2i" "2 edit edit" "3 interleaved interleaved" "4 reasoning reasoning"; do
    set -- $task_num_type
    TASK_NUM=$1; TASK_TYPE=$2; DATA_FILE=$3

    echo "[Task ${TASK_NUM}/4] ${TASK_TYPE}"
    python multi_gpu_evaluate.py \
        --evaluator_name "$EVALUATOR" \
        --task_type "$TASK_TYPE" \
        --pairs_path "$BASE_DATA_PATH/${DATA_FILE}.json" \
        --output_path "$OUTPUT_DIR/task${TASK_NUM}_${EVALUATOR}.json" \
        --n_gpu $N_GPU \
        --n 1
    echo "Task ${TASK_NUM} complete!"
    echo ""
done

echo "=========================================="
echo "All tasks completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
