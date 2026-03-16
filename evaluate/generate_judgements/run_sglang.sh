#!/bin/bash
# MMRB2 Benchmark - SGLang Backend Evaluation Script
#
# This script starts a SGLang server and runs MMRB2 evaluation through
# the OpenAI-compatible API. Much faster than transformers pipeline.
#
# Usage:
#   bash run_sglang.sh <HF_MODEL_ID> [OPTIONS]
#
# Options (positional):
#   HF_MODEL_ID  HuggingFace model ID (required)
#   N_WORKERS    Number of parallel evaluation workers (default: 8)
#   PORT         SGLang server port (default: 8000)
#   TP           Tensor parallel degree per replica (default: 1)
#   DP           Data parallel degree — number of model replicas (default: 1)
#   THINK        Thinking mode: "think" (default) or "nothink"
#
# GPU usage: TP * DP GPUs total. Each replica uses TP GPUs.
#
# Examples:
#   bash run_sglang.sh Qwen/Qwen3.5-9B                  # 1 GPU, 8 workers
#   bash run_sglang.sh Qwen/Qwen3.5-9B 16               # 1 GPU, 16 workers
#   bash run_sglang.sh Qwen/Qwen3.5-9B 16 8000 1 8      # 8 GPU DP, 16 workers
#   bash run_sglang.sh Qwen/Qwen3.5-27B 8 8000 4 2      # 4-way TP x 2 DP = 8 GPUs
#   bash run_sglang.sh OpenGVLab/InternVL3_5-8B-HF 16 8000 1 8  # InternVL, 8 DP
#   bash run_sglang.sh Qwen/Qwen3.5-9B 16 8000 1 8 nothink  # nothink mode

set -e

# Auto-activate venv
if [ -d ~/mmrb2_env ] && [ -z "$VIRTUAL_ENV" ]; then
    source ~/mmrb2_env/bin/activate
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Parse arguments
HF_MODEL_ID="${1:?Error: HF_MODEL_ID is required. Usage: bash run_sglang.sh <HF_MODEL_ID> [N_WORKERS] [PORT] [TP] [DP] [THINK]}"
N_WORKERS="${2:-8}"
PORT="${3:-8000}"
TP="${4:-1}"
DP="${5:-1}"
THINK="${6:-think}"

TOTAL_GPUS=$((TP * DP))

# Resolve thinking mode
if [ "$THINK" = "nothink" ]; then
    THINK_SUFFIX="_nothink"
    export SGLANG_ENABLE_THINKING="false"
else
    THINK_SUFFIX=""
    export SGLANG_ENABLE_THINKING="true"
fi

# Data paths
BASE_DATA_PATH="${MMRB2_DATA_PATH:-$SCRIPT_DIR/../../benchmark}"
OUTPUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$OUTPUT_DIR"

# Derive a short name for output files
SHORT_NAME=$(echo "$HF_MODEL_ID" | tr '/' '_' | tr '.' '-')

# Log directory
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
SGLANG_LOG="$LOG_DIR/sglang_${SHORT_NAME}_$(date +%Y%m%d_%H%M%S).log"

export SGLANG_BASE_URL="http://localhost:${PORT}/v1"

echo "=========================================="
echo "MMRB2 Benchmark - SGLang Backend"
echo "=========================================="
echo "Model:       $HF_MODEL_ID"
echo "TP:          $TP  (tensor parallel per replica)"
echo "DP:          $DP  (data parallel replicas)"
echo "Total GPUs:  $TOTAL_GPUS"
echo "Workers:     $N_WORKERS"
echo "Thinking:    $THINK"
echo "Server:      http://localhost:${PORT}"
echo "Output:      $OUTPUT_DIR"
echo "Server log:  $SGLANG_LOG"
echo ""

# ----------------------------------------
# Step 1: Start SGLang server
# ----------------------------------------
echo "[Step 1] Starting SGLang server..."

SGLANG_ARGS=(
    --model-path "$HF_MODEL_ID"
    --port "$PORT"
    --tp "$TP"
    --dtype bfloat16
    --mem-fraction-static 0.85
    --reasoning-parser qwen3
    --context-length 262144
)

if [ "$DP" -gt 1 ]; then
    SGLANG_ARGS+=(--dp-size "$DP")
fi

python -m sglang.launch_server "${SGLANG_ARGS[@]}" > "$SGLANG_LOG" 2>&1 &

SGLANG_PID=$!

# Wait for server to be ready
echo "Waiting for server to start (PID: $SGLANG_PID)..."
MAX_WAIT=600
WAITED=0
while ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 $SGLANG_PID 2>/dev/null; then
        echo "Error: SGLang server process died. Check log: $SGLANG_LOG"
        tail -20 "$SGLANG_LOG"
        exit 1
    fi
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "Error: SGLang server did not start within ${MAX_WAIT}s."
        kill $SGLANG_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $((WAITED % 10)) -eq 0 ]; then
        echo "  Waiting... (${WAITED}s)"
    fi
done
echo "Server is ready!"
echo ""

# ----------------------------------------
# Step 2: Run evaluation
# ----------------------------------------
if [ "$THINK" = "nothink" ]; then
    EVALUATOR="sglang-nothink-pairwise"
else
    EVALUATOR="sglang-pairwise"
fi

cleanup() {
    echo ""
    echo "Shutting down SGLang server (PID: $SGLANG_PID)..."
    kill $SGLANG_PID 2>/dev/null || true
    wait $SGLANG_PID 2>/dev/null || true
    echo "Server stopped."
}
trap cleanup EXIT

echo "[Step 2] Running evaluation with $N_WORKERS workers..."
echo ""

for task_info in "1 image t2i" "2 edit edit" "3 interleaved interleaved" "4 reasoning reasoning"; do
    set -- $task_info
    TASK_NUM=$1; TASK_TYPE=$2; DATA_FILE=$3

    echo "[Task ${TASK_NUM}/4] ${TASK_TYPE}"
    python multi_gpu_evaluate.py \
        --evaluator_name "$EVALUATOR" \
        --task_type "$TASK_TYPE" \
        --pairs_path "$BASE_DATA_PATH/${DATA_FILE}.json" \
        --output_path "$OUTPUT_DIR/task${TASK_NUM}_sglang_${SHORT_NAME}${THINK_SUFFIX}.json" \
        --n_gpu "$N_WORKERS" \
        --n 1
    echo "Task ${TASK_NUM} complete!"
    echo ""
done

echo "=========================================="
echo "All tasks completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To compute scores, run:"
echo "  bash scripts/compute_scores.sh $SHORT_NAME $THINK"
echo "=========================================="
