#!/bin/bash
# 计算评测 accuracy scores
#
# Usage:
#   bash scripts/compute_scores.sh <MODEL_SHORT_NAME> [THINK]
#   bash scripts/compute_scores.sh all              # 计算 outputs/ 下所有模型的分数
#
# Examples:
#   bash scripts/compute_scores.sh Qwen_Qwen3-5-9B
#   bash scripts/compute_scores.sh Qwen_Qwen3-5-9B nothink
#   bash scripts/compute_scores.sh OpenGVLab_InternVL3_5-8B-HF
#   bash scripts/compute_scores.sh all

set -e

# Auto-activate venv
if [ -d ~/mmrb2_env ] && [ -z "$VIRTUAL_ENV" ]; then
    source ~/mmrb2_env/bin/activate
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

OUTPUT_DIR="$SCRIPT_DIR/outputs"
SCORES_DIR="$SCRIPT_DIR/outputs/scores"
BENCHMARK_DIR="${MMRB2_DATA_PATH:-$SCRIPT_DIR/../../benchmark}"

mkdir -p "$SCORES_DIR"

compute_one() {
    local MODEL_NAME="$1"
    local THINK="${2:-think}"

    if [ "$THINK" = "nothink" ]; then
        SUFFIX="_nothink"
    else
        SUFFIX=""
    fi

    local PREFIX="$OUTPUT_DIR/task"
    local FILE_SUFFIX="_sglang_${MODEL_NAME}${SUFFIX}.json"
    local SCORE_FILE="$SCORES_DIR/scores_${MODEL_NAME}${SUFFIX}.json"

    # Check if judgement files exist
    if [ ! -f "${PREFIX}1${FILE_SUFFIX}" ]; then
        echo "Skipping ${MODEL_NAME}${SUFFIX}: no judgement files found"
        return
    fi

    echo "=========================================="
    echo "Computing scores: ${MODEL_NAME}${SUFFIX}"
    echo "=========================================="

    python "$SCRIPT_DIR/../compute_scores/compute_accuracy.py" \
        --task all \
        --predictions \
            "${PREFIX}1${FILE_SUFFIX}" \
            "${PREFIX}2${FILE_SUFFIX}" \
            "${PREFIX}3${FILE_SUFFIX}" \
            "${PREFIX}4${FILE_SUFFIX}" \
        --benchmark_dir "$BENCHMARK_DIR" \
        --output "$SCORE_FILE"

    echo "Saved: $SCORE_FILE"
    echo ""
}

if [ "${1:-}" = "all" ]; then
    # Find all unique model names from task1_sglang_*.json files
    for f in "$OUTPUT_DIR"/task1_sglang_*.json; do
        [ -f "$f" ] || continue
        BASE=$(basename "$f" .json)
        # Strip "task1_sglang_" prefix
        MODEL_SUFFIX="${BASE#task1_sglang_}"

        # Check if it's a nothink variant
        if [[ "$MODEL_SUFFIX" == *_nothink ]]; then
            MODEL_NAME="${MODEL_SUFFIX%_nothink}"
            compute_one "$MODEL_NAME" "nothink"
        else
            compute_one "$MODEL_SUFFIX" "think"
        fi
    done
else
    MODEL_NAME="${1:?Error: provide MODEL_SHORT_NAME or 'all'. See script header for usage.}"
    THINK="${2:-think}"
    compute_one "$MODEL_NAME" "$THINK"
fi

echo "=========================================="
echo "All scores saved to: $SCORES_DIR"
echo "=========================================="
