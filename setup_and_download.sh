#!/bin/bash
# MMRB2 - Download benchmark data and models
#
# Usage:
#   bash setup_and_download.sh [MODEL_FAMILY] [MODEL_SIZE]
#
# MODEL_FAMILY: qwen3.5 (default), internvl3.5
# MODEL_SIZE:   see below per family, or "all" / "none"
#
# Qwen3.5 sizes:    0.8b, 2b, 4b, 9b (default), 27b, 35b-a3b, 122b-a10b, 397b-a17b
# InternVL3.5 sizes: 1b, 2b, 4b, 8b (default), 14b, 38b
#
# Examples:
#   bash setup_and_download.sh                        # benchmark + Qwen3.5-9B
#   bash setup_and_download.sh qwen3.5 27b            # benchmark + Qwen3.5-27B
#   bash setup_and_download.sh internvl3.5             # benchmark + InternVL3.5-8B
#   bash setup_and_download.sh internvl3.5 38b         # benchmark + InternVL3.5-38B
#   bash setup_and_download.sh none                    # benchmark only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_FAMILY="${1:-qwen3.5}"
MODEL_SIZE="${2:-}"

# Handle "none" as family
if [[ "$MODEL_FAMILY" == "none" ]]; then
    MODEL_SIZE="none"
fi

# Qwen3.5 model map
declare -A QWEN_MAP=(
    ["0.8b"]="Qwen/Qwen3.5-0.8B"
    ["2b"]="Qwen/Qwen3.5-2B"
    ["4b"]="Qwen/Qwen3.5-4B"
    ["9b"]="Qwen/Qwen3.5-9B"
    ["27b"]="Qwen/Qwen3.5-27B"
    ["35b-a3b"]="Qwen/Qwen3.5-35B-A3B"
    ["122b-a10b"]="Qwen/Qwen3.5-122B-A10B"
    ["397b-a17b"]="Qwen/Qwen3.5-397B-A17B"
)
QWEN_SIZES="0.8b 2b 4b 9b 27b 35b-a3b 122b-a10b 397b-a17b"
QWEN_DEFAULT="9b"

# InternVL3.5 model map
declare -A INTERNVL_MAP=(
    ["1b"]="OpenGVLab/InternVL3_5-1B-HF"
    ["2b"]="OpenGVLab/InternVL3_5-2B-HF"
    ["4b"]="OpenGVLab/InternVL3_5-4B-HF"
    ["8b"]="OpenGVLab/InternVL3_5-8B-HF"
    ["14b"]="OpenGVLab/InternVL3_5-14B-HF"
    ["38b"]="OpenGVLab/InternVL3_5-38B-HF"
)
INTERNVL_SIZES="1b 2b 4b 8b 14b 38b"
INTERNVL_DEFAULT="8b"

# Resolve model family, size, and map
case "$MODEL_FAMILY" in
    qwen3.5|qwen)
        MODEL_FAMILY="qwen3.5"
        MODEL_SIZE="${MODEL_SIZE:-$QWEN_DEFAULT}"
        declare -n MAP=QWEN_MAP
        ALL_SIZES="$QWEN_SIZES"
        ;;
    internvl3.5|internvl)
        MODEL_FAMILY="internvl3.5"
        MODEL_SIZE="${MODEL_SIZE:-$INTERNVL_DEFAULT}"
        declare -n MAP=INTERNVL_MAP
        ALL_SIZES="$INTERNVL_SIZES"
        ;;
    none)
        ;;
    *)
        echo "Error: Unknown model family '$MODEL_FAMILY'"
        echo "Available: qwen3.5, internvl3.5, none"
        exit 1
        ;;
esac

# Validate model size
if [[ "$MODEL_SIZE" != "none" && "$MODEL_SIZE" != "all" && -n "${MAP+x}" && -z "${MAP[$MODEL_SIZE]}" ]]; then
    echo "Error: Unknown size '$MODEL_SIZE' for $MODEL_FAMILY"
    echo "Available: $ALL_SIZES, all, none"
    exit 1
fi

echo "=========================================="
echo "MMRB2 Setup"
echo "=========================================="
if [[ "$MODEL_SIZE" == "none" ]]; then
    echo "Model: none (benchmark only)"
else
    echo "Model: $MODEL_FAMILY $MODEL_SIZE"
fi
echo ""

# ----------------------------------------
# Step 1: Install Python dependencies
# ----------------------------------------
echo "[Step 1/3] Installing Python dependencies..."
pip install -r requirements.txt
echo "Done."
echo ""

# ----------------------------------------
# Step 2: Download benchmark data
# ----------------------------------------
echo "[Step 2/3] Downloading benchmark data from HuggingFace..."
echo "  Dataset: rl-research/multimodal-rewardbench-2"
echo "  Subsets: t2i, edit, interleaved, reasoning"
echo ""
cd benchmark
python build_from_hf.py --output-dir .
cd "$SCRIPT_DIR"
echo ""
echo "Benchmark data downloaded."
echo ""

# ----------------------------------------
# Step 3: Download model(s)
# ----------------------------------------
if [[ "$MODEL_SIZE" == "none" ]]; then
    echo "[Step 3/3] Skipping model download."
else
    echo "[Step 3/3] Downloading $MODEL_FAMILY model(s)..."

    download_model() {
        local hf_id="$1"
        echo ""
        echo "  Downloading $hf_id ..."
        huggingface-cli download "$hf_id"
        echo "  $hf_id done."
    }

    if [[ "$MODEL_SIZE" == "all" ]]; then
        for size in $ALL_SIZES; do
            download_model "${MAP[$size]}"
        done
    else
        download_model "${MAP[$MODEL_SIZE]}"
    fi
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Benchmark data: $SCRIPT_DIR/benchmark/"
echo "  - t2i.json, edit.json, interleaved.json, reasoning.json"
echo "  - images/, input_images/"
echo ""
if [[ "$MODEL_SIZE" != "none" ]]; then
    echo "Model cached in: $(python3 -c 'from huggingface_hub import constants; print(constants.HF_HUB_CACHE)' 2>/dev/null || echo '~/.cache/huggingface/hub')"
    echo ""
fi
echo "To run evaluation:"
echo "  cd evaluate/generate_judgements"
echo "  bash run_qwen35.sh [MODEL_SIZE] [N_GPU]"
echo "  bash run_internvl35.sh [MODEL_SIZE] [N_GPU]"
