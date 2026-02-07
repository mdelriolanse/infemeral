#!/bin/bash
# Download DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4 to RunPod network volume
# Usage: ./scripts/download_deepseek_r1.sh [output_dir]

MODEL_ID="dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4"
OUTPUT_DIR="${1:-/workspace/weights/deepseek-r1-32b}"

echo "Downloading $MODEL_ID to $OUTPUT_DIR..."

# Ensure directory exists
mkdir -p "$OUTPUT_DIR"

# Download using huggingface-cli (fastest method)
huggingface-cli download "$MODEL_ID" \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False

# Verify download
if [ -f "$OUTPUT_DIR/config.json" ]; then
    echo "Download complete!"
    echo ""
    echo "Files downloaded:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    echo "Total size:"
    du -sh "$OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Extract client weights:"
    echo "     python -m infemeral.model_prep \\"
    echo "         --model-id \"$MODEL_ID\" \\"
    echo "         --output-dir /workspace/weights/deepseek-r1-32b-client \\"
    echo "         --client-only"
    echo ""
    echo "  2. Set environment variables:"
    echo "     export INFEMERAL_SERVER_WEIGHTS_DIR=$OUTPUT_DIR"
    echo "     export INFEMERAL_SERVER_MODEL_ID=$MODEL_ID"
    echo "     export INFEMERAL_SERVER_MAX_CONTEXT_LENGTH=4096  # RTX 4090"
else
    echo "ERROR: Download failed - config.json not found"
    exit 1
fi
