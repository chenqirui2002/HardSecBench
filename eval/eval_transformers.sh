#!/bin/bash
# HardSecBench Evaluation with Local Transformers Models
# This script demonstrates how to use local transformers models for evaluation

# ============ Model Configuration ============
# HuggingFace model name or local path
# Examples:
# - Qwen/Qwen2.5-Coder-7B-Instruct
# - deepseek-ai/deepseek-coder-6.7b-instruct
# - /path/to/local/model
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"

# CUDA device selection (e.g., "0", "1", "0,1" for multiple GPUs)
CUDA_VISIBLE_DEVICES="0"

# ============ Evaluation Configuration ============
DATA_PATH="../data/HardSecBench_Data"
TEMPERATURE=0.2
MAX_ITERATIONS=5
MAX_ATTEMPTS=4
LIMIT=""  # Leave empty for all, or set like: LIMIT="--limit 10"

# Batch size for transformers inference (number of prompts processed together)
# Set to 1 to disable batch inference (process one prompt at a time)
# Larger batch size = faster but more GPU memory and may cause issues
# Recommended: Keep at 1 for stability, increase only if needed
BATCH_SIZE=1

# Language filter: Filter test cases by language
# Options: "c", "verilog", or leave empty for all languages
LANGUAGE_FILTER="verilog"  # Only run Verilog test cases

# Security hint level: 0=none, 1=general, 2=cwe
# Can be a single value (e.g., 0) or a comma-separated list (e.g., 0,1,2)
# Multiple hints will be evaluated sequentially
SECURITY_HINT="0,1,2"

# Generate only mode: true=only generate once without iteration, false=normal iterative mode
# For small models (transformers), set to true to avoid context explosion
GENERATE_ONLY="true"


# ============ Transformers Engine Configuration ============
# Engine type: "transformers" for local models, "openai" for API
export TARGET_ENGINE="transformers"

# Device: "cuda", "cpu", or "auto"
export TARGET_DEVICE="cuda"

# Batch size for transformers inference
export TARGET_BATCH_SIZE=$BATCH_SIZE

# Max tokens to generate for target LLM
export TARGET_MAX_TOKENS=4096

# Quantization options (to save GPU memory)
# Set to "true" to enable, "false" to disable
export TARGET_LOAD_IN_8BIT="false"
export TARGET_LOAD_IN_4BIT="false"

# Data type: "auto", "float16", "bfloat16", "float32"
export TARGET_TORCH_DTYPE="bfloat16"

# ============ Collaborator Configuration ============
# Collaborator still uses OpenAI-compatible API
# Make sure to set these in your .env file or here:
# export COLLABORATOR_API_KEY="your_api_key"
# export COLLABORATOR_BASE_URL="https://api.openai.com/v1"

# ========================================

cd "$(dirname "$0")"

echo "=========================================="
echo "HardSecBench Evaluation with Transformers"
echo "=========================================="
echo "Model: $MODEL"
echo "Engine: $TARGET_ENGINE"
echo "Device: $TARGET_DEVICE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Batch Size (Inference): $TARGET_BATCH_SIZE"
echo "Max Tokens: $TARGET_MAX_TOKENS"
echo "Data Path: $DATA_PATH"
echo "=========================================="
echo ""

# Check if transformers is installed
if ! python -c "import transformers" 2>/dev/null; then
    echo "ERROR: transformers not installed!"
    echo "Please install with: pip install transformers torch accelerate"
    exit 1
fi

# Check CUDA availability if using GPU
if [ "$TARGET_DEVICE" = "cuda" ]; then
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "WARNING: CUDA not available! Falling back to CPU..."
        export TARGET_DEVICE="cpu"
    else
        echo "CUDA is available"
        python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
        python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
        echo ""
    fi
fi

# Run evaluation
# Note: Transformers engine uses sequential execution (workers=1) with batch inference
# BATCH_SIZE controls how many prompts are processed together in each forward pass
# This is different from parallel workers which would load multiple model copies

# Build language filter argument
LANGUAGE_ARG=""
if [ -n "$LANGUAGE_FILTER" ]; then
    LANGUAGE_ARG="--language $LANGUAGE_FILTER"
    echo "Language Filter: $LANGUAGE_FILTER"
fi

# Build generate only argument
GENERATE_ONLY_ARG=""
if [ "$GENERATE_ONLY" = "true" ]; then
    GENERATE_ONLY_ARG="--generate-only"
    echo "Generate Only Mode: Enabled (no iteration, single generation)"
fi

# Parse SECURITY_HINT list (comma-separated)
IFS=',' read -ra HINT_ARRAY <<< "$SECURITY_HINT"

# Loop through each hint level and run evaluation sequentially
for hint in "${HINT_ARRAY[@]}"; do
    hint=$(echo "$hint" | xargs)  # Trim whitespace
    echo "=========================================="
    echo "Starting evaluation with SECURITY_HINT=$hint"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python main.py \
        --model "$MODEL" \
        --data-path "$DATA_PATH" \
        --batchall \
        --workers 1 \
        --temperature "$TEMPERATURE" \
        --max-iterations "$MAX_ITERATIONS" \
        --max-attempts "$MAX_ATTEMPTS" \
        --security-hint "$hint" \
        --output-dir outputs_trans \
        $LANGUAGE_ARG \
        $GENERATE_ONLY_ARG \
        $LIMIT
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "=========================================="
        echo "Evaluation with SECURITY_HINT=$hint failed with exit code $EXIT_CODE"
        echo "=========================================="
        exit $EXIT_CODE
    fi
    
    echo "=========================================="
    echo "Evaluation with SECURITY_HINT=$hint completed successfully"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All evaluations completed successfully"
echo "=========================================="