#!/bin/bash

# ============ Configuration ============
MODEL="mimo-v2.5"
DATA_PATH="../data/HardSecBench_Data"
TEMPERATURE=0.6
MAX_ITERATIONS=5
MAX_ATTEMPTS=3
MAX_TIMEOUT=1200
LIMIT=""  # Leave empty for all, or set like: LIMIT="--limit 10"

# Security hint level: 0=none, 1=general, 2=cwe
# Can be a single value (e.g., 0) or a comma-separated list (e.g., 0,1,2)
# Multiple hints will be evaluated sequentially
SECURITY_HINT="0"

# ============ Pass@k Configuration ============
# Pass@k: Number of independent runs per test case
# Set to 1 for standard evaluation, >1 for Pass@k evaluation
PASS_AT_K=1

# Code deduplication: Enable to avoid redundant evaluation of identical code
# Recommended: true for Pass@k > 1, false for Pass@k = 1
DEDUPLICATE_CODE=false

# ============ Worker Configuration ============
# For Pass@k = 1: Use WORKERS for both generation and evaluation
# For Pass@k > 1: Use separate GEN_WORKERS and EVAL_WORKERS for better control
WORKERS=10           # Used when Pass@k = 1
GEN_WORKERS=30       # Used when Pass@k > 1 (code generation phase)
EVAL_WORKERS=30      # Used when Pass@k > 1 (evaluation phase)

# ============ API Configuration ============
# Optional: Override .env settings
API_KEY=""
BASE_URL=""

# ========================================

cd "$(dirname "$0")"

# Build optional arguments
OPTIONAL_ARGS=""
if [ -n "$API_KEY" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --api-key $API_KEY"
fi
if [ -n "$BASE_URL" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --base-url $BASE_URL"
fi

# Add Pass@k arguments
if [ "$PASS_AT_K" -gt 1 ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --pass-at-k $PASS_AT_K"
    
    # Enable code deduplication for Pass@k > 1 if configured
    if [ "$DEDUPLICATE_CODE" = "true" ]; then
        OPTIONAL_ARGS="$OPTIONAL_ARGS --deduplicate-code"
    fi
    
    # Use separate workers for generation and evaluation
    WORKER_ARGS="--gen-workers $GEN_WORKERS --eval-workers $EVAL_WORKERS"
else
    # Use single worker configuration for Pass@1
    WORKER_ARGS="--workers $WORKERS"
fi

# Parse SECURITY_HINT list (comma-separated)
IFS=',' read -ra HINT_ARRAY <<< "$SECURITY_HINT"

echo "=========================================="
echo "HardSecBench Evaluation"
echo "=========================================="
echo "Model: $MODEL"
echo "Data Path: $DATA_PATH"
echo "Pass@k: $PASS_AT_K"
if [ "$PASS_AT_K" -gt 1 ]; then
    echo "Code Deduplication: $DEDUPLICATE_CODE"
    echo "Generation Workers: $GEN_WORKERS"
    echo "Evaluation Workers: $EVAL_WORKERS"
else
    echo "Workers: $WORKERS"
fi
echo "Security Hints: $SECURITY_HINT"
echo "=========================================="
echo ""

# Loop through each hint level and run evaluation sequentially
for hint in "${HINT_ARRAY[@]}"; do
    hint=$(echo "$hint" | xargs)  # Trim whitespace
    echo "=========================================="
    echo "Starting evaluation with SECURITY_HINT=$hint"
    echo "=========================================="
    
    python main.py \
        --model "$MODEL" \
        --data-path "$DATA_PATH" \
        --batchall \
        $WORKER_ARGS \
        --temperature "$TEMPERATURE" \
        --max-iterations "$MAX_ITERATIONS" \
        --max-attempts "$MAX_ATTEMPTS" \
        --max-timeout "$MAX_TIMEOUT" \
        --security-hint "$hint" \
        $OPTIONAL_ARGS \
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
