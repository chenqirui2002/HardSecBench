#!/bin/bash

# ============ Configuration ============
MAX_WORKERS=10
MAX_SEEDS_PER_CWE=50
MAX_ITERATIONS=5
MAX_TB_COMPILE_ATTEMPTS=5
LANGUAGE_PRIORITY="none"   # Options: c, verilog, none
# ======================================

cd "$(dirname "$0")"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WRAPPER_LOG="outputs/logs/batchall_wrapper_${TIMESTAMP}.log"

echo "=========================================="
echo "HardSecBench Batchall Generation"
echo "=========================================="
echo "Max Workers: $MAX_WORKERS"
echo "Max Seeds per CWE: $MAX_SEEDS_PER_CWE"
echo "Max Iterations: $MAX_ITERATIONS"
echo "TB Compile Attempts: $MAX_TB_COMPILE_ATTEMPTS"
echo "Language Priority: $LANGUAGE_PRIORITY"
echo "Wrapper Log: $WRAPPER_LOG"
echo "=========================================="
echo ""
echo "Starting batch generation for all CWEs"
echo ""

python main.py batchall \
    --max-workers "$MAX_WORKERS" \
    --max-seeds-per-cwe "$MAX_SEEDS_PER_CWE" \
    --max-iterations "$MAX_ITERATIONS" \
    --max-tb-compile-attempts "$MAX_TB_COMPILE_ATTEMPTS" \
    --language-priority "$LANGUAGE_PRIORITY" \
    2>&1 | tee "$WRAPPER_LOG"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "=========================================="
    echo "Batch generation failed with exit code $EXIT_CODE"
    echo "Wrapper Log: $WRAPPER_LOG"
    echo "Recent output:"
    tail -n 40 "$WRAPPER_LOG"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo "=========================================="
echo "Batch generation completed successfully"
echo "=========================================="

SUMMARY=$(sed -n '/Batchall Generation Complete/,$p' "$WRAPPER_LOG")
if [ -n "$SUMMARY" ]; then
    echo "$SUMMARY"
else
    echo "Summary block not found in wrapper log."
    echo "Wrapper Log: $WRAPPER_LOG"
fi
