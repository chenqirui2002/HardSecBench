#!/bin/bash
################################################################################
# HardSecBench Data Generation - Test Script
# 
# Purpose: Test batch generation for a single CWE with configurable parameters
################################################################################

# Configuration
CWE_ID="CWE-1243"
MAX_SEEDS=30
MAX_ITERATIONS=5
LANGUAGE_PRIORITY="verilog"  # Options: c, verilog, none
MAX_TB_COMPILE_ATTEMPTS=4

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  HardSecBench Test Generation  ${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo -e "${GREEN}CWE:${NC}                    $CWE_ID"
echo -e "${GREEN}Max Seeds:${NC}              $MAX_SEEDS"
echo -e "${GREEN}Max Iterations:${NC}         $MAX_ITERATIONS"
echo -e "${GREEN}Language Priority:${NC}      $LANGUAGE_PRIORITY"
echo -e "${GREEN}TB Compile Attempts:${NC}    $MAX_TB_COMPILE_ATTEMPTS"
echo ""
echo -e "${BLUE}Starting generation...${NC}"
echo ""

# Run the generation
python main.py batch-generate \
    --cwe "$CWE_ID" \
    --max-seeds "$MAX_SEEDS" \
    --max-iterations "$MAX_ITERATIONS" \
    --language-priority "$LANGUAGE_PRIORITY" \
    --max-tb-compile-attempts "$MAX_TB_COMPILE_ATTEMPTS"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Generation completed successfully!${NC}"
else
    echo ""
    echo -e "\033[0;31m✗ Generation failed with errors${NC}"
    exit 1
fi
