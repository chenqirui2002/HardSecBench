#!/bin/bash
################################################################################
# HardSecBench Data Generation - Filter Seeds by Testbench Quality
# 
# Purpose: Filter generated seeds based on testbench mutation testing results
#          to ensure high-quality testbenches that can effectively detect bugs
################################################################################

# Configuration
DATA_PATH="../data/batchall_20251218_152111_gemini-3-pro-preview"
MUTATIONS=5                    # Number of mutations to test per seed
MUTATION_THRESHOLD=0.5         # Minimum mutation detection rate (0.0-1.0)
COVERAGE_THRESHOLD=0.8         # Minimum code coverage required (0.0-1.0)

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  HardSecBench Testbench Quality Filter ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Data Path:${NC}              $DATA_PATH"
echo -e "${GREEN}Mutations per Seed:${NC}     $MUTATIONS"
echo -e "${GREEN}Mutation Threshold:${NC}     $MUTATION_THRESHOLD"
echo -e "${GREEN}Coverage Threshold:${NC}     $COVERAGE_THRESHOLD"
echo ""
echo -e "${BLUE}Starting filtering process...${NC}"
echo ""

# Run the filter
python scripts/filter_seeds_by_tb_quality.py "$DATA_PATH" \
    --mutations "$MUTATIONS" \
    --mutation-threshold "$MUTATION_THRESHOLD" \
    --coverage-threshold "$COVERAGE_THRESHOLD"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Filtering completed successfully!${NC}"
else
    echo ""
    echo -e "\033[0;31m✗ Filtering failed with errors${NC}"
    exit 1
fi
