#!/bin/bash
# Script to create train/validation/test data splits for LoL dataset
# Uses the existing dataloader.py script which already has argparse

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Default parameters (matching dataloader.py defaults)
INPUT_FILE="${INPUT_FILE:-data/processed/featured_data.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-data/splits}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.15}"
TEST_RATIO="${TEST_RATIO:-0.15}"
RANDOM_STATE="${RANDOM_STATE:-42}"

echo "=========================================="
echo "Creating Data Splits for LoL Dataset"
echo "=========================================="
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file not found: $INPUT_FILE"
    echo ""
    echo "Please make sure you have run:"
    echo "  - data_featuring.py (to create featured_data.csv)"
    echo "  - data_featuring_score.py (to create featured_data_with_scores.parquet)"
    echo ""
    echo "Or specify a different INPUT_FILE environment variable"
    exit 1
fi

echo "Configuration (defaults from dataloader.py):"
echo "  Input File:     $INPUT_FILE"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  Train Ratio:    $TRAIN_RATIO"
echo "  Val Ratio:      $VAL_RATIO"
echo "  Test Ratio:     $TEST_RATIO"
echo "  Random State:   $RANDOM_STATE"
echo ""
echo "To customize, set environment variables:"
echo "  INPUT_FILE=... OUTPUT_DIR=... TRAIN_RATIO=... etc."
echo ""

# Run the existing dataloader.py script with its argparse
python src/data/dataloader.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --train_ratio "$TRAIN_RATIO" \
    --val_ratio "$VAL_RATIO" \
    --test_ratio "$TEST_RATIO" \
    --random_state "$RANDOM_STATE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Data splits created successfully!"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - $OUTPUT_DIR/train.parquet"
    echo "  - $OUTPUT_DIR/val.parquet"
    echo "  - $OUTPUT_DIR/test.parquet"
    echo ""
    echo "These splits are ready for training!"
else
    echo ""
    echo "=========================================="
    echo "❌ Data split creation failed!"
    echo "=========================================="
    exit 1
fi

