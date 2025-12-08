#!/bin/bash
# Script to analyze all test games using the temporal transformer model
# Mirrors analyze_lstm_test_games.sh but targets transformer checkpoints

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Temporal Transformer Test Games Analysis"
echo "=========================================="
echo ""

# Configuration
MODEL_PATH="${MODEL_PATH:-models/temporal_transformer/temporal_transformer_best.pth}"
DATA_PATH="${DATA_PATH:-data/splits/test.parquet}"
MODEL_TYPE="${MODEL_TYPE:-temporal_transformer}"
FEATURE_LIST="${FEATURE_LIST:-None}"  # Use None to rely on checkpoint metadata
TARGET_COL="${TARGET_COL:-Total_Gold_Difference}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"
OUTPUT_PATH="${OUTPUT_PATH:-results/Model_Transformer/temporal_transformer_test_rmse_distribution.png}"
MIN_GAME_LENGTH="${MIN_GAME_LENGTH:-10}"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found: $MODEL_PATH"
    echo "   Please train the transformer first or specify a correct MODEL_PATH"
    exit 1
fi

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file not found: $DATA_PATH"
    echo "   Please run scripts/create_data_splits.sh first to create data splits"
    echo "   Or specify a correct DATA_PATH environment variable (e.g., full dataset path)"
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "Configuration:"
echo "  Model Path:      $MODEL_PATH"
echo "  Data Path:       $DATA_PATH"
echo "  Model Type:      $MODEL_TYPE"
echo "  Feature List:    $FEATURE_LIST"
echo "  Target Column:   $TARGET_COL"
echo "  Sequence Length: $SEQUENCE_LENGTH"
echo "  Output Path:     $OUTPUT_PATH"
echo ""

# Run the analysis script
echo "Running analysis..."
echo ""

# Handle None feature_list (skip argument) otherwise forward value
FEATURE_ARG=""
if [ "$FEATURE_LIST" != "None" ] && [ "$FEATURE_LIST" != "none" ]; then
    FEATURE_ARG="--feature_list $FEATURE_LIST"
fi

python src/Analyzation\&Visualization/analyze_all_test_games.py \
    --model_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --data_path "$DATA_PATH" \
    $FEATURE_ARG \
    --target_col "$TARGET_COL" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --output_path "$OUTPUT_PATH" \
    --min_game_length "$MIN_GAME_LENGTH"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Analysis completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $OUTPUT_PATH"
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "To view the plot, run:"
        echo "  open $OUTPUT_PATH"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "To view the plot, run:"
        echo "  xdg-open $OUTPUT_PATH"
    fi
else
    echo ""
    echo "=========================================="
    echo "❌ Analysis failed!"
    echo "=========================================="
    exit 1
fi


