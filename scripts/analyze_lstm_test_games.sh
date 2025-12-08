#!/bin/bash
# Script to analyze all test games using the trained LSTM model
# This script runs comprehensive RMSE distribution analysis

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"
echo "=========================================="
echo "LSTM Model Test Games Analysis"
echo "=========================================="
echo ""

# Configuration defaults (override via env or CLI flags)
MODEL_PATH="${MODEL_PATH:-models/lstm_specified/lstm_model_best.pth}"
DATA_PATH="${DATA_PATH:-data/splits/test.parquet}"          # default to prepared test split
MODEL_TYPE="${MODEL_TYPE:-lstm}"
FEATURE_LIST="${FEATURE_LIST:-specified}"                   # options: specified | none | CSV path | comma list
TARGET_COL="${TARGET_COL:-Total_Gold_Difference}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"
OUTPUT_PATH="${OUTPUT_PATH:-results/Model_LSTM_specified/lstm_model_test_rmse_distribution.png}"
MIN_GAME_LENGTH="${MIN_GAME_LENGTH:-10}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --data_path) DATA_PATH="$2"; shift 2 ;;
        --model_type) MODEL_TYPE="$2"; shift 2 ;;
        --feature_list) FEATURE_LIST="$2"; shift 2 ;;
        --target_col) TARGET_COL="$2"; shift 2 ;;
        --sequence_length) SEQUENCE_LENGTH="$2"; shift 2 ;;
        --output_path) OUTPUT_PATH="$2"; shift 2 ;;
        --min_game_length) MIN_GAME_LENGTH="$2"; shift 2 ;;
        --help|-h)
            cat <<EOF
Usage: $0 [options]

Options:
  --model_path PATH       Path to checkpoint (default: \$MODEL_PATH)
  --data_path PATH        Data parquet/CSV to analyze (default: \$DATA_PATH)
  --model_type TYPE       Model type flag passed to analyzer (default: \$MODEL_TYPE)
  --feature_list VALUE    Feature selection:
                            specified (default) → use get_specified_features()
                            none                → defer to checkpoint feature list
                            path.csv            → load features from CSV
                            f1,f2,f3            → explicit comma-separated list
  --target_col NAME       Target column (default: \$TARGET_COL)
  --sequence_length N     Sequence length (default: \$SEQUENCE_LENGTH)
  --output_path PATH      Output plot path (default: \$OUTPUT_PATH)
  --min_game_length N     Minimum frames per game (default: \$MIN_GAME_LENGTH)
EOF
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done
# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found: $MODEL_PATH"
    echo "   Please train the model first or specify correct MODEL_PATH"
    exit 1
fi

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file not found: $DATA_PATH"
    echo "   Please run scripts/create_data_splits.sh first to create data splits"
    echo "   Or specify correct DATA_PATH environment variable (e.g., full dataset path)"
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "Configuration:"
echo "  Model Path:      $MODEL_PATH"
echo "  Data Path:        $DATA_PATH"
echo "  Model Type:       $MODEL_TYPE"
echo "  Feature List:     $FEATURE_LIST"
echo "  Target Column:    $TARGET_COL"
echo "  Sequence Length:  $SEQUENCE_LENGTH"
echo "  Output Path:      $OUTPUT_PATH"
echo ""

# Run the analysis script
echo "Running analysis..."
echo ""

# Handle None feature_list (convert to empty string or use --feature_list None)
FEATURE_ARG=()
shopt -s nocasematch
if [[ "$FEATURE_LIST" != "none" ]]; then
    FEATURE_ARG=(--feature_list "$FEATURE_LIST")
fi
shopt -u nocasematch

python3 "src/Analyzation&Visualization/analyze_all_test_games.py" \
    --model_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --data_path "$DATA_PATH" \
    "${FEATURE_ARG[@]}" \
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
    echo "To view the plot, run:"
    echo "  open $OUTPUT_PATH"
else
    echo ""
    echo "=========================================="
    echo "❌ Analysis failed!"
    echo "=========================================="
    exit 1
fi

