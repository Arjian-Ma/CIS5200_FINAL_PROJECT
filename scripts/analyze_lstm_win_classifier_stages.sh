#!/bin/bash

# Script to analyze win classifier accuracy at different game stages.
# Usage: ./scripts/analyze_win_classifier_stages.sh [options]

set -e

# Defaults (override via env or CLI)
MODEL_PATH="${MODEL_PATH:-models/lstm_win_classifier/lstm_win_classifier_best.pth}"
MODEL_TYPE="${MODEL_TYPE:-lstm}"
STAGES="${STAGES:-10,15,20,25,30,35,40}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUTPUT_PATH="${OUTPUT_PATH:-results/Model_LSTM_Classifier/30min_stage_metrics.png}"
TEST_DATA_PATH="${TEST_DATA_PATH:-data/splits/reduced_test_30min.parquet}"


while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"; shift 2 ;;
        --model_type)
            MODEL_TYPE="$2"; shift 2 ;;
        --stages)
            STAGES="$2"; shift 2 ;;
        --batch_size)
            BATCH_SIZE="$2"; shift 2 ;;
        --output_path)
            OUTPUT_PATH="$2"; shift 2 ;;
        --test_data_path)
            TEST_DATA_PATH="$2"; shift 2 ;;
        --help|-h)
            cat <<EOF
Usage: $0 [options]

Options:
  --model_path PATH      Path to classifier checkpoint (default: \$MODEL_PATH)
  --model_type TYPE      Backbone type: lstm | transformer (auto if omitted)
  --stages LIST          Comma-separated minute marks (default: \$STAGES)
  --batch_size N         Batch size for evaluation (default: \$BATCH_SIZE)
  --output_path PATH     Optional explicit output path for plots
  --test_data_path PATH  Path to custom test dataset (e.g., data/splits/test_10min.parquet).
                         If provided, extracts time interval from filename and truncates sequences.

Examples:
  $0 --model_path models/win_classifier/lstm_win_classifier_best.pth --model_type transformer
  $0 --model_type transformer --model_path models/win_transformer_classifier/temporal_transformer_win_classifier_best.pth
EOF
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1 ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model checkpoint not found: $MODEL_PATH"
    exit 1
fi

CMD="python3 \"src/Analyzation&Visualization/analyze_win_classifier_stages.py\""
CMD="$CMD --model_path \"$MODEL_PATH\""
if [ -n "$MODEL_TYPE" ]; then
    CMD="$CMD --model_type \"$MODEL_TYPE\""
fi
CMD="$CMD --stages \"$STAGES\""
CMD="$CMD --batch_size $BATCH_SIZE"

if [ -n "$OUTPUT_PATH" ]; then
    CMD="$CMD --output_path \"$OUTPUT_PATH\""
fi

if [ -n "$TEST_DATA_PATH" ]; then
    CMD="$CMD --test_data_path \"$TEST_DATA_PATH\""
fi

echo "Running: $CMD"
eval $CMD

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "✅ Stage analysis completed successfully."
else
    echo "❌ Stage analysis failed."
    exit $STATUS
fi


