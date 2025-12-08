#!/bin/bash
# General script to analyze all test games using any trained model
# Supports LSTM, Autoregressive, and other model types

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Parse arguments
MODEL_PATH=""
MODEL_TYPE="auto"
DATA_PATH="data/processed/featured_data_with_scores.parquet"
FEATURE_LIST="specified"
TARGET_COL="Total_Gold_Difference"
SEQUENCE_LENGTH=15
OUTPUT_PATH=""

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model_path PATH     Path to model checkpoint (required)"
    echo "  -t, --model_type TYPE      Model type: auto, lstm, autoregressive (default: auto)"
    echo "  -d, --data_path PATH       Path to featured data (default: data/processed/featured_data_with_scores.parquet)"
    echo "  -f, --feature_list LIST   Feature list: 'specified', CSV path, or comma-separated (default: specified)"
    echo "  -c, --target_col COL      Target column name (default: Total_Gold_Difference)"
    echo "  -s, --sequence_length N    Sequence length for LSTM (default: 15)"
    echo "  -o, --output_path PATH     Output path for plot (default: auto-generated)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Analyze LSTM model"
    echo "  $0 -m models/lstm_model.pth -t lstm"
    echo ""
    echo "  # Analyze autoregressive model"
    echo "  $0 -m results/autoregressive_hierarchical_model.pth -t autoregressive"
    echo ""
    echo "  # Use custom feature list"
    echo "  $0 -m models/lstm_model.pth -f data/Feature_Selection_RFE/best_features_rfe.csv"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -t|--model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -d|--data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        -f|--feature_list)
            FEATURE_LIST="$2"
            shift 2
            ;;
        -c|--target_col)
            TARGET_COL="$2"
            shift 2
            ;;
        -s|--sequence_length)
            SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        -o|--output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    echo "❌ Error: Model path is required"
    echo ""
    usage
    exit 1
fi

echo "=========================================="
echo "Model Test Games Analysis"
echo "=========================================="
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file not found: $DATA_PATH"
    exit 1
fi

# Auto-generate output path if not provided
if [ -z "$OUTPUT_PATH" ]; then
    MODEL_BASENAME=$(basename "$MODEL_PATH" .pth)
    OUTPUT_PATH="results/${MODEL_BASENAME}_test_rmse_distribution.png"
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

python src/Analyzation\&Visualization/analyze_all_test_games.py \
    --model_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --data_path "$DATA_PATH" \
    --feature_list "$FEATURE_LIST" \
    --target_col "$TARGET_COL" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --output_path "$OUTPUT_PATH"

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

