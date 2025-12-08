#!/bin/bash
# General script to visualize games by performance category
# Supports LSTM, Autoregressive, and other model types

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Parse arguments
MODEL_PATH="models/lstm/lstm_model_best.pth"
MODEL_TYPE="lstm"
DATA_PATH="data/splits/test.parquet"
FEATURE_LIST="specified"
TARGET_COL="Total_Gold_Difference"
SEQUENCE_LENGTH=15
GAMES_PER_CATEGORY=20
OUTPUT_DIR="results/Model_LSTM"
FORECAST_HORIZON=""

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model_path PATH     Path to model checkpoint (required)"
    echo "  -t, --model_type TYPE     Model type: auto, lstm, autoregressive (default: auto)"
    echo "  -d, --data_path PATH      Path to featured data (default: data/processed/featured_data_with_scores.parquet)"
    echo "  -f, --feature_list LIST   Feature list: 'specified', CSV path, or comma-separated (default: specified)"
    echo "  -c, --target_col COL      Target column name (default: Total_Gold_Difference)"
    echo "  -s, --sequence_length N   Sequence length for LSTM (default: 15)"
    echo "  -n, --games_per_category N Number of games per category (default: 20)"
    echo "  -h, --forecast_horizon N Forecast horizon for autoregressive models (default: from checkpoint)"
    echo "  -o, --output_dir PATH     Output directory for plots (default: auto-generated)"
    echo "  --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Visualize LSTM model by performance"
    echo "  $0 -m models/lstm_model.pth -t lstm"
    echo ""
    echo "  # Visualize autoregressive model"
    echo "  $0 -m results/autoregressive_hierarchical_model.pth -t autoregressive"
    echo ""
    echo "  # Custom number of games per category"
    echo "  $0 -m models/lstm_model.pth -t lstm -n 15"
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
        -n|--games_per_category)
            GAMES_PER_CATEGORY="$2"
            shift 2
            ;;
        -h|--forecast_horizon)
            FORECAST_HORIZON="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
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
echo "Visualize Games by Performance Category"
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

echo "Configuration:"
echo "  Model Path:        $MODEL_PATH"
echo "  Data Path:         $DATA_PATH"
echo "  Model Type:        $MODEL_TYPE"
echo "  Feature List:      $FEATURE_LIST"
echo "  Target Column:     $TARGET_COL"
echo "  Sequence Length:   $SEQUENCE_LENGTH"
echo "  Games per Category: $GAMES_PER_CATEGORY"
if [ -n "$FORECAST_HORIZON" ]; then
    echo "  Forecast Horizon:  $FORECAST_HORIZON"
fi
if [ -n "$OUTPUT_DIR" ]; then
    echo "  Output Directory:  $OUTPUT_DIR"
fi
echo ""

# Run the visualization script
echo "Running visualization..."
echo ""

# Build command
CMD="python src/Analyzation\&Visualization/visualize_by_performance.py \
    --model_path \"$MODEL_PATH\" \
    --model_type \"$MODEL_TYPE\" \
    --data_path \"$DATA_PATH\" \
    --feature_list \"$FEATURE_LIST\" \
    --target_col \"$TARGET_COL\" \
    --sequence_length $SEQUENCE_LENGTH \
    --games_per_category $GAMES_PER_CATEGORY"

if [ -n "$FORECAST_HORIZON" ]; then
    CMD="$CMD --forecast_horizon $FORECAST_HORIZON"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir \"$OUTPUT_DIR\""
fi

eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Visualization completed successfully!"
    echo "=========================================="
    echo ""
    echo "Plots saved to output directory (see above for path)"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Visualization failed!"
    echo "=========================================="
    exit 1
fi

