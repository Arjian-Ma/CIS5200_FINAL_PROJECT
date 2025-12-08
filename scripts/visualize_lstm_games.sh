#!/bin/bash
# Script to visualize LSTM model predictions on test games

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "LSTM Model Visualization"
echo "=========================================="
echo ""

# Configuration
MODEL_PATH="${MODEL_PATH:-models/lstm/lstm_model_best.pth}"
DATA_PATH="${DATA_PATH:-data/splits/test.parquet}"
MODEL_TYPE="${MODEL_TYPE:-lstm}"
FEATURE_LIST="${FEATURE_LIST:-specified }"
TARGET_COL="${TARGET_COL:-Total_Gold_Difference}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-15}"
NUM_GAMES="${NUM_GAMES:-30}"
OUTPUT_PATH="${OUTPUT_PATH:-results/Model_LSTM/lstm_model_visualization_${NUM_GAMES}games.png}"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found: $MODEL_PATH"
    echo "   Please train the model first or specify correct MODEL_PATH"
    exit 1
fi

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file not found: $DATA_PATH"
    echo "   Please run data_featuring_score.py first or specify correct DATA_PATH"
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
echo "  Number of Games:  $NUM_GAMES"
echo "  Output Path:      $OUTPUT_PATH"
echo ""

# Run the visualization script
echo "Running visualization..."
echo ""

python src/Analyzation\&Visualization/visualize_endgame_model.py \
    --model_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --data_path "$DATA_PATH" \
    --feature_list "$FEATURE_LIST" \
    --target_col "$TARGET_COL" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --num_games "$NUM_GAMES" \
    --output_path "$OUTPUT_PATH"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Visualization completed successfully!"
    echo "=========================================="
    echo ""
    echo "Plot saved to: $OUTPUT_PATH"
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
    echo "❌ Visualization failed!"
    echo "=========================================="
    exit 1
fi

