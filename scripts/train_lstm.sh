#!/bin/bash

# Script to train LSTM model for League of Legends Gold Difference Prediction
# Usage: ./scripts/train_lstm.sh [options]

# Default parameters
FEATURE_LIST="${FEATURE_LIST:-specified}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.3}"
FORECAST_HORIZON="${FORECAST_HORIZON:-5}"
AUTOREGRESSIVE="${AUTOREGRESSIVE:-true}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-models/lstm_specified}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-lstm_model.pth}"
BEST_CHECKPOINT_NAME="${BEST_CHECKPOINT_NAME:-lstm_model_best.pth}"
CURVE_SAVE_PATH="${CURVE_SAVE_PATH:-results/Model_LSTM_specified/training_curves.png}"
USE_PREFIX_DATA="${USE_PREFIX_DATA:-false}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-data/splits/reduced_train.parquet}"
VAL_DATA_PATH="${VAL_DATA_PATH:-data/splits/reduced_val.parquet}"
TEST_DATA_PATH="${TEST_DATA_PATH:-data/splits/reduced_test.parquet}"
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --feature_list)
            FEATURE_LIST="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --sequence_length)
            SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --hidden_size)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --forecast_horizon)
            FORECAST_HORIZON="$2"
            shift 2
            ;;
        --autoregressive)
            AUTOREGRESSIVE="true"
            shift
            ;;
        --no_autoregressive)
            AUTOREGRESSIVE="false"
            shift
            ;;
        --model_save_dir)
            MODEL_SAVE_DIR="$2"
            shift 2
            ;;
        --checkpoint_name)
            CHECKPOINT_NAME="$2"
            shift 2
            ;;
        --best_checkpoint_name)
            BEST_CHECKPOINT_NAME="$2"
            shift 2
            ;;
        --curve_save_path)
            CURVE_SAVE_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --feature_list PATH           Feature list: CSV path, 'specified', 'None', or comma-separated"
            echo "  --num_epochs N                Number of training epochs (default: 50)"
            echo "  --patience N                  Early stopping patience (default: 5)"
            echo "  --batch_size N                Batch size (default: 32)"
            echo "  --sequence_length N           Sequence length (default: 15)"
            echo "  --learning_rate FLOAT         Learning rate (default: 0.001)"
            echo "  --hidden_size N               LSTM hidden size (default: 128)"
            echo "  --num_layers N                Number of LSTM layers (default: 2)"
            echo "  --dropout FLOAT               Dropout rate (default: 0.4)"
            echo "  --forecast_horizon N          Forecast horizon for autoregressive mode (default: 5)"
            echo "  --autoregressive              Enable autoregressive mode (default: enabled)"
            echo "  --no_autoregressive           Disable autoregressive mode (use many-to-one)"
            echo "  --model_save_dir PATH         Directory to save models (default: models/)"
            echo "  --checkpoint_name NAME        Final model checkpoint name (default: lstm_model.pth)"
            echo "  --best_checkpoint_name NAME   Best model checkpoint name (default: lstm_model_best.pth)"
            echo "  --curve_save_path PATH        Path to save training curves (default: models/training_curves.png)"
            echo ""
            echo "Examples:"
            echo "  # Basic training with default settings"
            echo "  $0"
            echo ""
            echo "  # Custom epochs and patience"
            echo "  $0 --num_epochs 100 --patience 10"
            echo ""
            echo "  # Custom architecture"
            echo "  $0 --hidden_size 256 --num_layers 3 --dropout 0.5"
            echo ""
            echo "  # Autoregressive mode with forecast horizon 5 (default)"
            echo "  $0 --forecast_horizon 5 --autoregressive"
            echo ""
            echo "  # Many-to-one mode (disable autoregressive)"
            echo "  $0 --no_autoregressive"
            echo ""
            echo "  # Custom save locations"
            echo "  $0 --model_save_dir models/experiments/exp1 --checkpoint_name exp1_model.pth"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT" || exit 1

echo "=========================================="
echo "LSTM Training Script"
echo "=========================================="
echo "Feature List: $FEATURE_LIST"
echo "Epochs: $NUM_EPOCHS"
echo "Patience: $PATIENCE"
echo "Batch Size: $BATCH_SIZE"
echo "Sequence Length: $SEQUENCE_LENGTH"
echo "Learning Rate: $LEARNING_RATE"
echo "Hidden Size: $HIDDEN_SIZE"
echo "Num Layers: $NUM_LAYERS"
echo "Dropout: $DROPOUT"
echo "Forecast Horizon: $FORECAST_HORIZON"
echo "Autoregressive: $AUTOREGRESSIVE"
if [ -n "$MODEL_SAVE_DIR" ]; then
    echo "Model Save Dir: $MODEL_SAVE_DIR"
fi
echo "Checkpoint Name: $CHECKPOINT_NAME"
echo "Best Checkpoint Name: $BEST_CHECKPOINT_NAME"
if [ -n "$CURVE_SAVE_PATH" ]; then
    echo "Curve Save Path: $CURVE_SAVE_PATH"
fi
echo "=========================================="
echo ""

# Build command
CMD="python3 src/models/lstm_model.py"
CMD="$CMD --feature_list \"$FEATURE_LIST\""
CMD="$CMD --num_epochs $NUM_EPOCHS"
CMD="$CMD --patience $PATIENCE"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --sequence_length $SEQUENCE_LENGTH"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --hidden_size $HIDDEN_SIZE"
CMD="$CMD --num_layers $NUM_LAYERS"
CMD="$CMD --dropout $DROPOUT"
CMD="$CMD --forecast_horizon $FORECAST_HORIZON"

if [ "$AUTOREGRESSIVE" = "true" ]; then
    CMD="$CMD --autoregressive"
else
    CMD="$CMD --no_autoregressive"
fi

if [ -n "$MODEL_SAVE_DIR" ]; then
    CMD="$CMD --model_save_dir \"$MODEL_SAVE_DIR\""
fi

CMD="$CMD --checkpoint_name \"$CHECKPOINT_NAME\""
CMD="$CMD --best_checkpoint_name \"$BEST_CHECKPOINT_NAME\""

if [ -n "$CURVE_SAVE_PATH" ]; then
    CMD="$CMD --curve_save_path \"$CURVE_SAVE_PATH\""
fi

# Execute command
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Training failed!"
    echo "=========================================="
    exit 1
fi

