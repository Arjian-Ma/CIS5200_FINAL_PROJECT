#!/bin/bash

# Script to train the Temporal Transformer model
# Usage: ./scripts/train_temporal_transformer.sh [options]

# Default parameters
FEATURE_LIST="${FEATURE_LIST:-specified}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
D_MODEL="${D_MODEL:-256}"
NHEAD="${NHEAD:-8}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-512}"
DROPOUT="${DROPOUT:-0.1}"
FORECAST_HORIZON="${FORECAST_HORIZON:-5}"
AUTOREGRESSIVE="${AUTOREGRESSIVE:-true}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-models/temporal_transformer}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-temporal_transformer.pth}"
BEST_CHECKPOINT_NAME="${BEST_CHECKPOINT_NAME:-temporal_transformer_best.pth}"
CURVE_SAVE_PATH="${CURVE_SAVE_PATH:-results/Model_Transformer/training_curves.png}"

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
        --d_model)
            D_MODEL="$2"
            shift 2
            ;;
        --nhead)
            NHEAD="$2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --dim_feedforward)
            DIM_FEEDFORWARD="$2"
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
            echo "  --feature_list PATH           Feature list CSV, 'specified', 'None', or comma-separated list"
            echo "  --num_epochs N                Number of training epochs (default: 50)"
            echo "  --patience N                  Early stopping patience (default: 5)"
            echo "  --batch_size N                Batch size (default: 32)"
            echo "  --sequence_length N           Input sequence length (default: 15)"
            echo "  --learning_rate FLOAT         Learning rate (default: 0.0001)"
            echo "  --d_model N                   Transformer hidden size (default: 256)"
            echo "  --nhead N                     Number of attention heads (default: 8)"
            echo "  --num_layers N                Transformer encoder layers (default: 4)"
            echo "  --dim_feedforward N           Feed-forward dimension (default: 512)"
            echo "  --dropout FLOAT               Dropout rate (default: 0.1)"
            echo "  --forecast_horizon N          Forecast horizon (default: 5)"
            echo "  --autoregressive              Enable autoregressive mode (default)"
            echo "  --no_autoregressive           Disable autoregressive mode"
            echo "  --model_save_dir PATH         Directory to save models"
            echo "  --checkpoint_name NAME        Final checkpoint filename"
            echo "  --best_checkpoint_name NAME   Best checkpoint filename"
            echo "  --curve_save_path PATH        Path to save training curves"
            echo ""
            echo "Examples:"
            echo "  $0 --num_epochs 100 --patience 10"
            echo "  $0 --d_model 384 --nhead 12 --num_layers 6"
            echo "  $0 --forecast_horizon 3 --autoregressive"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

echo "=========================================="
echo "Temporal Transformer Training Script"
echo "=========================================="
echo "Feature List: $FEATURE_LIST"
echo "Epochs: $NUM_EPOCHS"
echo "Patience: $PATIENCE"
echo "Batch Size: $BATCH_SIZE"
echo "Sequence Length: $SEQUENCE_LENGTH"
echo "Learning Rate: $LEARNING_RATE"
echo "d_model: $D_MODEL"
echo "nhead: $NHEAD"
echo "num_layers: $NUM_LAYERS"
echo "dim_feedforward: $DIM_FEEDFORWARD"
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
CMD="python3 src/models/temporal_transformer.py"
CMD="$CMD --feature_list \"$FEATURE_LIST\""
CMD="$CMD --num_epochs $NUM_EPOCHS"
CMD="$CMD --patience $PATIENCE"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --sequence_length $SEQUENCE_LENGTH"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --dropout $DROPOUT"
CMD="$CMD --forecast_horizon $FORECAST_HORIZON"
CMD="$CMD --d_model $D_MODEL"
CMD="$CMD --nhead $NHEAD"
CMD="$CMD --num_layers $NUM_LAYERS"
CMD="$CMD --dim_feedforward $DIM_FEEDFORWARD"

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
    echo "✅ Transformer training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Transformer training failed!"
    echo "=========================================="
    exit 1
fi


