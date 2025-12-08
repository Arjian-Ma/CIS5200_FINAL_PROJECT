#!/bin/bash

# Script to train the LSTM win probability classifier
# Usage: ./scripts/train_lstm_win_classifier.sh [options]

# Default parameters (can be overridden via environment or CLI flags)
FEATURE_LIST="${FEATURE_LIST:-data/Feature_Selection_RFE/best_features_rfe.csv}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.5}"
BIDIRECTIONAL="${BIDIRECTIONAL:-false}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-50}"
MIN_SEQUENCE_LENGTH="${MIN_SEQUENCE_LENGTH:-5}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-models/lstm_win_classifier_rfe}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-lstm_win_classifier.pth}"
BEST_CHECKPOINT_NAME="${BEST_CHECKPOINT_NAME:-lstm_win_classifier_best.pth}"
CURVE_SAVE_PATH="${CURVE_SAVE_PATH:-results/Model_LSTM_Classifier_rfe/win_classifier_training_curves.png}"
USE_PREFIX_DATA="${USE_PREFIX_DATA:-false}"
MIN_CUTOFF_RATIO="${MIN_CUTOFF_RATIO:-0.5}"
MAX_CUTOFF_RATIO="${MAX_CUTOFF_RATIO:-0.9}"
NUM_PREFIX_SEQUENCES="${NUM_PREFIX_SEQUENCES:-3}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-data/splits/train.parquet}"
VAL_DATA_PATH="${VAL_DATA_PATH:-data/splits/val.parquet}"
TEST_DATA_PATH="${TEST_DATA_PATH:-data/splits/test.parquet}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --feature_list)
            FEATURE_LIST="$2"; shift 2 ;;
        --num_epochs)
            NUM_EPOCHS="$2"; shift 2 ;;
        --patience)
            PATIENCE="$2"; shift 2 ;;
        --batch_size)
            BATCH_SIZE="$2"; shift 2 ;;
        --learning_rate)
            LEARNING_RATE="$2"; shift 2 ;;
        --weight_decay)
            WEIGHT_DECAY="$2"; shift 2 ;;
        --hidden_size)
            HIDDEN_SIZE="$2"; shift 2 ;;
        --num_layers)
            NUM_LAYERS="$2"; shift 2 ;;
        --dropout)
            DROPOUT="$2"; shift 2 ;;
        --bidirectional)
            BIDIRECTIONAL="true"; shift ;;
        --no_bidirectional)
            BIDIRECTIONAL="false"; shift ;;
        --max_sequence_length)
            MAX_SEQUENCE_LENGTH="$2"; shift 2 ;;
        --min_sequence_length)
            MIN_SEQUENCE_LENGTH="$2"; shift 2 ;;
        --use_prefix_data)
            USE_PREFIX_DATA="true"; shift ;;
        --min_cutoff_ratio)
            MIN_CUTOFF_RATIO="$2"; shift 2 ;;
        --max_cutoff_ratio)
            MAX_CUTOFF_RATIO="$2"; shift 2 ;;
        --num_prefix_sequences)
            NUM_PREFIX_SEQUENCES="$2"; shift 2 ;;
        --train_data_path)
            TRAIN_DATA_PATH="$2"; shift 2 ;;
        --val_data_path)
            VAL_DATA_PATH="$2"; shift 2 ;;
        --test_data_path)
            TEST_DATA_PATH="$2"; shift 2 ;;
        --model_save_dir)
            MODEL_SAVE_DIR="$2"; shift 2 ;;
        --checkpoint_name)
            CHECKPOINT_NAME="$2"; shift 2 ;;
        --best_checkpoint_name)
            BEST_CHECKPOINT_NAME="$2"; shift 2 ;;
        --curve_save_path)
            CURVE_SAVE_PATH="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --feature_list PATH           Feature list (CSV path, 'specified', 'None', or comma-separated)"
            echo "  --num_epochs N                Training epochs (default: 30)"
            echo "  --patience N                  Early stopping patience (default: 5)"
            echo "  --batch_size N                Batch size (default: 64)"
            echo "  --learning_rate FLOAT         Learning rate (default: 0.001)"
            echo "  --weight_decay FLOAT          Weight decay (default: 0.0)"
            echo "  --hidden_size N               LSTM hidden size (default: 128)"
            echo "  --num_layers N                LSTM layers (default: 2)"
            echo "  --dropout FLOAT               Dropout rate (default: 0.3)"
            echo "  --bidirectional               Enable bidirectional LSTM"
            echo "  --no_bidirectional            Disable bidirectional LSTM (default)"
            echo "  --max_sequence_length N       Max sequence length (default: 40)"
            echo "  --min_sequence_length N       Min sequence length (default: 5)"
            echo "  --use_prefix_data             Use prefix data mode (multiple sequences per game with random cutoffs)"
            echo "  --min_cutoff_ratio FLOAT      Minimum cutoff ratio for random cutoff (default: 0.5, i.e., 50%% of game)"
            echo "  --max_cutoff_ratio FLOAT      Maximum cutoff ratio for random cutoff (default: 0.9, i.e., 90%% of game)"
            echo "  --num_prefix_sequences N      Number of sequences per game in prefix mode (default: 3)"
            echo "  --train_data_path PATH        Path to train parquet file (default: uses data/splits/train.parquet)"
            echo "  --val_data_path PATH          Path to val parquet file (default: uses data/splits/val.parquet)"
            echo "  --test_data_path PATH         Path to test parquet file (default: uses data/splits/test.parquet)"
            echo "  --model_save_dir PATH         Directory to save checkpoints"
            echo "  --checkpoint_name NAME        Final checkpoint filename"
            echo "  --best_checkpoint_name NAME   Best checkpoint filename"
            echo "  --curve_save_path PATH        Path to save training curves"
            echo ""
            echo "Examples:"
            echo "  $0 --num_epochs 50 --patience 10"
            echo "  $0 --bidirectional --hidden_size 256"
            echo "  $0 --feature_list specified --max_sequence_length 60"
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1 ;;
    esac
done

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

echo "=========================================="
echo "LSTM Win Probability Classifier Training"
echo "=========================================="
echo "Feature List: $FEATURE_LIST"
echo "Epochs: $NUM_EPOCHS"
echo "Patience: $PATIENCE"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Hidden Size: $HIDDEN_SIZE"
echo "Num Layers: $NUM_LAYERS"
echo "Dropout: $DROPOUT"
echo "Bidirectional: $BIDIRECTIONAL"
echo "Use Prefix Data: $USE_PREFIX_DATA"
echo "Min Cutoff Ratio: $MIN_CUTOFF_RATIO"
echo "Max Cutoff Ratio: $MAX_CUTOFF_RATIO"
echo "Num Prefix Sequences: $NUM_PREFIX_SEQUENCES"
echo "Max Sequence Length: $MAX_SEQUENCE_LENGTH"
echo "Min Sequence Length: $MIN_SEQUENCE_LENGTH"
echo "Model Save Dir: $MODEL_SAVE_DIR"
echo "Checkpoint Name: $CHECKPOINT_NAME"
echo "Best Checkpoint Name: $BEST_CHECKPOINT_NAME"
echo "Curve Save Path: $CURVE_SAVE_PATH"
echo "=========================================="
echo ""

# Build command
CMD="python3 src/models/lstm_win_classifier.py"
CMD="$CMD --feature_list \"$FEATURE_LIST\""
CMD="$CMD --num_epochs $NUM_EPOCHS"
CMD="$CMD --patience $PATIENCE"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --weight_decay $WEIGHT_DECAY"
CMD="$CMD --hidden_size $HIDDEN_SIZE"
CMD="$CMD --num_layers $NUM_LAYERS"
CMD="$CMD --dropout $DROPOUT"
CMD="$CMD --max_sequence_length $MAX_SEQUENCE_LENGTH"
CMD="$CMD --min_sequence_length $MIN_SEQUENCE_LENGTH"
CMD="$CMD --min_cutoff_ratio $MIN_CUTOFF_RATIO"
CMD="$CMD --max_cutoff_ratio $MAX_CUTOFF_RATIO"
CMD="$CMD --num_prefix_sequences $NUM_PREFIX_SEQUENCES"
CMD="$CMD --model_save_dir \"$MODEL_SAVE_DIR\""
CMD="$CMD --checkpoint_name \"$CHECKPOINT_NAME\""
CMD="$CMD --best_checkpoint_name \"$BEST_CHECKPOINT_NAME\""
CMD="$CMD --curve_save_path \"$CURVE_SAVE_PATH\""

if [ "$BIDIRECTIONAL" = "true" ]; then
    CMD="$CMD --bidirectional"
fi

if [ "$USE_PREFIX_DATA" = "true" ]; then
    CMD="$CMD --use_prefix_data"
fi

if [ -n "$TRAIN_DATA_PATH" ]; then
    CMD="$CMD --train_data_path \"$TRAIN_DATA_PATH\""
fi

if [ -n "$VAL_DATA_PATH" ]; then
    CMD="$CMD --val_data_path \"$VAL_DATA_PATH\""
fi

if [ -n "$TEST_DATA_PATH" ]; then
    CMD="$CMD --test_data_path \"$TEST_DATA_PATH\""
fi

# Execute command
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Win classifier training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Win classifier training failed!"
    echo "=========================================="
    exit 1
fi


