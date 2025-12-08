#!/bin/bash

# Script to train the Multi-Task Temporal Transformer win probability classifier.
# Usage: ./scripts/train_multi_task_transformer_win_classifier.sh [options]

set -e

# Default configuration (override via env vars or CLI args)
FEATURE_LIST="${FEATURE_LIST:-specified}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
DROPOUT="${DROPOUT:-0.4}"
D_MODEL="${D_MODEL:-256}"
NHEAD="${NHEAD:-8}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-512}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-40}"
MIN_SEQUENCE_LENGTH="${MIN_SEQUENCE_LENGTH:-5}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-models/multi_task_transformer_classifier_high_weights_new}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-multi_task_transformer_win_classifier.pth}"
BEST_CHECKPOINT_NAME="${BEST_CHECKPOINT_NAME:-multi_task_transformer_win_classifier_best.pth}"
CURVE_SAVE_PATH="${CURVE_SAVE_PATH:-results/Model_MultiTask_Transformer_high_weights_new/win_classifier_training_curves.png}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-data/splits/reduced_train.parquet}"
VAL_DATA_PATH="${VAL_DATA_PATH:-data/splits/reduced_val.parquet}"
TEST_DATA_PATH="${TEST_DATA_PATH:-data/splits/reduced_test.parquet}"
USE_PREFIX_DATA="${USE_PREFIX_DATA:-false}"
MIN_CUTOFF_RATIO="${MIN_CUTOFF_RATIO:-0.5}"
MAX_CUTOFF_RATIO="${MAX_CUTOFF_RATIO:-0.9}"
NUM_PREFIX_SEQUENCES="${NUM_PREFIX_SEQUENCES:-3}"
LOSS_WEIGHT_WIN="${LOSS_WEIGHT_WIN:-2.0}"
LOSS_WEIGHT_ELITE="${LOSS_WEIGHT_ELITE:-0.5}"
LOSS_WEIGHT_BUILDINGS="${LOSS_WEIGHT_BUILDINGS:-0.5}"
USE_ADAPTIVE_LOSS="${USE_ADAPTIVE_LOSS:-false}"

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
        --dropout)
            DROPOUT="$2"; shift 2 ;;
        --d_model)
            D_MODEL="$2"; shift 2 ;;
        --nhead)
            NHEAD="$2"; shift 2 ;;
        --num_layers)
            NUM_LAYERS="$2"; shift 2 ;;
        --dim_feedforward)
            DIM_FEEDFORWARD="$2"; shift 2 ;;
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
        --loss_weight_win)
            LOSS_WEIGHT_WIN="$2"; shift 2 ;;
        --loss_weight_elite)
            LOSS_WEIGHT_ELITE="$2"; shift 2 ;;
        --loss_weight_buildings)
            LOSS_WEIGHT_BUILDINGS="$2"; shift 2 ;;
        --use_adaptive_loss)
            USE_ADAPTIVE_LOSS="true"; shift ;;
        --model_save_dir)
            MODEL_SAVE_DIR="$2"; shift 2 ;;
        --checkpoint_name)
            CHECKPOINT_NAME="$2"; shift 2 ;;
        --best_checkpoint_name)
            BEST_CHECKPOINT_NAME="$2"; shift 2 ;;
        --curve_save_path)
            CURVE_SAVE_PATH="$2"; shift 2 ;;
        --help|-h)
            cat <<EOF
Usage: $0 [options]

Multi-Task Transformer Training Script
Predicts three binary classification tasks:
  1. Win Probability (Blue wins = 1, Red wins = 0)
  2. Elite_Monster_Killed_Difference (Blue ahead = 1, Red ahead/tied = 0)
  3. Buildings_Taken_Difference (Blue ahead = 1, Red ahead/tied = 0)

Options:
  --feature_list PATH           Feature list (CSV path, "specified", "None", or comma-separated)
  --num_epochs N                Number of training epochs (default: $NUM_EPOCHS)
  --patience N                  Early stopping patience (default: $PATIENCE)
  --batch_size N                Batch size (default: $BATCH_SIZE)
  --learning_rate FLOAT         Learning rate (default: $LEARNING_RATE)
  --weight_decay FLOAT          Weight decay (default: $WEIGHT_DECAY)
  --dropout FLOAT               Dropout rate (default: $DROPOUT)
  --d_model N                   Transformer hidden size (default: $D_MODEL)
  --nhead N                     Number of attention heads (default: $NHEAD)
  --num_layers N                Transformer encoder layers (default: $NUM_LAYERS)
  --dim_feedforward N           Feedforward dimension (default: $DIM_FEEDFORWARD)
  --max_sequence_length N       Maximum sequence length (default: $MAX_SEQUENCE_LENGTH)
  --min_sequence_length N       Minimum sequence length (default: $MIN_SEQUENCE_LENGTH)
  --use_prefix_data             Use prefix data mode (multiple sequences per game with random cutoffs)
  --min_cutoff_ratio FLOAT      Minimum cutoff ratio for random cutoff (default: 0.5, i.e., 50%% of game)
  --max_cutoff_ratio FLOAT      Maximum cutoff ratio for random cutoff (default: 0.9, i.e., 90%% of game)
  --num_prefix_sequences N      Number of sequences per game in prefix mode (default: 3)
  --train_data_path PATH        Path to train parquet file (default: uses data/splits/train.parquet)
  --val_data_path PATH          Path to val parquet file (default: uses data/splits/val.parquet)
  --test_data_path PATH         Path to test parquet file (default: uses data/splits/test.parquet)
  --loss_weight_win FLOAT       Weight for win probability loss (default: $LOSS_WEIGHT_WIN)
                                ⚠ Only used if --use_adaptive_loss is NOT set
  --loss_weight_elite FLOAT     Weight for Elite_Monster loss (default: $LOSS_WEIGHT_ELITE)
                                ⚠ Only used if --use_adaptive_loss is NOT set
  --loss_weight_buildings FLOAT Weight for Buildings loss (default: $LOSS_WEIGHT_BUILDINGS)
                                ⚠ Only used if --use_adaptive_loss is NOT set
  --use_adaptive_loss           Enable adaptive loss weighting (uncertainty-based method).
                                When enabled:
                                  - Model learns log variance parameters for each task automatically
                                  - Weights adapt based on task uncertainty during training
                                  - Manual loss weights (--loss_weight_*) are IGNORED
                                When disabled (default):
                                  - Uses fixed manual weights specified by --loss_weight_* arguments
  --model_save_dir PATH         Directory to save checkpoints (default: $MODEL_SAVE_DIR)
  --checkpoint_name NAME        Final checkpoint filename (default: $CHECKPOINT_NAME)
  --best_checkpoint_name NAME   Best checkpoint filename (default: $BEST_CHECKPOINT_NAME)
  --curve_save_path PATH        Path to save training curves (default: $CURVE_SAVE_PATH)
                                When using --use_adaptive_loss, also saves adaptive_weights.png
                                in the same directory (replaces "training_curves.png" with "adaptive_weights.png")

Loss Weighting Examples:
  # Use adaptive loss (learns weights automatically):
  $0 --use_adaptive_loss
  
  # Use manual fixed weights:
  $0 --loss_weight_win 1.0 --loss_weight_elite 0.3 --loss_weight_buildings 0.3
  
  # ⚠ Cannot use both - if --use_adaptive_loss is set, manual weights are ignored

Other Examples:
  $0 --num_epochs 50 --patience 10
  $0 --feature_list specified --max_sequence_length 60
  FEATURE_LIST=specified BATCH_SIZE=128 $0 --learning_rate 5e-5
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

echo "=========================================="
echo "Multi-Task Temporal Transformer Training"
echo "=========================================="
echo "Tasks:"
echo "  1. Win Probability (binary classification)"
echo "  2. Elite_Monster_Killed_Difference (binary: Blue ahead = 1)"
echo "  3. Buildings_Taken_Difference (binary: Blue ahead = 1)"
echo "=========================================="
echo "Feature List:           $FEATURE_LIST"
echo "Epochs:                 $NUM_EPOCHS"
echo "Patience:               $PATIENCE"
echo "Batch Size:             $BATCH_SIZE"
echo "Learning Rate:          $LEARNING_RATE"
echo "Weight Decay:           $WEIGHT_DECAY"
echo "Dropout:                $DROPOUT"
echo "d_model:                $D_MODEL"
echo "nhead:                  $NHEAD"
echo "num_layers:             $NUM_LAYERS"
echo "dim_feedforward:        $DIM_FEEDFORWARD"
echo "Max Sequence Length:    $MAX_SEQUENCE_LENGTH"
echo "Min Sequence Length:    $MIN_SEQUENCE_LENGTH"
echo "Use Prefix Data:        $USE_PREFIX_DATA"
echo "Min Cutoff Ratio:       $MIN_CUTOFF_RATIO"
echo "Max Cutoff Ratio:       $MAX_CUTOFF_RATIO"
echo "Num Prefix Sequences:   $NUM_PREFIX_SEQUENCES"
if [ "$USE_ADAPTIVE_LOSS" = "true" ]; then
    echo "Loss Weighting:          ADAPTIVE (uncertainty-based, learned during training)"
    echo "  ⚠ Manual loss weights (--loss_weight_*) are IGNORED when --use_adaptive_loss is enabled"
else
    echo "Loss Weighting:          MANUAL (fixed weights)"
    echo "Loss Weight (Win):      $LOSS_WEIGHT_WIN"
    echo "Loss Weight (Elite):    $LOSS_WEIGHT_ELITE"
    echo "Loss Weight (Buildings): $LOSS_WEIGHT_BUILDINGS"
fi
echo "Model Save Dir:         $MODEL_SAVE_DIR"
echo "Checkpoint Name:       $CHECKPOINT_NAME"
echo "Best Checkpoint Name:   $BEST_CHECKPOINT_NAME"
echo "Curve Save Path:        $CURVE_SAVE_PATH"
if [ "$USE_ADAPTIVE_LOSS" = "true" ]; then
    ADAPTIVE_WEIGHTS_PATH="${CURVE_SAVE_PATH/training_curves.png/adaptive_weights.png}"
    echo "Adaptive Weights Plot:  $ADAPTIVE_WEIGHTS_PATH"
    echo "  (Will be generated automatically when using adaptive loss)"
fi
echo "=========================================="
echo ""

CMD="python3 src/models/multi_task_transformer_win_classifier.py"
CMD="$CMD --num_epochs $NUM_EPOCHS"
CMD="$CMD --patience $PATIENCE"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --weight_decay $WEIGHT_DECAY"
CMD="$CMD --dropout $DROPOUT"
CMD="$CMD --d_model $D_MODEL"
CMD="$CMD --nhead $NHEAD"
CMD="$CMD --num_layers $NUM_LAYERS"
CMD="$CMD --dim_feedforward $DIM_FEEDFORWARD"
CMD="$CMD --max_sequence_length $MAX_SEQUENCE_LENGTH"
CMD="$CMD --min_sequence_length $MIN_SEQUENCE_LENGTH"
CMD="$CMD --min_cutoff_ratio $MIN_CUTOFF_RATIO"
CMD="$CMD --max_cutoff_ratio $MAX_CUTOFF_RATIO"
CMD="$CMD --num_prefix_sequences $NUM_PREFIX_SEQUENCES"

# Loss weighting: either adaptive or manual (mutually exclusive)
if [ "$USE_ADAPTIVE_LOSS" = "true" ]; then
    CMD="$CMD --use_adaptive_loss"
    echo "ℹ️  Using ADAPTIVE loss weighting (manual weights will be ignored)"
else
    CMD="$CMD --loss_weight_win $LOSS_WEIGHT_WIN"
    CMD="$CMD --loss_weight_elite $LOSS_WEIGHT_ELITE"
    CMD="$CMD --loss_weight_buildings $LOSS_WEIGHT_BUILDINGS"
    echo "ℹ️  Using MANUAL loss weights: Win=$LOSS_WEIGHT_WIN, Elite=$LOSS_WEIGHT_ELITE, Buildings=$LOSS_WEIGHT_BUILDINGS"
fi

CMD="$CMD --model_save_dir \"$MODEL_SAVE_DIR\""
CMD="$CMD --checkpoint_name \"$CHECKPOINT_NAME\""
CMD="$CMD --best_checkpoint_name \"$BEST_CHECKPOINT_NAME\""
CMD="$CMD --curve_save_path \"$CURVE_SAVE_PATH\""

if [ -n "$FEATURE_LIST" ]; then
    CMD="$CMD --feature_list \"$FEATURE_LIST\""
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

eval $CMD
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Multi-task transformer training completed!"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - Training curves: $CURVE_SAVE_PATH"
    if [ "$USE_ADAPTIVE_LOSS" = "true" ]; then
        ADAPTIVE_WEIGHTS_PATH="${CURVE_SAVE_PATH/training_curves.png/adaptive_weights.png}"
        echo "  - Adaptive weights plot: $ADAPTIVE_WEIGHTS_PATH"
    fi
    echo "  - Best model: $MODEL_SAVE_DIR/$BEST_CHECKPOINT_NAME"
    echo "  - Final model: $MODEL_SAVE_DIR/$CHECKPOINT_NAME"
else
    echo ""
    echo "=========================================="
    echo "❌ Training failed!"
    echo "=========================================="
    exit $STATUS
fi

