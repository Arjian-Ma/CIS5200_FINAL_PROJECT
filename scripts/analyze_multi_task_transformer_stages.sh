#!/bin/bash

# Script to analyze multi-task transformer win classifier accuracy at different game stages.
# Usage: ./scripts/analyze_multi_task_transformer_stages.sh [options]

set -e

# Defaults (override via env or CLI)
MODEL_PATH="${MODEL_PATH:-models/multi_task_transformer_classifier_high_weights_new/multi_task_transformer_win_classifier_best.pth}"
MODEL_TYPE="${MODEL_TYPE:-multi_task_transformer}"
STAGES="${STAGES:-10,15,20,25,30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
TEST_DATA_PATH="${TEST_DATA_PATH:-}"s
TIME_INTERVALS="${TIME_INTERVALS:-10-30}"
TEST_DATA_PREFIX="${TEST_DATA_PREFIX:-data/splits/reduced_test}"

# Function to parse time intervals from range or list
parse_time_intervals() {
    local input="$1"
    local intervals=()
    
    # Check if it's a range (e.g., "10-35")
    if [[ "$input" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        local start="${BASH_REMATCH[1]}"
        local end="${BASH_REMATCH[2]}"
        local step=5  # Default step of 5 minutes
        
        # Generate intervals
        for ((i=start; i<=end; i+=step)); do
            intervals+=("$i")
        done
    # Check if it's a comma-separated list
    elif [[ "$input" =~ ^[0-9,]+$ ]]; then
        IFS=',' read -ra intervals <<< "$input"
    else
        echo "❌ Invalid time intervals format: $input"
        echo "   Use range format (e.g., '10-35') or comma-separated list (e.g., '10,15,20,25,30,35')"
        exit 1
    fi
    
    echo "${intervals[@]}"
}

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
        --time_intervals)
            TIME_INTERVALS="$2"; shift 2 ;;
        --test_data_prefix)
            TEST_DATA_PREFIX="$2"; shift 2 ;;
        --help|-h)
            cat <<EOF
Usage: $0 [options]

Analyze Multi-Task Transformer Win Classifier
This script analyzes the win prediction task of a multi-task transformer model
that predicts three binary classification tasks:
  1. Win Probability (Blue wins = 1, Red wins = 0)
  2. Elite_Monster_Killed_Difference (Blue ahead = 1, Red ahead/tied = 0)
  3. Buildings_Taken_Difference (Blue ahead = 1, Red ahead/tied = 0)

Note: The analysis focuses on the win prediction task (primary task).

Options:
  --model_path PATH         Path to multi-task transformer checkpoint (default: \$MODEL_PATH)
  --model_type TYPE         Backbone type: multi_task_transformer (auto-detected if omitted)
  --stages LIST             Comma-separated minute marks (default: \$STAGES)
                            (Used only if --time_intervals is not provided)
  --batch_size N            Batch size for evaluation (default: \$BATCH_SIZE)
  --output_path PATH        Optional explicit output path for plots
  --test_data_path PATH     Path to single test dataset (e.g., data/splits/reduced_test_10min.parquet).
                            If provided, extracts time interval from filename and truncates sequences.
  --time_intervals RANGE    Time intervals to analyze. Can be:
                            - Range format: "10-35" (generates 10,15,20,25,30,35)
                            - List format: "10,15,20,25,30,35"
                            Auto-generates test data paths: \${TEST_DATA_PREFIX}_\${interval}min.parquet
  --test_data_prefix PATH   Prefix for auto-generated test data paths (default: data/splits/reduced_test)

Examples:
  # Analyze multiple intervals using range (10,15,20,25,30) - DEFAULT
  $0

  # Analyze specific intervals
  $0 --time_intervals "10,15,20,25,30,35"

  # Analyze single time interval
  $0 --test_data_path data/splits/reduced_test_10min.parquet

  # Custom model path and intervals
  $0 --model_path models/multi_task_transformer_classifier_rfe/multi_task_transformer_win_classifier_best.pth --time_intervals "10-30"
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

# If TEST_DATA_PATH looks like a time interval range/list (and is not a valid file), treat it as TIME_INTERVALS
if [ -n "$TEST_DATA_PATH" ] && [[ "$TEST_DATA_PATH" =~ ^[0-9,-]+$ ]]; then
    # Check if it's actually a file path that exists
    if [ ! -f "$TEST_DATA_PATH" ]; then
        # It's not a file, so treat as time interval range/list
        if [ -z "$TIME_INTERVALS" ]; then
            echo "ℹ️  TEST_DATA_PATH looks like a time interval range/list (not a file), using as TIME_INTERVALS"
            TIME_INTERVALS="$TEST_DATA_PATH"
            TEST_DATA_PATH=""
        else
            echo "⚠ Warning: Both TIME_INTERVALS and TEST_DATA_PATH (range format) are set. Using TIME_INTERVALS."
            TEST_DATA_PATH=""
        fi
    fi
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model checkpoint not found: $MODEL_PATH"
    echo "   Please train the multi-task transformer first or specify correct MODEL_PATH"
    exit 1
fi

# Determine output directory from model path if not specified
if [ -z "$OUTPUT_PATH" ]; then
    MODEL_DIR=$(dirname "$MODEL_PATH")
    MODEL_NAME=$(basename "$MODEL_PATH" .pth)
    OUTPUT_DIR="results/MultiTask_Transformer_Analysis/$(basename "$MODEL_DIR")"
    mkdir -p "$OUTPUT_DIR"
    
    if [ -n "$TIME_INTERVALS" ]; then
        # Extract range for output filename
        if [[ "$TIME_INTERVALS" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            OUTPUT_PATH="${OUTPUT_DIR}/${BASH_REMATCH[1]}-${BASH_REMATCH[2]}min_stage_metrics.png"
        else
            # Use first and last from list
            IFS=',' read -ra intervals <<< "$TIME_INTERVALS"
            first="${intervals[0]}"
            last="${intervals[-1]}"
            OUTPUT_PATH="${OUTPUT_DIR}/${first}-${last}min_stage_metrics.png"
        fi
    else
        OUTPUT_PATH="${OUTPUT_DIR}/stage_metrics.png"
    fi
fi

# If time_intervals is provided, run analysis for each interval
if [ -n "$TIME_INTERVALS" ]; then
    echo "=========================================="
    echo "Multi-Task Transformer Multi-Interval Analysis"
    echo "=========================================="
    echo "Model Path:      $MODEL_PATH"
    echo "Time Intervals:  $TIME_INTERVALS"
    echo "Test Data Prefix: $TEST_DATA_PREFIX"
    echo "Output Path:     $OUTPUT_PATH"
    echo "=========================================="
    echo ""
    
    # Parse intervals
    intervals_array=($(parse_time_intervals "$TIME_INTERVALS"))
    
    if [ ${#intervals_array[@]} -eq 0 ]; then
        echo "❌ No valid time intervals found"
        exit 1
    fi
    
    echo "📊 Analyzing ${#intervals_array[@]} time intervals: ${intervals_array[*]}"
    echo ""
    
    # Check which test data files exist
    missing_files=()
    test_paths=()
    for interval in "${intervals_array[@]}"; do
        test_path="${TEST_DATA_PREFIX}_${interval}min.parquet"
        if [ -f "$test_path" ]; then
            test_paths+=("$test_path")
            echo "  ✓ Found: $test_path"
        else
            missing_files+=("$test_path")
            echo "  ✗ Missing: $test_path"
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        echo ""
        echo "⚠ Warning: ${#missing_files[@]} test data file(s) not found:"
        for file in "${missing_files[@]}"; do
            echo "    - $file"
        done
        echo ""
        if [ ${#test_paths[@]} -eq 0 ]; then
            echo "❌ No test data files found. Exiting."
            exit 1
        fi
        echo "Continuing with available files..."
        echo ""
    fi
    
    # Build command with all test data paths (comma-separated)
    # The Python script will handle multiple paths and combine results
    
    CMD="python3 \"src/Analyzation&Visualization/analyze_win_classifier_stages.py\""
    CMD="$CMD --model_path \"$MODEL_PATH\""
    if [ -n "$MODEL_TYPE" ]; then
        CMD="$CMD --model_type \"$MODEL_TYPE\""
    fi
    CMD="$CMD --batch_size $BATCH_SIZE"
    CMD="$CMD --output_path \"$OUTPUT_PATH\""
    
    # Join all test paths with commas
    test_paths_str=$(IFS=','; echo "${test_paths[*]}")
    CMD="$CMD --test_data_path \"$test_paths_str\""
    
    echo "Running analysis for all intervals..."
    echo "Command: $CMD"
    echo ""
    
    if eval $CMD; then
        echo ""
        echo "=========================================="
        echo "✅ Multi-interval analysis completed successfully!"
        echo "=========================================="
        echo ""
        echo "Results saved to: $OUTPUT_PATH"
        echo ""
        echo "Analyzed intervals: ${intervals_array[*]}"
        echo "Test data files: ${test_paths[*]}"
    else
        echo ""
        echo "=========================================="
        echo "❌ Multi-interval analysis failed!"
        echo "=========================================="
        exit 1
    fi
    
else
    # Original single analysis mode
    echo "=========================================="
    echo "Multi-Task Transformer Stage Analysis"
    echo "=========================================="
    echo "Model Path:      $MODEL_PATH"
    echo "Model Type:      $MODEL_TYPE"
    echo "Stages:          $STAGES"
    echo "Batch Size:      $BATCH_SIZE"
    if [ -n "$TEST_DATA_PATH" ]; then
        echo "Test Data Path:  $TEST_DATA_PATH"
    fi
    echo "Output Path:     $OUTPUT_PATH"
    echo "=========================================="
    echo ""
    
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
    echo ""
    eval $CMD
    
    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ Multi-task transformer stage analysis completed successfully!"
        echo "=========================================="
        echo ""
        echo "Results saved to: $OUTPUT_PATH"
    else
        echo ""
        echo "=========================================="
        echo "❌ Stage analysis failed!"
        echo "=========================================="
        exit $STATUS
    fi
fi

