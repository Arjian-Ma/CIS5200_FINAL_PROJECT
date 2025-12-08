# Scripts Directory

This directory contains shell scripts for running model analysis and evaluation.

## Available Scripts

### 1. `analyze_lstm_test_games.sh`

Quick script to analyze LSTM model test games with default settings.

**Usage:**
```bash
./scripts/analyze_lstm_test_games.sh
```

**Environment Variables (optional):**
```bash
MODEL_PATH="models/lstm_model.pth" \
DATA_PATH="data/processed/featured_data_with_scores.parquet" \
OUTPUT_PATH="results/lstm_analysis.png" \
./scripts/analyze_lstm_test_games.sh
```

### 2. `analyze_model_test_games.sh`

General script to analyze any trained model (LSTM, Autoregressive, etc.).

**Usage:**
```bash
# Analyze LSTM model
./scripts/analyze_model_test_games.sh -m models/lstm_model.pth -t lstm

# Analyze autoregressive model
./scripts/analyze_model_test_games.sh -m results/autoregressive_hierarchical_model.pth -t autoregressive

# Use custom feature list from CSV
./scripts/analyze_model_test_games.sh \
    -m models/lstm_model.pth \
    -t lstm \
    -f data/Feature_Selection_RFE/best_features_rfe.csv

# Use comma-separated feature list
./scripts/analyze_model_test_games.sh \
    -m models/lstm_model.pth \
    -t lstm \
    -f "Total_Gold_Difference_Last_Time_Frame,Total_Xp_Difference_Last_Time_Frame,CentroidDist"

# Show help
./scripts/analyze_model_test_games.sh -h
```

**Options:**
- `-m, --model_path PATH`: Path to model checkpoint (required)
- `-t, --model_type TYPE`: Model type: auto, lstm, autoregressive (default: auto)
- `-d, --data_path PATH`: Path to featured data (default: data/processed/featured_data_with_scores.parquet)
- `-f, --feature_list LIST`: Feature list: 'specified', CSV path, or comma-separated (default: specified)
- `-c, --target_col COL`: Target column name (default: Total_Gold_Difference)
- `-s, --sequence_length N`: Sequence length for LSTM (default: 15)
- `-o, --output_path PATH`: Output path for plot (default: auto-generated)
- `-h, --help`: Show help message

## Feature Lists

The scripts support different ways to specify features:

1. **`'specified'`** (default for LSTM): Uses `get_specified_features()` from `lstm_model.py`, which returns the 39 features used during training:
   - Total_Gold_Difference_Last_Time_Frame
   - Total_Xp_Difference_Last_Time_Frame
   - Damage features (Magic, Physical, True damage done/taken)
   - Game stats (Kills, Assists, Wards, etc.)
   - Spatial features (CentroidDist, MinInterTeamDist, etc.)
   - Team scores (Offensive, Defensive, Overall)

2. **CSV file**: Path to a CSV file containing feature names in a column (e.g., `data/Feature_Selection_RFE/best_features_rfe.csv`)

3. **Comma-separated**: Direct list of features separated by commas

## Output

Both scripts generate:
- Console output with RMSE distribution statistics
- A plot saved to `results/` directory showing:
  - Histogram of RMSE distribution
  - Cumulative distribution curve
  - Threshold markers (1K, 3K gold)

## Prerequisites

1. Trained model checkpoint (`.pth` file)
2. Featured data file (`.parquet` or `.csv`)
3. Python environment with required packages:
   - torch
   - pandas
   - numpy
   - matplotlib
   - sklearn

## Examples

### Example 1: Analyze LSTM Model with Default Settings
```bash
./scripts/analyze_lstm_test_games.sh
```

### Example 2: Analyze LSTM Model with Custom Output
```bash
./scripts/analyze_model_test_games.sh \
    -m models/lstm_model.pth \
    -t lstm \
    -o results/my_lstm_analysis.png
```

### Example 3: Analyze with RFE Selected Features
```bash
./scripts/analyze_model_test_games.sh \
    -m models/lstm_model.pth \
    -t lstm \
    -f data/Feature_Selection_RFE/best_features_rfe.csv \
    -o results/lstm_rfe_features_analysis.png
```

### Example 4: Analyze Autoregressive Model
```bash
./scripts/analyze_model_test_games.sh \
    -m results/Model_20251028_Autoregressive5TimeFrame100Batch/autoregressive_hierarchical_model.pth \
    -t autoregressive \
    -d Data/processed/featured_data.parquet
```

## Troubleshooting

1. **Model file not found**: Make sure the model path is correct and the model has been trained
2. **Data file not found**: Run `data_featuring_score.py` first to generate the featured data
3. **Feature mismatch**: Ensure the feature list matches what the model was trained with
4. **Import errors**: Make sure you're running from the project root directory

## Notes

- The scripts automatically detect model type from checkpoint if `--model_type auto` is used
- LSTM models use sequence-based prediction (sliding window)
- Autoregressive models use endgame forecasting (predicts future frames)
- The analysis processes all test games and generates comprehensive statistics

