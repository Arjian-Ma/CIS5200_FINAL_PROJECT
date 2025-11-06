# ESE5380 Final Project - Directory Structure

## Overview
This project implements a comprehensive machine learning pipeline for analyzing League of Legends match data using various models (RNN, LSTM, Transformer, Diffusion) and traditional ML approaches.

## Directory Structure

```
ESE5380_FINAL/
├── 📁 configs/                    # Configuration files
├── 📁 data/                      # Data storage (organized by processing stage)
│   ├── 📁 raw/                   # Original/raw data files
│   ├── 📁 processed/             # Cleaned and feature-engineered data
│   └── 📁 splits/                # Train/validation/test data splits
├── 📁 models/                    # Saved model files and checkpoints
│   ├── 📁 rnn/                   # RNN model checkpoints
│   ├── 📁 lstm/                  # LSTM model checkpoints
│   ├── 📁 transformer/           # Transformer model checkpoints
│   ├── 📁 diffusion/             # Diffusion model checkpoints
│   └── 📁 baselines/             # Traditional ML models (RF, XGBoost)
├── 📁 notebooks/                 # Jupyter notebooks for analysis
├── 📁 results/                   # Model outputs and visualizations
│   ├── 📁 logs/                  # Training logs
│   ├── 📁 checkpoints/           # Model checkpoints
│   └── 📁 predictions/           # Model predictions
├── 📁 scripts/                   # Training and utility scripts
├── 📁 src/                       # Source code (Python modules)
│   ├── 📁 data/                  # Data processing modules
│   ├── 📁 models/                # Model definitions
│   ├── 📁 training/              # Training utilities
│   └── 📁 evaluation/            # Evaluation metrics
├── 📁 timeline_data/             # Raw timeline JSON files (temporary)
├── 📄 README.md                  # Project documentation
└── 📄 PARSER_README.md           # Riot API parser documentation
```

## Detailed Directory Descriptions

### 📁 `configs/`
**Purpose**: Configuration files for different components
- **`data_config.yaml`**: Data processing parameters (splits, features, etc.)
- **`model_config.yaml`**: Model hyperparameters and architecture settings
- **`training_config.yaml`**: Training parameters (batch size, learning rate, etc.)

### 📁 `data/`
**Purpose**: Organized data storage following ML best practices

#### 📁 `data/raw/`
**Purpose**: Original, unprocessed data files
- **`xy_rows.csv`**: Raw match data with 10 rows per timestamp (1 per player)
- **`opgg_leaderboard.csv`**: Player leaderboard data for API scraping
- **`timeline_data/`**: Raw timeline JSON files from Riot API (should be moved here)

#### 📁 `data/processed/`
**Purpose**: Cleaned and feature-engineered data
- **`featured_data.csv`**: Aggregated team-level features (548 rows)
- **`featured_data_with_scores.csv`**: Data with composite player scores (548 rows)

#### 📁 `data/splits/`
**Purpose**: Train/validation/test data splits
- **`train.csv`**: Training data (70% of matches)
- **`val.csv`**: Validation data (15% of matches)
- **`test.csv`**: Test data (15% of matches)

### 📁 `models/`
**Purpose**: Saved model files and checkpoints
- **`rnn/`**: RNN model weights and checkpoints
- **`lstm/`**: LSTM model weights and checkpoints
- **`transformer/`**: Transformer model weights and checkpoints
- **`diffusion/`**: Diffusion model weights and checkpoints
- **`baselines/`**: Traditional ML models (Random Forest, XGBoost, etc.)

### 📁 `results/`
**Purpose**: Model outputs, visualizations, and analysis results
- **`logs/`**: Training logs and metrics
- **`checkpoints/`**: Model checkpoints during training
- **`predictions/`**: Model predictions on test data
- **`*.png`**: Visualization plots (feature importance, results, etc.)

### 📁 `src/`
**Purpose**: Source code organized by functionality

#### 📁 `src/data/`
**Purpose**: Data processing and feature engineering modules
- **`riot_parser.py`**: Riot API parser for collecting match timeline data
- **`build_xy_dataframe.py`**: Converts timeline JSONs to structured DataFrame
- **`data_featuring.py`**: Feature engineering (team aggregation, differences)
- **`data_featuring_score.py`**: Player scoring system (offensive, defensive, etc.)
- **`ID.py`**: Player ID management utilities
- **`RiotData.py`**: Riot API data structures

#### 📁 `src/models/`
**Purpose**: Model definitions and implementations
- **`RNN.py`**: RNN model implementation
- **`random_forest.py`**: Random Forest baseline model
- **`gradient_tree_boost.py`**: Gradient Boosting model
- **`regression.py`**: Linear regression baseline

#### 📁 `src/training/`
**Purpose**: Training utilities and trainers
- **`trainer.py`**: Generic training framework
- **`utils.py`**: Training utilities and helpers

#### 📁 `src/evaluation/`
**Purpose**: Evaluation metrics and analysis
- **`metrics.py`**: Model evaluation metrics

### 📁 `scripts/`
**Purpose**: Training and utility scripts
- **`train_lstm.py`**: LSTM training script
- **`train_transformer.py`**: Transformer training script
- **`train_diffusion.py`**: Diffusion model training script

### 📁 `notebooks/`
**Purpose**: Jupyter notebooks for analysis and exploration
- **`data_exploration.ipynb`**: Data exploration and visualization

### 📁 `timeline_data/` (Temporary)
**Purpose**: Currently contains raw timeline JSON files
- **Note**: This should be moved to `data/raw/timeline_data/` for proper organization
- **Contains**: 60+ timeline JSON files from Riot API

## Data Flow

```
Raw Data → Feature Engineering → Model Training → Evaluation
    ↓              ↓                    ↓            ↓
timeline_data/ → data/processed/ → models/ → results/
```

## Key Features

### 🎯 **Data Leakage Prevention**
- Removed deterministic features that directly compute targets
- Proper temporal splits to prevent future information leakage

### 🔄 **Modular Architecture**
- Clean separation between data processing, modeling, and evaluation
- Reusable components for different model types

### 📊 **Multi-Model Support**
- Sequential models: RNN, LSTM, Transformer, Diffusion
- Traditional ML: Random Forest, XGBoost, Linear Regression
- Proper PyTorch DataLoader implementation

### 🏗️ **ML Best Practices**
- Proper train/val/test splits by match (not by row)
- Configuration-driven approach
- Comprehensive logging and checkpointing
- Feature engineering pipeline

## Usage

1. **Data Collection**: Run `src/data/riot_parser.py` to collect timeline data
2. **Feature Engineering**: Run `src/data/data_featuring.py` and `src/data/data_featuring_score.py`
3. **Model Training**: Use scripts in `scripts/` directory
4. **Evaluation**: Check results in `results/` directory

## Notes

- The `timeline_data/` directory should be moved to `data/raw/timeline_data/` for proper organization
- All data processing scripts have been updated to use the new directory structure
- Configuration files provide easy parameter management
- The project follows ML engineering best practices with proper separation of concerns
