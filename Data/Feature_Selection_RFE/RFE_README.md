# Recursive Feature Elimination (RFE) Script

## Overview
`src/data/feature_ranking.py` implements Recursive Feature Elimination with Gradient Boosting to find the optimal feature subset for predicting gold difference.

## Key Features

### 1. **Smart Feature Grouping**
- **Player Features**: All `Player1_X`, `Player2_X`, ..., `Player10_X` are treated as a group
- **Position Features**: `x_position` and `y_position` importance are combined into a single "position" metric
- When dropping features, entire player groups are dropped together (not individual players)

### 2. **Spatial Feature Integration**
- Automatically loads and merges spatial features from `featured_data_with_scores.parquet`:
  - CentroidDist
  - MinInterTeamDist
  - FrontlineOverlap
  - RadialVelocityDiff

### 3. **RFE Process**
- **Initial Features**: ~268 features
- **Drop Rate**: 15% of features per iteration (by base feature group)
- **Minimum Features**: Stops at 20 features
- **Model**: Gradient Boosting with 50 estimators per iteration
- **Tracking**: Records train/test RMSE and R² at each iteration

### 4. **Progress Logging**
The script provides detailed progress information:

```
[Step 1/7] Loading base feature data...
[Step 2/7] Loading spatial features...
[Step 3/7] Removing specific features...
[Step 4/7] Preparing features and target...
[Step 5/7] Splitting data (80/20 train/test)...
[Step 6/7] STARTING RECURSIVE FEATURE ELIMINATION
  [Iteration 1] Training with 268 features... ✓
  [Iteration 1] Dropping 40 features (5 base feature groups):
      - stat_name: 10 features
      - another_stat: 10 features
      ...
[Step 7/7] TRAINING FINAL MODEL WITH BEST FEATURES
  Progress: 10% 20% 30% ... 100% ✓ Complete!
```

## Output Files

### CSV Files
1. **rfe_history.csv** - Full iteration log with features, RMSE, R² for each step
2. **best_features_rfe.csv** - List of features selected by RFE
3. **feature_importance_aggregated.csv** - All features ranked by aggregated importance
4. **feature_importance_top50.csv** - Top 50 most important feature groups
5. **spatial_feature_importance.csv** - Spatial features only

### Visualizations
1. **rfe_analysis.png** - 3-panel plot showing:
   - RMSE vs number of features (train/test)
   - R² score progression
   - RMSE improvement by iteration

2. **feature_importance_gradient_boosting.png** - 2-panel plot showing:
   - Top 30 feature groups by importance
   - Cumulative importance curve

3. **feature_importance_by_category.png** - Pie chart of importance by category:
   - Combat, Economy, Farm, Vision, XP/Level, Objectives, Stats, Spatial

## How Feature Dropping Works

### Example: Dropping "kills"
```python
# If "kills" has low aggregated importance, RFE drops:
Player1_kills
Player2_kills
Player3_kills
...
Player10_kills
# All 10 features dropped together!
```

### Example: Position Features
```python
# x_position and y_position are combined in importance calculation:
Player1_x_position importance: 0.005
Player1_y_position importance: 0.003
# Combined "position" importance: 0.008

# When dropped, BOTH x and y are removed together
```

## Configuration

Edit these parameters in the script:

```python
drop_percentage = 0.15    # Drop 15% per iteration
min_features = 20         # Stop at 20 features
n_estimators = 50         # Boosting rounds per RFE iteration
```

## Running the Script

```bash
cd /Users/arjianma/ESE5380_FINAL
python3 src/data/feature_ranking.py
```

The script will:
1. Load data (~128k samples)
2. Run RFE iterations (typically 8-12 iterations)
3. Train final model with best features
4. Generate all visualizations and CSVs
5. Print summary statistics

Expected runtime: 10-20 minutes depending on hardware.

