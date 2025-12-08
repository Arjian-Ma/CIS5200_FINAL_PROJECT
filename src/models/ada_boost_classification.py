#!/usr/bin/env python3
"""
AdaBoost Classifier for League of Legends match prediction
Classification version predicting win/loss (Y_won) instead of regression
Predicts at a specific frame_idx (e.g., 15 minutes into the game)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from datetime import datetime

# Configuration
PREDICTION_FRAME_IDX = 25  # Predict at 15 minutes (frame_idx = 15)
OUTPUT_DIR = "results/Model_AdaBoost_Classification"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*60)
print("ADABOOST CLASSIFIER - WIN PREDICTION")
print("="*60)
print(f"Prediction Frame Index: {PREDICTION_FRAME_IDX}")
print(f"Output Directory: {OUTPUT_DIR}")
print("="*60)

# Load data
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)
print("Loading data from parquet file...")
df = pd.read_parquet("Data/processed/featured_data.parquet")

print(f"✓ Full data loaded: {df.shape}")
print(f"Frame index range: {df['frame_idx'].min()} to {df['frame_idx'].max()}")

# Filter data: only use frames up to PREDICTION_FRAME_IDX
print(f"\nFiltering data for frame_idx <= {PREDICTION_FRAME_IDX}...")
df_filtered = df[df['frame_idx'] <= PREDICTION_FRAME_IDX].copy()
print(f"Filtered data shape: {df_filtered.shape}")

# Check game durations and filter out games shorter than PREDICTION_FRAME_IDX
print(f"\nChecking game durations...")
game_durations = df_filtered.groupby('match_id')['frame_idx'].max()
games_to_keep = game_durations[game_durations >= PREDICTION_FRAME_IDX].index
print(f"Total games: {len(df_filtered['match_id'].unique())}")
print(f"Games with duration >= {PREDICTION_FRAME_IDX}: {len(games_to_keep)}")
print(f"Games to exclude (too short): {len(df_filtered['match_id'].unique()) - len(games_to_keep)}")

df_filtered = df_filtered[df_filtered['match_id'].isin(games_to_keep)].copy()
print(f"Final filtered data shape: {df_filtered.shape}")

# Consolidate all frames 0 to PREDICTION_FRAME_IDX into a single row per match
# This concatenates features from all frames horizontally
print(f"\nConsolidating frames 0 to {PREDICTION_FRAME_IDX} into single rows per match...")
print("This will create one row per match with features from all frames concatenated")

# Get feature columns first (exclude match_id, frame_idx, timestamp, Y_won)
exclude_cols_temp = ['match_id', 'frame_idx', 'timestamp', 'Y_won']
feature_cols = [col for col in df_filtered.columns if col not in exclude_cols_temp]

def consolidate_match_frames(group):
    """Consolidate all frames for a match into a single row"""
    # Sort by frame_idx to ensure consistent ordering
    group_sorted = group.sort_values('frame_idx')
    
    # Get Y_won (should be same for all frames, take first)
    y_won = group_sorted['Y_won'].iloc[0]
    match_id = group_sorted['match_id'].iloc[0]
    
    # Concatenate features from each frame
    consolidated_features = {}
    for idx, row in group_sorted.iterrows():
        frame_idx = int(row['frame_idx'])
        for col in feature_cols:
            # Add frame suffix to feature name
            consolidated_features[f"{col}_frame{frame_idx}"] = row[col]
    
    # Add match_id and Y_won
    consolidated_features['match_id'] = match_id
    consolidated_features['Y_won'] = y_won
    
    return pd.Series(consolidated_features)

# Group by match_id and consolidate
print("Processing matches...")
consolidated_rows = []
for match_id, group in tqdm(df_filtered.groupby('match_id'), desc="Consolidating matches"):
    consolidated_row = consolidate_match_frames(group)
    consolidated_rows.append(consolidated_row)

df_consolidated = pd.DataFrame(consolidated_rows)
print(f"Consolidated data shape: {df_consolidated.shape}")
print(f"Expected shape: ({len(games_to_keep)}, {len(feature_cols) * (PREDICTION_FRAME_IDX + 1) + 2})")

# Check Y_won distribution (now per match, not per frame)
print(f"\nY_won distribution (per match):")
print(df_consolidated['Y_won'].value_counts())
print(f"Total matches: {len(df_consolidated)}")
print(f"Class balance: {df_consolidated['Y_won'].mean():.3f} (1 = win)")

# Keep all features - no leakage removal for this trial
print(f"\nKeeping all features (no leakage removal for this trial)")

# Split data by match_id (70% train, 15% val, 15% test)
unique_matches = df_consolidated['match_id'].unique()
n_matches = len(unique_matches)
print(f"\nTotal unique matches: {n_matches}")

np.random.seed(42)
shuffled_matches = np.random.permutation(unique_matches)

n_train = int(n_matches * 0.7)
n_val = int(n_matches * 0.15)

train_matches = shuffled_matches[:n_train]
val_matches = shuffled_matches[n_train:n_train + n_val]
test_matches = shuffled_matches[n_train + n_val:]

train_df = df_consolidated[df_consolidated['match_id'].isin(train_matches)].copy()
val_df = df_consolidated[df_consolidated['match_id'].isin(val_matches)].copy()
test_df = df_consolidated[df_consolidated['match_id'].isin(test_matches)].copy()

print(f"\nData split:")
print(f"Train: {train_df.shape} ({len(train_matches)} matches)")
print(f"Val: {val_df.shape} ({len(val_matches)} matches)")
print(f"Test: {test_df.shape} ({len(test_matches)} matches)")

# Prepare features and labels
label_col = 'Y_won'
exclude_cols = ['match_id', label_col]
feature_columns = [col for col in df_consolidated.columns if col not in exclude_cols]

print(f"\nNumber of features: {len(feature_columns)}")
print(f"Features per frame: {len(feature_columns) // (PREDICTION_FRAME_IDX + 1)}")
print(f"Total frames: {PREDICTION_FRAME_IDX + 1}")
print(f"First 10 features: {feature_columns[:10]}")

# Prepare X and y
X_train = train_df[feature_columns].values
y_train = train_df[label_col].values

X_val = val_df[feature_columns].values
y_val = val_df[label_col].values

X_test = test_df[feature_columns].values
y_test = test_df[label_col].values

# Handle any NaN values
X_train = np.nan_to_num(X_train, nan=0.0)
X_val = np.nan_to_num(X_val, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

print(f"\nTrain samples: {len(X_train)}")
print(f"Val samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Train AdaBoost Classifier
print("\n" + "="*60)
print("TRAINING ADABOOST CLASSIFIER")
print("="*60)

n_estimators = 200
# Create base estimator (Decision Tree)
# Note: In newer sklearn versions, use 'estimator' instead of 'base_estimator'
# Increased max_depth from 1 to 3 to allow more confident predictions
base_estimator = DecisionTreeClassifier(
    max_depth=3,  # Increased from 1 to allow stronger base learners
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

ada_model = AdaBoostClassifier(
    estimator=base_estimator,  # Changed from base_estimator to estimator
    n_estimators=n_estimators,
    learning_rate=0.8,  # Slightly lower learning rate for more stable, confident predictions
    algorithm='SAMME',  
    random_state=42
)

# Track metrics during training
train_accuracies = []
val_accuracies = []
train_f1_scores = []
val_f1_scores = []

print(f"Training with {n_estimators} estimators...")

# Train incrementally to track metrics
for i in tqdm(range(10, n_estimators + 1, 10), desc="Training", unit="estimator"):
    ada_model.set_params(n_estimators=i)
    ada_model.fit(X_train, y_train)
    
    # Calculate metrics
    train_pred = ada_model.predict(X_train)
    val_pred = ada_model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    
    train_accuracies.append((i, train_acc))
    val_accuracies.append((i, val_acc))
    train_f1_scores.append((i, train_f1))
    val_f1_scores.append((i, val_f1))

# Final training with all estimators (if not already done)
if n_estimators % 10 != 0:
    print("Final training with all estimators...")
    ada_model.set_params(n_estimators=n_estimators)
    ada_model.fit(X_train, y_train)
    
    # Final metrics
    train_pred = ada_model.predict(X_train)
    val_pred = ada_model.predict(X_val)
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    
    train_accuracies.append((n_estimators, train_acc))
    val_accuracies.append((n_estimators, val_acc))
    train_f1_scores.append((n_estimators, train_f1))
    val_f1_scores.append((n_estimators, val_f1))

print("\n✓ Training complete!")

# Make predictions (using raw probabilities for initial predictions)
# We'll update these after sharpening
y_train_pred_raw = ada_model.predict(X_train)
y_val_pred_raw = ada_model.predict(X_val)
y_test_pred_raw = ada_model.predict(X_test)

# Get prediction probabilities (winning rate)
y_train_proba_raw = ada_model.predict_proba(X_train)[:, 1]
y_val_proba_raw = ada_model.predict_proba(X_val)[:, 1]
y_test_proba_raw = ada_model.predict_proba(X_test)[:, 1]

# Apply probability sharpening to push predictions away from 0.5
# This penalizes uncertain predictions and encourages more confident/extreme predictions
def sharpen_probabilities(proba, sharpening_factor=2.0):
    """
    Sharpen probabilities to make them more extreme (closer to 0 or 1).
    
    Args:
        proba: Array of probabilities (0-1)
        sharpening_factor: How much to sharpen (higher = more extreme)
                          - 1.0 = no change
                          - 2.0 = moderate sharpening
                          - 3.0 = strong sharpening
    
    Returns:
        Sharpened probabilities
    """
    # Transform probabilities to logit space, scale, then transform back
    # This pushes values away from 0.5
    epsilon = 1e-10  # Avoid log(0) or log(1)
    proba_clipped = np.clip(proba, epsilon, 1 - epsilon)
    logit = np.log(proba_clipped / (1 - proba_clipped))
    sharpened_logit = logit * sharpening_factor
    sharpened_proba = 1 / (1 + np.exp(-sharpened_logit))
    return np.clip(sharpened_proba, 0, 1)

# Apply sharpening
SHARPENING_FACTOR = 2.0  # Adjust this: higher = more extreme predictions
y_train_proba = sharpen_probabilities(y_train_proba_raw, SHARPENING_FACTOR)
y_val_proba = sharpen_probabilities(y_val_proba_raw, SHARPENING_FACTOR)
y_test_proba = sharpen_probabilities(y_test_proba_raw, SHARPENING_FACTOR)

# Print statistics about prediction confidence
print(f"\n{'='*60}")
print("PREDICTION CONFIDENCE STATISTICS")
print(f"{'='*60}")
print(f"Sharpening Factor: {SHARPENING_FACTOR}")
print(f"\nTest Set - Before Sharpening:")
print(f"  Min: {y_test_proba_raw.min():.4f}, Max: {y_test_proba_raw.max():.4f}")
print(f"  Mean: {y_test_proba_raw.mean():.4f}, Std: {y_test_proba_raw.std():.4f}")
print(f"  Predictions near 0.5 (0.4-0.6): {(np.abs(y_test_proba_raw - 0.5) < 0.1).sum()} / {len(y_test_proba_raw)} ({(np.abs(y_test_proba_raw - 0.5) < 0.1).mean()*100:.1f}%)")
print(f"  Predictions extreme (<0.2 or >0.8): {((y_test_proba_raw < 0.2) | (y_test_proba_raw > 0.8)).sum()} / {len(y_test_proba_raw)} ({((y_test_proba_raw < 0.2) | (y_test_proba_raw > 0.8)).mean()*100:.1f}%)")

print(f"\nTest Set - After Sharpening:")
print(f"  Min: {y_test_proba.min():.4f}, Max: {y_test_proba.max():.4f}")
print(f"  Mean: {y_test_proba.mean():.4f}, Std: {y_test_proba.std():.4f}")
print(f"  Predictions near 0.5 (0.4-0.6): {(np.abs(y_test_proba - 0.5) < 0.1).sum()} / {len(y_test_proba)} ({(np.abs(y_test_proba - 0.5) < 0.1).mean()*100:.1f}%)")
print(f"  Predictions extreme (<0.2 or >0.8): {((y_test_proba < 0.2) | (y_test_proba > 0.8)).sum()} / {len(y_test_proba)} ({((y_test_proba < 0.2) | (y_test_proba > 0.8)).mean()*100:.1f}%)")
print(f"{'='*60}\n")

# Update predictions based on sharpened probabilities
y_train_pred = (y_train_proba >= 0.5).astype(int)
y_val_pred = (y_val_proba >= 0.5).astype(int)
y_test_pred = (y_test_proba >= 0.5).astype(int)

# Calculate metrics for all sets
def print_metrics(y_true, y_pred, y_proba, set_name):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = 0.0
    
    print(f"\n{'='*60}")
    print(f"{set_name} METRICS")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

train_metrics = print_metrics(y_train, y_train_pred, y_train_proba, "TRAINING SET")
val_metrics = print_metrics(y_val, y_val_pred, y_val_proba, "VALIDATION SET")
test_metrics = print_metrics(y_test, y_test_pred, y_test_proba, "TEST SET")

# Create metrics table
print(f"\n{'='*60}")
print("METRICS SUMMARY TABLE")
print(f"{'='*60}")
metrics_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Train': [
        train_metrics['accuracy'],
        train_metrics['precision'],
        train_metrics['recall'],
        train_metrics['f1']
    ],
    'Validation': [
        val_metrics['accuracy'],
        val_metrics['precision'],
        val_metrics['recall'],
        val_metrics['f1']
    ],
    'Test': [
        test_metrics['accuracy'],
        test_metrics['precision'],
        test_metrics['recall'],
        test_metrics['f1']
    ]
})
print(metrics_table.to_string(index=False))
print(f"{'='*60}")

# Save metrics table to CSV
metrics_table.to_csv(f'{OUTPUT_DIR}/metrics_table_frame{PREDICTION_FRAME_IDX}.csv', index=False)
print(f"✓ Metrics table saved as '{OUTPUT_DIR}/metrics_table_frame{PREDICTION_FRAME_IDX}.csv'")

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

# Create visualization with ROC curve and Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: ROC Curve for Test Set
axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {test_metrics["auc"]:.4f})')
axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title(f'ROC Curve - Test Set (Frame {PREDICTION_FRAME_IDX}) - AdaBoost', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Confusion Matrix - Test Set
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[1])
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_title(f'Confusion Matrix - Test Set (Frame {PREDICTION_FRAME_IDX}) - AdaBoost', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(['Lose (0)', 'Win (1)'])
axes[1].set_yticklabels(['Lose (0)', 'Win (1)'])

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/adaboost_classification_results_frame{PREDICTION_FRAME_IDX}.png', 
            dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved as '{OUTPUT_DIR}/adaboost_classification_results_frame{PREDICTION_FRAME_IDX}.png'")

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
print(f"\nSummary:")
print(f"  - Prediction Frame: {PREDICTION_FRAME_IDX}")
print(f"  - Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  - Test Precision: {test_metrics['precision']:.4f}")
print(f"  - Test Recall: {test_metrics['recall']:.4f}")
print(f"  - Test F1 Score: {test_metrics['f1']:.4f}")
print(f"  - Test ROC AUC: {test_metrics['auc']:.4f}")
print(f"{'='*60}\n")

