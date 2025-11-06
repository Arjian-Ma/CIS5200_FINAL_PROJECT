#!/usr/bin/env python3
"""
Visualize Gradient Boosting model predictions on multiple test games
Shows real vs predicted gold difference over time for individual matches
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import pickle

print("="*60)
print("GRADIENT BOOSTING - TEST GAME VISUALIZATION")
print("="*60)

# Load data
print("\nLoading data...")
df = pd.read_parquet("Data/processed/featured_data.parquet")

# Remove data leakage features
leakage_features = [
    'Total_Gold_Difference_Last_Time_Frame',
    'Total_Minions_Killed_Difference',
    'Total_Jungle_Minions_Killed_Difference',
    'Total_Kill_Difference',
    'Total_Assist_Difference',
    'Elite_Monster_Killed_Difference',
    'Buildings_Taken_Difference',
    'Total_Xp_Difference_Last_Time_Frame',
]

df_clean = df.drop(columns=[f for f in leakage_features if f in df.columns], errors='ignore')
print(f"✓ Data loaded: {df_clean.shape}")

# Split data the same way as training
unique_matches = df_clean['match_id'].unique()
np.random.seed(42)
shuffled_matches = np.random.permutation(unique_matches)

n_train = int(len(unique_matches) * 0.7)
n_val = int(len(unique_matches) * 0.15)

test_matches = shuffled_matches[n_train + n_val:]
test_df = df_clean[df_clean['match_id'].isin(test_matches)].copy()

print(f"✓ Test set: {len(test_df)} frames from {len(test_matches)} matches")

# Prepare features
label_col = 'Total_Gold_Difference'
exclude_cols = ['match_id', 'frame_idx', 'timestamp', label_col]
feature_columns = [col for col in df_clean.columns if col not in exclude_cols]

print(f"✓ Features: {len(feature_columns)}")

# Train a fresh model (or load if you have it saved)
print("\nTraining Gradient Boosting model...")

train_matches = shuffled_matches[:n_train]
val_matches = shuffled_matches[n_train:n_train + n_val]

train_df = df_clean[df_clean['match_id'].isin(train_matches)].copy()
val_df = df_clean[df_clean['match_id'].isin(val_matches)].copy()

X_train = train_df[feature_columns].values
y_train = train_df[label_col].values
X_train = np.nan_to_num(X_train, nan=0.0)

# Train model (using same hyperparameters as your trained model)
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=0
)
gb_model.fit(X_train, y_train)
print("✓ Model trained")

# Select 6 diverse test games
print("\nSelecting test games...")
test_match_frame_counts = test_df.groupby('match_id').size()
good_test_matches = test_match_frame_counts[test_match_frame_counts >= 15].index

np.random.seed(100)
selected_matches = np.random.choice(good_test_matches, size=min(6, len(good_test_matches)), replace=False)

print(f"Selected {len(selected_matches)} test games")

# Evaluate each game
all_game_results = []

for match_idx, match_id in enumerate(selected_matches):
    print(f"Processing game {match_idx + 1}/{len(selected_matches)}: {match_id}...")
    
    game_data = test_df[test_df['match_id'] == match_id].copy().sort_values('frame_idx').reset_index(drop=True)
    
    # Get features and predictions
    X_game = game_data[feature_columns].values
    X_game = np.nan_to_num(X_game, nan=0.0)
    
    y_real = game_data[label_col].values
    y_pred = gb_model.predict(X_game)
    
    frames = game_data['frame_idx'].values
    
    all_game_results.append({
        'match_id': match_id,
        'real': y_real,
        'predicted': y_pred,
        'frames': frames
    })

# Calculate overall statistics
all_real = np.concatenate([g['real'] for g in all_game_results])
all_pred = np.concatenate([g['predicted'] for g in all_game_results])

overall_rmse = np.sqrt(np.mean((all_real - all_pred) ** 2))
overall_mae = np.mean(np.abs(all_real - all_pred))
overall_r2 = 1 - (np.sum((all_real - all_pred) ** 2) / np.sum((all_real - np.mean(all_real)) ** 2))

print(f"\n{'='*60}")
print("OVERALL TEST PERFORMANCE (6 GAMES)")
print(f"{'='*60}")
print(f"RMSE: {overall_rmse:.2f} gold")
print(f"MAE: {overall_mae:.2f} gold")
print(f"R² Score: {overall_r2:.4f}")
print(f"Total predictions: {len(all_real)}")

# Create individual game predictions plot (6 games)
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig = plt.figure(figsize=(18, 12))

for idx, game in enumerate(all_game_results):
    ax = plt.subplot(2, 3, idx + 1)
    
    real = game['real']
    pred = game['predicted']
    frames = game['frames']
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((real - pred) ** 2))
    mae = np.mean(np.abs(real - pred))
    r2 = 1 - (np.sum((real - pred) ** 2) / np.sum((real - np.mean(real)) ** 2))
    
    # Plot
    ax.plot(frames, real, 'b-', label='Real', linewidth=2.5, marker='o', markersize=5, alpha=0.8)
    ax.plot(frames, pred, 'r--', label='Predicted', linewidth=2.5, marker='s', markersize=5, alpha=0.8)
    ax.fill_between(frames, real, pred, alpha=0.2, color='purple')
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Frame Index', fontsize=11)
    ax.set_ylabel('Gold Difference', fontsize=11)
    
    # Truncate match ID for display
    match_display = game['match_id'][:20] + '...' if len(game['match_id']) > 20 else game['match_id']
    ax.set_title(f'Game {idx+1}: {match_display}\nRMSE={rmse:.0f}, MAE={mae:.0f}, R²={r2:.3f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('Gradient Boosting - Individual Test Game Predictions', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/gradient_boosting_test_games.png', dpi=300, bbox_inches='tight')
print(f"✓ Individual game predictions saved to: results/gradient_boosting_test_games.png")
plt.close()

# Create comprehensive evaluation plot
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: All games combined - scatter
ax1 = axes[0, 0]
colors = plt.cm.tab10(np.linspace(0, 1, len(all_game_results)))
for idx, game in enumerate(all_game_results):
    ax1.scatter(game['real'], game['predicted'], alpha=0.6, s=60, 
               label=f'Game {idx+1}', color=colors[idx], edgecolors='k', linewidth=0.5)

min_val = min(all_real.min(), all_pred.min())
max_val = max(all_real.max(), all_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction', alpha=0.8)
ax1.set_xlabel('Real Gold Difference', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Gold Difference', fontsize=12, fontweight='bold')
ax1.set_title('All Test Games: Predicted vs Real', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='best', ncol=2)
ax1.grid(True, alpha=0.3)

metrics_text = f'RMSE: {overall_rmse:.2f}\nMAE: {overall_mae:.2f}\nR²: {overall_r2:.4f}'
ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Plot 2: Error distribution across all games
ax2 = axes[0, 1]
errors = all_pred - all_real
ax2.hist(errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1)
ax2.axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
ax2.axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2.5, 
           label=f'Mean: {np.mean(errors):.2f}')
ax2.set_xlabel('Prediction Error (Gold)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Error Distribution (All Test Games)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: RMSE per game
ax3 = axes[1, 0]
game_rmses = [np.sqrt(np.mean((g['real'] - g['predicted']) ** 2)) for g in all_game_results]
game_labels = [f"Game {i+1}" for i in range(len(all_game_results))]
bars = ax3.bar(game_labels, game_rmses, color='coral', edgecolor='k', linewidth=1.5, alpha=0.8)
ax3.axhline(y=overall_rmse, color='r', linestyle='--', linewidth=2.5, label=f'Average: {overall_rmse:.2f}')
ax3.set_xlabel('Test Game', fontsize=12, fontweight='bold')
ax3.set_ylabel('RMSE (Gold)', fontsize=12, fontweight='bold')
ax3.set_title('RMSE by Test Game', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, rmse in zip(bars, game_rmses):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{rmse:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: Prediction quality over game progression
ax4 = axes[1, 1]
all_frames = np.concatenate([g['frames'] for g in all_game_results])
all_errors_abs = np.abs(errors)

# Create bins: early (0-10), mid (10-20), late (20+)
early_mask = all_frames < 10
mid_mask = (all_frames >= 10) & (all_frames < 20)
late_mask = all_frames >= 20

early_mae = np.mean(all_errors_abs[early_mask]) if early_mask.sum() > 0 else 0
mid_mae = np.mean(all_errors_abs[mid_mask]) if mid_mask.sum() > 0 else 0
late_mae = np.mean(all_errors_abs[late_mask]) if late_mask.sum() > 0 else 0

phases = ['Early Game\n(0-10 min)', 'Mid Game\n(10-20 min)', 'Late Game\n(20+ min)']
maes = [early_mae, mid_mae, late_mae]
colors_phase = ['green', 'orange', 'red']

bars = ax4.bar(phases, maes, color=colors_phase, edgecolor='k', linewidth=1.5, alpha=0.7)
ax4.set_ylabel('Mean Absolute Error (Gold)', fontsize=12, fontweight='bold')
ax4.set_title('Prediction Error by Game Phase', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, mae_val in zip(bars, maes):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{mae_val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Gradient Boosting Model - Comprehensive Test Evaluation', 
             fontsize=16, fontweight='bold', y=0.998)
plt.tight_layout()
plt.savefig('results/gradient_boosting_comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Comprehensive evaluation saved to: results/gradient_boosting_comprehensive_evaluation.png")
plt.close()

# Print detailed game-by-game results
print(f"\n{'='*60}")
print("GAME-BY-GAME RESULTS")
print(f"{'='*60}")
print(f"{'Game':<6} {'Match ID':<25} {'Frames':<8} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
print("-"*60)

for idx, game in enumerate(all_game_results):
    rmse = np.sqrt(np.mean((game['real'] - game['predicted']) ** 2))
    mae = np.mean(np.abs(game['real'] - game['predicted']))
    r2 = 1 - (np.sum((game['real'] - game['predicted']) ** 2) / np.sum((game['real'] - np.mean(game['real'])) ** 2))
    
    match_display = game['match_id'][:25]
    print(f"{idx+1:<6} {match_display:<25} {len(game['real']):<8} {rmse:>10.2f} {mae:>10.2f} {r2:>8.4f}")

print(f"\n{'='*60}")
print("OVERALL STATISTICS")
print(f"{'='*60}")
print(f"Total test predictions: {len(all_real)}")
print(f"Average RMSE: {overall_rmse:.2f} gold")
print(f"Average MAE: {overall_mae:.2f} gold")
print(f"Average R²: {overall_r2:.4f}")

# Print sample predictions from one game
print(f"\n{'='*60}")
print(f"SAMPLE PREDICTIONS - GAME 1: {all_game_results[0]['match_id']}")
print(f"{'='*60}")
print(f"{'Frame':<8} {'Real':>12} {'Predicted':>12} {'Error':>12} {'%Error':>10}")
print("-"*60)

game = all_game_results[0]
for i in range(min(15, len(game['frames']))):
    real_val = game['real'][i]
    pred_val = game['predicted'][i]
    error = pred_val - real_val
    pct_error = 100 * error / (abs(real_val) + 1) if abs(real_val) > 100 else 0
    print(f"{game['frames'][i]:<8} {real_val:>12.2f} {pred_val:>12.2f} {error:>12.2f} {pct_error:>9.1f}%")

print(f"\n{'='*60}")
print("EVALUATION COMPLETE")
print(f"{'='*60}")
print("\nGenerated files:")
print("  1. results/gradient_boosting_test_games.png - Individual game trajectories")
print("  2. results/gradient_boosting_comprehensive_evaluation.png - Statistical analysis")

