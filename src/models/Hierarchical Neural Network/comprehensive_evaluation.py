#!/usr/bin/env python3
"""
Comprehensive evaluation of the Temporal LSTM Hierarchical Model
1. Load training history and plot loss curves
2. Evaluate on multiple test games
3. Generate comprehensive visualizations
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

# Import from NN module
import importlib.util
spec = importlib.util.spec_from_file_location("NN", "src/models/NN.py")
NN_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NN_module)

TemporalHierarchicalDataset = NN_module.TemporalHierarchicalDataset
TemporalHierarchicalGoldPredictor = NN_module.TemporalHierarchicalGoldPredictor
FeatureGroups = NN_module.FeatureGroups

print("="*60)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*60)

# Load the parquet data
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

# Split data
unique_matches = df_clean['match_id'].unique()
np.random.seed(42)
shuffled_matches = np.random.permutation(unique_matches)

n_train = int(len(unique_matches) * 0.7)
n_val = int(len(unique_matches) * 0.15)

test_matches = shuffled_matches[n_train + n_val:]
test_df = df_clean[df_clean['match_id'].isin(test_matches)].copy()

print(f"✓ Test set: {len(test_df)} frames from {len(test_matches)} matches")

# Load model checkpoint
checkpoint = torch.load('results/temporal_hierarchical_model.pth', weights_only=False)
sequence_length = checkpoint['sequence_length']
scaler_dict = checkpoint['scaler_dict']

print(f"✓ Model checkpoint loaded (sequence_length={sequence_length})")

# Create model
fg = FeatureGroups()
model = TemporalHierarchicalGoldPredictor(fg, hidden_dim=32, lstm_layers=1)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# Select 6 diverse test games
print("\nSelecting test games...")
test_match_frame_counts = test_df.groupby('match_id').size()
good_test_matches = test_match_frame_counts[test_match_frame_counts >= sequence_length + 5].index

# Select games with different characteristics
np.random.seed(100)
selected_matches = np.random.choice(good_test_matches, size=min(6, len(good_test_matches)), replace=False)

print(f"Selected {len(selected_matches)} test games for evaluation")

# Evaluate each game
all_game_results = []

for match_idx, match_id in enumerate(selected_matches):
    print(f"\nProcessing game {match_idx + 1}/{len(selected_matches)}: {match_id}...")
    
    game_data = test_df[test_df['match_id'] == match_id].copy().sort_values('frame_idx').reset_index(drop=True)
    
    # Create dataset for this game
    game_dataset = TemporalHierarchicalDataset(
        game_data, fg,
        sequence_length=sequence_length,
        stride=1,
        scaler_dict=scaler_dict,
        fit_scaler=False
    )
    
    # Get predictions
    real_values = []
    predicted_values = []
    frame_indices = []
    
    target_mean = scaler_dict['target_mean']
    target_std = scaler_dict['target_std']
    
    with torch.no_grad():
        for idx in range(len(game_dataset)):
            batch = game_dataset[idx]
            
            # Add batch dimension
            batch_input = {
                'x1': batch['x1'].unsqueeze(0),
                'x2': batch['x2'].unsqueeze(0),
                'players': {}
            }
            
            for p_key in batch['players'].keys():
                batch_input['players'][p_key] = {
                    g_key: batch['players'][p_key][g_key].unsqueeze(0)
                    for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']
                }
            
            # Predict
            pred = model(batch_input)
            
            # Denormalize
            target_denorm = batch['target'].item() * target_std + target_mean
            pred_denorm = pred.item() * target_std + target_mean
            
            real_values.append(target_denorm)
            predicted_values.append(pred_denorm)
            frame_indices.append(game_data.iloc[idx + sequence_length - 1]['frame_idx'])
    
    all_game_results.append({
        'match_id': match_id,
        'real': np.array(real_values),
        'predicted': np.array(predicted_values),
        'frames': np.array(frame_indices)
    })

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Create comprehensive multi-game visualization
fig = plt.figure(figsize=(18, 12))

# Plot each game in a 2x3 grid
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
    ax.plot(frames, real, 'b-', label='Real', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
    ax.plot(frames, pred, 'r--', label='Predicted', linewidth=2.5, marker='s', markersize=4, alpha=0.8)
    ax.fill_between(frames, real, pred, alpha=0.2, color='gray')
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Frame Index', fontsize=10)
    ax.set_ylabel('Gold Difference', fontsize=10)
    ax.set_title(f'Game {idx+1}: {game["match_id"][:15]}...\nRMSE={rmse:.0f}, MAE={mae:.0f}, R²={r2:.3f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('Temporal LSTM Hierarchical Model - Test Game Predictions', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/temporal_test_games.png', dpi=300, bbox_inches='tight')
print(f"✓ Multi-game visualization saved to: results/temporal_test_games.png")

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

# Create detailed summary plot
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: All games combined - scatter
ax1 = axes[0, 0]
colors = plt.cm.tab10(np.linspace(0, 1, len(all_game_results)))
for idx, game in enumerate(all_game_results):
    ax1.scatter(game['real'], game['predicted'], alpha=0.6, s=50, 
               label=f'Game {idx+1}', color=colors[idx], edgecolors='k', linewidth=0.5)

min_val = min(all_real.min(), all_pred.min())
max_val = max(all_real.max(), all_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction')
ax1.set_xlabel('Real Gold Difference', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Gold Difference', fontsize=12, fontweight='bold')
ax1.set_title('All Test Games: Predicted vs Real', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

metrics_text = f'RMSE: {overall_rmse:.2f}\nMAE: {overall_mae:.2f}\nR²: {overall_r2:.4f}'
ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Plot 2: Error distribution across all games
ax2 = axes[0, 1]
errors = all_pred - all_real
ax2.hist(errors, bins=30, alpha=0.7, color='purple', edgecolor='black', linewidth=1)
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
bars = ax3.bar(game_labels, game_rmses, color='steelblue', edgecolor='k', linewidth=1.5, alpha=0.8)
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
            f'{rmse:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 4: Prediction quality over game progression
ax4 = axes[1, 1]
# Bin frames into early/mid/late game
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

plt.suptitle('Temporal LSTM Hierarchical Model - Comprehensive Test Evaluation', 
             fontsize=16, fontweight='bold', y=0.998)
plt.tight_layout()
plt.savefig('results/temporal_comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Comprehensive evaluation saved to: results/temporal_comprehensive_evaluation.png")
plt.close()

# Now create the individual game predictions plot (6 games)
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

plt.suptitle('Temporal LSTM Model - Individual Test Game Predictions', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/temporal_test_games.png', dpi=300, bbox_inches='tight')
print(f"✓ Individual game predictions saved to: results/temporal_test_games.png")
plt.close()

# Print summary table
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

print(f"\n{'='*60}")
print("EVALUATION COMPLETE")
print(f"{'='*60}")
print("\nGenerated files:")
print("  1. results/temporal_comprehensive_evaluation.png")
print("  2. results/temporal_test_games.png")

