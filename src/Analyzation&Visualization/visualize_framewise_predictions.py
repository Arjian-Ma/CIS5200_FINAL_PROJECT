#!/usr/bin/env python3
"""
Frame-by-frame predictions for full games
Predict gold difference at EVERY timestep from t=0 to t=end
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

# Import from NN module
import importlib.util
spec = importlib.util.spec_from_file_location("NN", "src/models/Hierarchical Neural Network/NN.py")
NN_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NN_module)

FullGameTemporalDataset = NN_module.FullGameTemporalDataset
TemporalHierarchicalGoldPredictor = NN_module.TemporalHierarchicalGoldPredictor
FeatureGroups = NN_module.FeatureGroups

print("="*60)
print("FRAME-BY-FRAME PREDICTIONS FOR FULL GAMES")
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

# Split data
unique_matches = df_clean['match_id'].unique()
np.random.seed(42)
shuffled_matches = np.random.permutation(unique_matches)

n_train = int(len(unique_matches) * 0.7)
n_val = int(len(unique_matches) * 0.15)

test_matches = shuffled_matches[n_train + n_val:]
test_df = df_clean[df_clean['match_id'].isin(test_matches)].copy()

print(f"✓ Test set: {len(test_matches)} matches")

# Load model checkpoint
checkpoint = torch.load('results/temporal_hierarchical_model.pth', weights_only=False)
scaler_dict = checkpoint['scaler_dict']

print(f"✓ Model checkpoint loaded")

# Create model
fg = FeatureGroups()
model = TemporalHierarchicalGoldPredictor(fg, hidden_dim=32, lstm_layers=1)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# Select 6 test games
print("\nSelecting test games...")
test_match_frame_counts = test_df.groupby('match_id').size()
good_test_matches = test_match_frame_counts[test_match_frame_counts >= 15].index

np.random.seed(100)
selected_matches = np.random.choice(good_test_matches, size=min(6, len(good_test_matches)), replace=False)

print(f"Selected {len(selected_matches)} test games")

# Get scaler parameters
target_mean = scaler_dict['target_mean']
target_std = scaler_dict['target_std']

# Evaluate each game FRAME-BY-FRAME
all_game_results = []

for match_idx, match_id in enumerate(selected_matches):
    print(f"Processing game {match_idx + 1}/{len(selected_matches)}: {match_id}...")
    
    game_data = test_df[test_df['match_id'] == match_id].copy().sort_values('frame_idx').reset_index(drop=True)
    
    real_trajectory = game_data['Total_Gold_Difference'].values
    frames = game_data['frame_idx'].values
    num_frames = len(game_data)
    
    # Create dataset for this match
    single_match_df = test_df[test_df['match_id'] == match_id].copy()
    
    # Predict at each timestep t=0, t=1, ..., t=end
    predicted_trajectory = []
    
    with torch.no_grad():
        for t in range(num_frames):
            # Use frames 0 to t as input sequence
            partial_game = single_match_df.iloc[:t+1].copy()
            
            # Create dataset with this partial game
            partial_dataset = FullGameTemporalDataset(
                partial_game, fg,
                scaler_dict=scaler_dict,
                fit_scaler=False,
                min_frames=1  # Allow even single frames
            )
            
            if len(partial_dataset) > 0:
                batch = partial_dataset[0]
                
                # Add batch dimension
                batch_input = {
                    'x1': batch['x1'].unsqueeze(0),
                    'x2': batch['x2'].unsqueeze(0),
                    'players': {},
                    'seq_lens': torch.LongTensor([batch['seq_len']])
                }
                
                for p_key in batch['players'].keys():
                    batch_input['players'][p_key] = {
                        g_key: batch['players'][p_key][g_key].unsqueeze(0)
                        for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']
                    }
                
                # Predict current gold difference
                pred = model(batch_input)
                pred_denorm = pred.item() * target_std + target_mean
                predicted_trajectory.append(pred_denorm)
            else:
                predicted_trajectory.append(0.0)  # Fallback
    
    predicted_trajectory = np.array(predicted_trajectory)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((real_trajectory - predicted_trajectory) ** 2))
    mae = np.mean(np.abs(real_trajectory - predicted_trajectory))
    r2 = 1 - (np.sum((real_trajectory - predicted_trajectory) ** 2) / np.sum((real_trajectory - np.mean(real_trajectory)) ** 2))
    
    all_game_results.append({
        'match_id': match_id,
        'real': real_trajectory,
        'predicted': predicted_trajectory,
        'frames': frames,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    })
    
    print(f"  Frames: {num_frames}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

# Calculate overall statistics
all_real = np.concatenate([g['real'] for g in all_game_results])
all_pred = np.concatenate([g['predicted'] for g in all_game_results])

overall_rmse = np.sqrt(np.mean((all_real - all_pred) ** 2))
overall_mae = np.mean(np.abs(all_real - all_pred))
overall_r2 = 1 - (np.sum((all_real - all_pred) ** 2) / np.sum((all_real - np.mean(all_real)) ** 2))

print(f"\n{'='*60}")
print("OVERALL FRAME-BY-FRAME PERFORMANCE")
print(f"{'='*60}")
print(f"RMSE: {overall_rmse:.2f} gold")
print(f"MAE: {overall_mae:.2f} gold")
print(f"R² Score: {overall_r2:.4f}")
print(f"Total frame predictions: {len(all_real)}")

# Create visualization - Frame-by-frame predictions
fig = plt.figure(figsize=(18, 12))

for idx, game in enumerate(all_game_results):
    ax = plt.subplot(2, 3, idx + 1)
    
    real = game['real']
    pred = game['predicted']
    frames = game['frames']
    
    # Plot
    ax.plot(frames, real, 'b-', label='Real Gold', linewidth=2.5, marker='o', markersize=5, alpha=0.8)
    ax.plot(frames, pred, 'r--', label='Predicted Gold', linewidth=2.5, marker='s', markersize=5, alpha=0.8)
    ax.fill_between(frames, real, pred, alpha=0.2, color='purple', label='Error')
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    
    # Highlight final frame
    ax.scatter([frames[-1]], [real[-1]], color='green', s=250, marker='*', 
              edgecolors='k', linewidth=2, zorder=5, label='Final')
    
    ax.set_xlabel('Frame Index (time)', fontsize=11)
    ax.set_ylabel('Gold Difference', fontsize=11)
    
    match_display = game['match_id'][:18] + '...' if len(game['match_id']) > 18 else game['match_id']
    ax.set_title(f'Game {idx+1}: {match_display}\nRMSE={game["rmse"]:.0f}, MAE={game["mae"]:.0f}, R²={game["r2"]:.3f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('Full-Game LSTM - Frame-by-Frame Predictions (t=0 to t=end)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/fullgame_framewise_predictions.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Frame-by-frame predictions saved to: results/fullgame_framewise_predictions.png")
plt.close()

# Create comprehensive summary
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Scatter - all frames
ax1 = axes[0, 0]
colors = plt.cm.tab10(np.linspace(0, 1, len(all_game_results)))
for idx, game in enumerate(all_game_results):
    ax1.scatter(game['real'], game['predicted'], alpha=0.6, s=50, 
               label=f'Game {idx+1}', color=colors[idx], edgecolors='k', linewidth=0.5)

min_val = min(all_real.min(), all_pred.min())
max_val = max(all_real.max(), all_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect')
ax1.set_xlabel('Real Gold Difference', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Gold Difference', fontsize=12, fontweight='bold')
ax1.set_title('All Frames: Predicted vs Real', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, ncol=2)
ax1.grid(True, alpha=0.3)

metrics_text = f'RMSE: {overall_rmse:.2f}\nMAE: {overall_mae:.2f}\nR²: {overall_r2:.4f}'
ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Plot 2: Error distribution
ax2 = axes[0, 1]
errors = all_pred - all_real
ax2.hist(errors, bins=40, alpha=0.7, color='coral', edgecolor='black', linewidth=1)
ax2.axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
ax2.axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2.5, label=f'Mean: {np.mean(errors):.2f}')
ax2.set_xlabel('Prediction Error (Gold)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Error Distribution (All Frames)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: RMSE per game
ax3 = axes[1, 0]
game_rmses = [g['rmse'] for g in all_game_results]
game_labels = [f"Game {i+1}" for i in range(len(all_game_results))]
bars = ax3.bar(game_labels, game_rmses, color='steelblue', edgecolor='k', linewidth=1.5, alpha=0.8)
ax3.axhline(y=overall_rmse, color='r', linestyle='--', linewidth=2.5, label=f'Average: {overall_rmse:.2f}')
ax3.set_xlabel('Test Game', fontsize=12, fontweight='bold')
ax3.set_ylabel('RMSE (Gold)', fontsize=12, fontweight='bold')
ax3.set_title('RMSE by Game (Frame-by-Frame)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

for bar, rmse in zip(bars, game_rmses):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{rmse:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: Error evolution over time
ax4 = axes[1, 1]
all_frames = np.concatenate([g['frames'] for g in all_game_results])
all_errors_abs = np.abs(errors)

# Bin by frame index
early_mask = all_frames < 10
mid_mask = (all_frames >= 10) & (all_frames < 20)
late_mask = all_frames >= 20

early_mae = np.mean(all_errors_abs[early_mask]) if early_mask.sum() > 0 else 0
mid_mae = np.mean(all_errors_abs[mid_mask]) if mid_mask.sum() > 0 else 0
late_mae = np.mean(all_errors_abs[late_mask]) if late_mask.sum() > 0 else 0

phases = ['Early\n(0-10)', 'Mid\n(10-20)', 'Late\n(20+)']
maes = [early_mae, mid_mae, late_mae]
colors_phase = ['green', 'orange', 'red']

bars = ax4.bar(phases, maes, color=colors_phase, edgecolor='k', linewidth=1.5, alpha=0.7)
ax4.set_ylabel('Mean Absolute Error (Gold)', fontsize=12, fontweight='bold')
ax4.set_title('Prediction Error by Game Phase', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for bar, mae_val in zip(bars, maes):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{mae_val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Full-Game LSTM: Frame-by-Frame Predictions (t=0 to t=end)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/fullgame_framewise_evaluation.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Frame-wise evaluation saved to: results/fullgame_framewise_evaluation.png")
plt.close()

# Print results
print(f"\n{'='*60}")
print("GAME-BY-GAME RESULTS (Frame-by-Frame)")
print(f"{'='*60}")
print(f"{'Game':<6} {'Match ID':<25} {'Frames':<8} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
print("-"*60)

for idx, game in enumerate(all_game_results):
    match_display = game['match_id'][:25]
    print(f"{idx+1:<6} {match_display:<25} {len(game['frames']):<8} {game['rmse']:>10.2f} {game['mae']:>10.2f} {game['r2']:>8.4f}")

print(f"\n{'='*60}")
print("OVERALL STATISTICS")
print(f"{'='*60}")
print(f"Total frame predictions: {len(all_real)}")
print(f"Average RMSE: {overall_rmse:.2f} gold")
print(f"Average MAE: {overall_mae:.2f} gold")
print(f"Average R²: {overall_r2:.4f}")

print(f"\n{'='*60}")
print("WHAT THIS SHOWS")
print(f"{'='*60}")
print("For each game, the model:")
print("  - At t=0: Predicts gold diff using frame 0 only")
print("  - At t=5: Predicts gold diff using frames 0-5")
print("  - At t=15: Predicts gold diff using frames 0-15")
print("  - At t=end: Predicts final gold using entire game history")
print("\nThe LSTM processes the growing sequence at each timestep,")
print("capturing how the game evolved up to that point!")

print(f"\n{'='*60}")
print("EVALUATION COMPLETE")
print(f"{'='*60}")
print("\nGenerated files:")
print("  1. results/fullgame_framewise_predictions.png - 6 game trajectories")
print("  2. results/fullgame_framewise_evaluation.png - Statistical summary")



