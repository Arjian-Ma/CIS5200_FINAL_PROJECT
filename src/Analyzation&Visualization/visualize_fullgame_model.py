#!/usr/bin/env python3
"""
Visualize Full-Game Variable-Length LSTM model predictions
Shows real vs predicted for multiple test games
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
collate_variable_length = NN_module.collate_variable_length

print("="*60)
print("FULL-GAME VARIABLE-LENGTH MODEL EVALUATION")
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

# Select 6 diverse test games
print("\nSelecting test games...")
test_match_frame_counts = test_df.groupby('match_id').size()
good_test_matches = test_match_frame_counts[test_match_frame_counts >= 15].index

np.random.seed(100)
selected_matches = np.random.choice(good_test_matches, size=min(6, len(good_test_matches)), replace=False)

print(f"Selected {len(selected_matches)} test games")

# Evaluate each game
all_game_results = []
target_mean = scaler_dict['target_mean']
target_std = scaler_dict['target_std']

for match_idx, match_id in enumerate(selected_matches):
    print(f"Processing game {match_idx + 1}/{len(selected_matches)}: {match_id}...")
    
    game_data = test_df[test_df['match_id'] == match_id].copy().sort_values('frame_idx').reset_index(drop=True)
    
    # Get predictions for entire game
    real_values = game_data['Total_Gold_Difference'].values
    frames = game_data['frame_idx'].values
    
    # Create single-game dataset
    game_df_single = pd.DataFrame({col: [game_data] for col in [match_id]})
    game_df_single = game_data.to_frame().T if len(game_data) == 1 else game_data
    
    # Actually, just create dataset with this one match
    single_match_df = test_df[test_df['match_id'] == match_id].copy()
    
    game_dataset = FullGameTemporalDataset(
        single_match_df, fg,
        scaler_dict=scaler_dict,
        fit_scaler=False,
        min_frames=10
    )
    
    # Get prediction for the full game
    with torch.no_grad():
        if len(game_dataset) > 0:
            batch = game_dataset[0]
            
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
            
            # Predict (predicts final frame gold difference)
            pred = model(batch_input)
            pred_final = pred.item() * target_std + target_mean
            real_final = batch['target'].item() * target_std + target_mean
            
            all_game_results.append({
                'match_id': match_id,
                'real': real_values,
                'predicted_final': pred_final,
                'real_final': real_final,
                'frames': frames,
                'num_frames': len(frames)
            })

# Calculate statistics
all_real_finals = np.array([g['real_final'] for g in all_game_results])
all_pred_finals = np.array([g['predicted_final'] for g in all_game_results])

overall_rmse = np.sqrt(np.mean((all_real_finals - all_pred_finals) ** 2))
overall_mae = np.mean(np.abs(all_real_finals - all_pred_finals))
overall_r2 = 1 - (np.sum((all_real_finals - all_pred_finals) ** 2) / np.sum((all_real_finals - np.mean(all_real_finals)) ** 2))

print(f"\n{'='*60}")
print("OVERALL PERFORMANCE ON FINAL GOLD DIFFERENCE")
print(f"{'='*60}")
print(f"RMSE: {overall_rmse:.2f} gold")
print(f"MAE: {overall_mae:.2f} gold")
print(f"R² Score: {overall_r2:.4f}")

# Create visualization - Real trajectory vs Final Prediction
fig = plt.figure(figsize=(18, 12))

for idx, game in enumerate(all_game_results):
    ax = plt.subplot(2, 3, idx + 1)
    
    real_traj = game['real']
    frames = game['frames']
    pred_final = game['predicted_final']
    real_final = game['real_final']
    
    # Plot real trajectory
    ax.plot(frames, real_traj, 'b-', label='Real Gold Trajectory', linewidth=2.5, marker='o', markersize=5, alpha=0.8)
    
    # Plot predicted final as horizontal line
    ax.axhline(y=pred_final, color='r', linestyle='--', linewidth=3, label=f'Predicted Final: {pred_final:.0f}', alpha=0.8)
    ax.axhline(y=real_final, color='g', linestyle=':', linewidth=3, label=f'Real Final: {real_final:.0f}', alpha=0.8)
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    
    # Highlight the final frame
    ax.scatter([frames[-1]], [real_final], color='green', s=200, marker='*', edgecolors='k', linewidth=2, zorder=5, label='Game End')
    
    error = pred_final - real_final
    
    ax.set_xlabel('Frame Index', fontsize=11)
    ax.set_ylabel('Gold Difference', fontsize=11)
    
    match_display = game['match_id'][:20] + '...' if len(game['match_id']) > 20 else game['match_id']
    ax.set_title(f'Game {idx+1}: {match_display}\nFrames={game["num_frames"]}, Error={error:.0f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('Full-Game LSTM Model - Predicting Final Gold Difference', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/fullgame_model_predictions.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Game predictions saved to: results/fullgame_model_predictions.png")
plt.close()

# Create summary statistics plot
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Predicted vs Real Final Gold
ax1 = axes[0, 0]
colors = plt.cm.tab10(np.linspace(0, 1, len(all_game_results)))
for idx, game in enumerate(all_game_results):
    ax1.scatter(game['real_final'], game['predicted_final'], alpha=0.8, s=200, 
               label=f'Game {idx+1}', color=colors[idx], edgecolors='k', linewidth=2, marker='D')

min_val = min(all_real_finals.min(), all_pred_finals.min())
max_val = max(all_real_finals.max(), all_pred_finals.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction', alpha=0.8)
ax1.set_xlabel('Real Final Gold Difference', fontsize=13, fontweight='bold')
ax1.set_ylabel('Predicted Final Gold Difference', fontsize=13, fontweight='bold')
ax1.set_title('Final Gold Prediction: Predicted vs Real', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)

metrics_text = f'RMSE: {overall_rmse:.2f}\nMAE: {overall_mae:.2f}\nR²: {overall_r2:.4f}'
ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Plot 2: Error distribution
ax2 = axes[0, 1]
errors = all_pred_finals - all_real_finals
ax2.hist(errors, bins=10, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
ax2.axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
ax2.axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2.5, 
           label=f'Mean: {np.mean(errors):.2f}')
ax2.set_xlabel('Prediction Error (Gold)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax2.set_title('Final Gold Prediction Error Distribution', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Error by game length
ax3 = axes[1, 0]
game_lengths = [g['num_frames'] for g in all_game_results]
game_errors = [g['predicted_final'] - g['real_final'] for g in all_game_results]
ax3.scatter(game_lengths, game_errors, alpha=0.7, s=150, edgecolors='k', linewidth=2, c=game_lengths, cmap='viridis')
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
ax3.set_xlabel('Game Length (frames)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Prediction Error (Gold)', fontsize=13, fontweight='bold')
ax3.set_title('Error vs Game Length', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: Summary table
ax4 = axes[1, 1]
ax4.axis('off')

table_data = []
table_data.append(['Game', 'Frames', 'Real Final', 'Pred Final', 'Error'])
for idx, game in enumerate(all_game_results):
    error = game['predicted_final'] - game['real_final']
    table_data.append([
        f'{idx+1}',
        f'{game["num_frames"]}',
        f'{game["real_final"]:.0f}',
        f'{game["predicted_final"]:.0f}',
        f'{error:.0f}'
    ])

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.15, 0.15, 0.25, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax4.set_title('Game-by-Game Results Summary', fontsize=14, fontweight='bold', pad=20)

plt.suptitle('Full-Game Variable-Length LSTM Model - Test Evaluation', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/fullgame_model_evaluation.png', dpi=300, bbox_inches='tight')
print(f"✓ Evaluation summary saved to: results/fullgame_model_evaluation.png")
plt.close()

# Print detailed results
print(f"\n{'='*60}")
print("GAME-BY-GAME RESULTS")
print(f"{'='*60}")
print(f"{'Game':<6} {'Match ID':<25} {'Frames':<8} {'Real Final':>12} {'Pred Final':>12} {'Error':>12}")
print("-"*60)

for idx, game in enumerate(all_game_results):
    error = game['predicted_final'] - game['real_final']
    match_display = game['match_id'][:25]
    print(f"{idx+1:<6} {match_display:<25} {game['num_frames']:<8} {game['real_final']:>12.0f} {game['predicted_final']:>12.0f} {error:>12.0f}")

print(f"\n{'='*60}")
print("OVERALL STATISTICS")
print(f"{'='*60}")
print(f"Number of test games: {len(all_game_results)}")
print(f"RMSE on final gold: {overall_rmse:.2f} gold")
print(f"MAE on final gold: {overall_mae:.2f} gold")
print(f"R² Score: {overall_r2:.4f}")

print(f"\n{'='*60}")
print("MODEL CHARACTERISTICS")
print(f"{'='*60}")
print("This model:")
print("  - Processes ENTIRE GAME as one sequence (variable length)")
print("  - Uses pack_padded_sequence for efficient variable-length LSTM")
print("  - Predicts FINAL gold difference (who wins by how much)")
print("  - Each game = 1 sample (no overlapping windows)")
print("  - LSTM captures full game evolution: early → mid → late")

print(f"\n{'='*60}")
print("EVALUATION COMPLETE")
print(f"{'='*60}")
print("\nGenerated files:")
print("  1. results/fullgame_model_predictions.png - Individual game trajectories")
print("  2. results/fullgame_model_evaluation.png - Statistical summary")

