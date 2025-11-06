#!/usr/bin/env python3
"""
Predict the last 5 frames of test games using autoregressive model
Shows: Can the model predict the endgame outcome given early/mid-game data?
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

# Import from NN_autoregressive module
import importlib.util
spec = importlib.util.spec_from_file_location("NN_auto", "src/models/NN_autoregressive.py")
NN_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NN_module)

AutoregressiveGameDataset = NN_module.AutoregressiveGameDataset
AutoregressiveHierarchicalPredictor = NN_module.AutoregressiveHierarchicalPredictor
FeatureGroups = NN_module.FeatureGroups

print("="*60)
print("ENDGAME PREDICTION: Last 5 Frames Forecast")
print("="*60)

# Load data
print("\nLoading data...")
df = pd.read_parquet("Data/processed/featured_data.parquet")

# Remove data leakage features
leakage_features = [
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

# Load model
checkpoint = torch.load('results/autoregressive_hierarchical_model.pth', weights_only=False)
scaler_dict = checkpoint['scaler_dict']
forecast_horizon = checkpoint['forecast_horizon']

fg = FeatureGroups()
model = AutoregressiveHierarchicalPredictor(fg, hidden_dim=32, lstm_layers=1, forecast_horizon=forecast_horizon)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

# Select test games with sufficient length
test_match_frame_counts = test_df.groupby('match_id').size()
good_test_matches = test_match_frame_counts[test_match_frame_counts >= 20].index

np.random.seed(200)
selected_matches = np.random.choice(good_test_matches, size=min(30, len(good_test_matches)), replace=False)

print(f"Selected {len(selected_matches)} test games")

# Get scaler parameters
target_mean = scaler_dict['target_mean']
target_std = scaler_dict['target_std']

# Evaluate each game - predict last 5 frames
all_game_results = []

for match_idx, match_id in enumerate(selected_matches):
    print(f"\nProcessing game {match_idx + 1}/{len(selected_matches)}: {match_id}...")
    
    game_data = test_df[test_df['match_id'] == match_id].copy().sort_values('frame_idx').reset_index(drop=True)
    
    total_frames = len(game_data)
    print(f"  Total frames: {total_frames}")
    
    if total_frames < 15:
        print(f"  Skipping - too short")
        continue
    
    # Use all frames EXCEPT last 5 as input
    cutoff_frame = total_frames - 5
    input_data = game_data.iloc[:cutoff_frame].copy()
    target_data = game_data.iloc[cutoff_frame:].copy()
    
    print(f"  Using frames 0-{cutoff_frame-1} to predict frames {cutoff_frame}-{total_frames-1}")
    
    # Create dataset with input data only
    input_match_df = input_data.copy()
    input_match_df['match_id'] = match_id  # Ensure match_id is set
    
    # We need to manually extract features since we're only using partial game
    fg_obj = FeatureGroups()
    
    # Extract features
    x1_seq = input_data[fg_obj.damage_features].values
    x2_seq = input_data[fg_obj.vision_features].values
    y_history = input_data['Total_Gold_Difference'].values
    
    # Normalize
    def normalize_feature(data, key):
        mean = scaler_dict.get(f'{key}_mean', 0)
        std = scaler_dict.get(f'{key}_std', 1)
        return (data - mean) / std
    
    x1_norm = normalize_feature(x1_seq, 'x1')
    x2_norm = normalize_feature(x2_seq, 'x2')
    y_hist_norm = normalize_feature(y_history, 'y_history')
    
    # Extract player features
    player_features = {}
    for p_idx in range(1, 11):
        player_features[f'p{p_idx}'] = {}
        
        for g_key, g_features in [
            ('g1', fg_obj.offensive_stats),
            ('g2', fg_obj.defensive_stats),
            ('g3', fg_obj.vamp_stats),
            ('g4', fg_obj.resource_stats),
            ('g5', fg_obj.mobility_stats)
        ]:
            cols = [f'Player{p_idx}_{stat}' for stat in g_features]
            g_data = input_data[cols].values
            g_norm = normalize_feature(g_data, f'p{p_idx}_{g_key}')
            player_features[f'p{p_idx}'][g_key] = torch.FloatTensor(g_norm)
    
    # Create batch
    batch_input = {
        'x1': torch.FloatTensor(x1_norm).unsqueeze(0),  # [1, seq_len, 6]
        'x2': torch.FloatTensor(x2_norm).unsqueeze(0),  # [1, seq_len, 3]
        'y_history': torch.FloatTensor(y_hist_norm).unsqueeze(0),  # [1, seq_len]
        'players': {},
        'seq_lens': torch.LongTensor([len(input_data)])
    }
    
    for p_key in player_features.keys():
        batch_input['players'][p_key] = {
            g_key: player_features[p_key][g_key].unsqueeze(0)
            for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']
        }
    
    # Predict last 5 frames
    with torch.no_grad():
        predictions = model(batch_input)  # [1, 5]
        predictions_denorm = predictions[0].numpy() * target_std + target_mean
    
    # Get real last 5 frames
    real_last_5 = target_data['Total_Gold_Difference'].values
    frames_last_5 = target_data['frame_idx'].values
    
    # Calculate metrics
    min_len = min(len(real_last_5), len(predictions_denorm))
    rmse = np.sqrt(np.mean((real_last_5[:min_len] - predictions_denorm[:min_len]) ** 2))
    mae = np.mean(np.abs(real_last_5[:min_len] - predictions_denorm[:min_len]))
    
    all_game_results.append({
        'match_id': match_id,
        'full_real': game_data['Total_Gold_Difference'].values,
        'full_frames': game_data['frame_idx'].values,
        'cutoff_frame': cutoff_frame,
        'predicted_last_5': predictions_denorm,
        'real_last_5': real_last_5,
        'frames_last_5': frames_last_5,
        'rmse': rmse,
        'mae': mae
    })
    
    print(f"  Predicted last 5 frames - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Create visualization - 30 games in 6x5 grid
fig = plt.figure(figsize=(24, 20))

for idx, game in enumerate(all_game_results):
    ax = plt.subplot(6, 5, idx + 1)
    
    full_real = game['full_real']
    full_frames = game['full_frames']
    cutoff = game['cutoff_frame']
    pred_last_5 = game['predicted_last_5']
    real_last_5 = game['real_last_5']
    frames_last_5 = game['frames_last_5']
    
    # Plot full real trajectory
    ax.plot(full_frames[:cutoff], full_real[:cutoff], 'b-', 
           label='Known History', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
    
    # Plot real last 5 frames (ground truth)
    ax.plot(frames_last_5, real_last_5, 'g-', 
           label='Real Endgame', linewidth=3, marker='*', markersize=10, alpha=0.9, zorder=5)
    
    # Plot predicted last 5 frames
    ax.plot(frames_last_5[:len(pred_last_5)], pred_last_5, 'r--', 
           label='Predicted Endgame', linewidth=3, marker='D', markersize=8, alpha=0.8, zorder=4)
    
    # Vertical line at cutoff
    ax.axvline(x=full_frames[cutoff-1], color='orange', linestyle=':', linewidth=2.5, 
              label=f'Forecast Point (t={full_frames[cutoff-1]})', alpha=0.8)
    
    # Shade the forecast region
    if len(frames_last_5) > 0:
        ax.axvspan(frames_last_5[0], frames_last_5[-1], alpha=0.1, color='yellow')
    
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Frame', fontsize=8)
    ax.set_ylabel('Gold Diff', fontsize=8)
    
    match_display = game['match_id'][:12] + '...' if len(game['match_id']) > 12 else game['match_id']
    ax.set_title(f'G{idx+1}: {match_display}\nRMSE={game["rmse"]:.0f}', 
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

plt.suptitle('Endgame Prediction: 30 Test Games - Using Early/Mid Game to Forecast Last 5 Frames', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/autoregressive_endgame_predictions.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Endgame predictions saved to: results/autoregressive_endgame_predictions.png")
plt.close()

# Create detailed comparison for one game
if len(all_game_results) > 0:
    game = all_game_results[0]  # Pick first game for detailed view
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Full trajectory with forecast region highlighted
    ax1 = axes[0, 0]
    ax1.plot(game['full_frames'], game['full_real'], 'b-', linewidth=2, marker='o', 
            markersize=4, alpha=0.6, label='Full Real Trajectory')
    
    cutoff = game['cutoff_frame']
    ax1.axvline(x=game['full_frames'][cutoff-1], color='orange', linestyle='--', linewidth=3, 
               label=f'Forecast Point (t={game["full_frames"][cutoff-1]})')
    
    # Highlight forecast region
    ax1.axvspan(game['frames_last_5'][0], game['frames_last_5'][-1], alpha=0.2, color='yellow', 
               label='Forecast Region')
    
    # Plot predictions
    ax1.plot(game['frames_last_5'][:len(game['predicted_last_5'])], game['predicted_last_5'], 
            'r--', linewidth=3, marker='D', markersize=8, label='Predicted', zorder=5)
    ax1.plot(game['frames_last_5'], game['real_last_5'], 'g-', 
            linewidth=3, marker='*', markersize=12, label='Real', zorder=6)
    
    ax1.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Gold Difference', fontsize=12, fontweight='bold')
    ax1.set_title(f'Full Game: {game["match_id"][:30]}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoom into forecast region
    ax2 = axes[0, 1]
    step_indices = list(range(1, len(game['real_last_5']) + 1))
    
    ax2.plot(step_indices, game['real_last_5'], 'g-', 
            label='Real Endgame', linewidth=3.5, marker='*', markersize=12, alpha=0.9)
    ax2.plot(step_indices[:len(game['predicted_last_5'])], game['predicted_last_5'], 
            'r--', label='Predicted', linewidth=3.5, marker='D', markersize=10, alpha=0.8)
    
    # Draw error bars
    for i in range(min(len(game['real_last_5']), len(game['predicted_last_5']))):
        ax2.plot([step_indices[i], step_indices[i]], 
                [game['real_last_5'][i], game['predicted_last_5'][i]], 
                'gray', linewidth=2, alpha=0.5)
    
    ax2.axhline(y=0, color='k', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Steps into Forecast', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Gold Difference', fontsize=12, fontweight='bold')
    ax2.set_title(f'Endgame Forecast Detail (Last {len(game["real_last_5"])} Frames)', 
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(step_indices)
    
    # Add RMSE annotation
    ax2.text(0.5, 0.95, f'RMSE: {game["rmse"]:.0f} gold\nMAE: {game["mae"]:.0f} gold', 
            transform=ax2.transAxes, fontsize=11, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 3: Predicted vs Real (scatter)
    ax3 = axes[1, 0]
    min_len = min(len(game['real_last_5']), len(game['predicted_last_5']))
    
    ax3.scatter(game['real_last_5'][:min_len], game['predicted_last_5'][:min_len], 
               alpha=0.8, s=150, edgecolors='k', linewidth=2, c=range(min_len), cmap='viridis')
    
    min_val = min(game['real_last_5'].min(), game['predicted_last_5'].min())
    max_val = max(game['real_last_5'].max(), game['predicted_last_5'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction')
    
    ax3.set_xlabel('Real Gold Difference', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Predicted Gold Difference', fontsize=12, fontweight='bold')
    ax3.set_title('Predicted vs Real (Last 5 Frames)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar for time
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Step into Forecast', fontsize=10)
    
    # Plot 4: Forecast errors by step
    ax4 = axes[1, 1]
    errors = game['predicted_last_5'][:min_len] - game['real_last_5'][:min_len]
    
    ax4.bar(step_indices[:min_len], errors, color='coral', edgecolor='k', linewidth=1.5, alpha=0.8)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2.5)
    ax4.set_xlabel('Steps into Forecast', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Prediction Error (Gold)', fontsize=12, fontweight='bold')
    ax4.set_title('Forecast Error by Step', fontsize=13, fontweight='bold')
    ax4.set_xticks(step_indices[:min_len])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, err in enumerate(errors):
        ax4.text(step_indices[i], err, f'{err:.0f}', ha='center', 
                va='bottom' if err > 0 else 'top', fontsize=9, fontweight='bold')
    
    plt.suptitle(f'Detailed Endgame Forecast Analysis: {game["match_id"][:40]}', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('results/autoregressive_endgame_detailed.png', dpi=300, bbox_inches='tight')
    print(f"✓ Detailed endgame analysis saved to: results/autoregressive_endgame_detailed.png")
    plt.close()

# Print summary statistics
print(f"\n{'='*60}")
print("ENDGAME PREDICTION SUMMARY")
print(f"{'='*60}")
print(f"{'Game':<6} {'Match ID':<25} {'Frames':<8} {'Cutoff':<8} {'RMSE':>10} {'MAE':>10}")
print("-"*60)

for idx, game in enumerate(all_game_results):
    match_display = game['match_id'][:25]
    total = len(game['full_real'])
    cutoff = game['cutoff_frame']
    print(f"{idx+1:<6} {match_display:<25} {total:<8} {cutoff:<8} {game['rmse']:>10.2f} {game['mae']:>10.2f}")

# Overall statistics
if len(all_game_results) > 0:
    all_rmses = [g['rmse'] for g in all_game_results]
    all_maes = [g['mae'] for g in all_game_results]
    
    print(f"\n{'='*60}")
    print("OVERALL ENDGAME FORECAST PERFORMANCE")
    print(f"{'='*60}")
    print(f"Average RMSE: {np.mean(all_rmses):.2f} gold")
    print(f"Average MAE: {np.mean(all_maes):.2f} gold")
    print(f"Best RMSE: {np.min(all_rmses):.2f} gold")
    print(f"Worst RMSE: {np.max(all_rmses):.2f} gold")

print(f"\n{'='*60}")
print("WHAT THIS SHOWS")
print(f"{'='*60}")
print("For each game:")
print("  1. Model sees frames 0 to (N-5)")
print("  2. Model predicts frames (N-4), (N-3), (N-2), (N-1), N")
print("  3. Shows if model can forecast endgame outcome from mid-game state")
print("\nYellow shaded region = Forecast zone (unknown to model)")
print("Orange line = Last known frame before forecasting")

print(f"\n{'='*60}")
print("EVALUATION COMPLETE")
print(f"{'='*60}")
print("\nGenerated files:")
print("  1. results/autoregressive_endgame_predictions.png - 6 games with endgame forecasts")
print("  2. results/autoregressive_endgame_detailed.png - Detailed analysis of one game")

