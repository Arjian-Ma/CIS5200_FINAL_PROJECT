#!/usr/bin/env python3
"""
Visualize autoregressive model predictions on test games
Shows multi-step ahead forecasts (predicting 5 frames into future)
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
collate_autoregressive = NN_module.collate_autoregressive

print("="*60)
print("AUTOREGRESSIVE MODEL - TEST GAME VISUALIZATION")
print("="*60)

# Load data
print("\nLoading data...")
df = pd.read_parquet("Data/processed/featured_data.parquet")

# Remove data leakage features (keep gold history for autoregressive)
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

# Load model checkpoint
checkpoint = torch.load('results/autoregressive_hierarchical_model.pth', weights_only=False)
scaler_dict = checkpoint['scaler_dict']
forecast_horizon = checkpoint['forecast_horizon']

print(f"✓ Model checkpoint loaded")
print(f"  Forecast horizon: {forecast_horizon} steps ahead")

# Create model
fg = FeatureGroups()
model = AutoregressiveHierarchicalPredictor(fg, hidden_dim=32, lstm_layers=1, forecast_horizon=forecast_horizon)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# Select 6 test games
print("\nSelecting test games...")
test_match_frame_counts = test_df.groupby('match_id').size()
good_test_matches = test_match_frame_counts[test_match_frame_counts >= 25].index

np.random.seed(100)
selected_matches = np.random.choice(good_test_matches, size=min(30, len(good_test_matches)), replace=False)

print(f"Selected {len(selected_matches)} test games")

# Get scaler parameters
target_mean = scaler_dict['target_mean']
target_std = scaler_dict['target_std']

# Evaluate each game
all_game_results = []

for match_idx, match_id in enumerate(selected_matches):
    print(f"Processing game {match_idx + 1}/{len(selected_matches)}: {match_id}...")
    
    game_data = test_df[test_df['match_id'] == match_id].copy().sort_values('frame_idx').reset_index(drop=True)
    
    real_trajectory = game_data['Total_Gold_Difference'].values
    frames = game_data['frame_idx'].values
    
    # Create dataset for this match
    single_match_df = test_df[test_df['match_id'] == match_id].copy()
    
    game_dataset = AutoregressiveGameDataset(
        single_match_df, fg,
        forecast_horizon=forecast_horizon,
        scaler_dict=scaler_dict,
        fit_scaler=False,
        min_frames=10
    )
    
    print(f"  Created {len(game_dataset)} forecast sequences")
    
    # Collect predictions at each timestep
    all_preds_at_t = []
    forecast_times = []
    
    with torch.no_grad():
        for idx in range(len(game_dataset)):
            batch = game_dataset[idx]
            
            # Add batch dimension
            batch_input = {
                'x1': batch['x1'].unsqueeze(0),
                'x2': batch['x2'].unsqueeze(0),
                'y_history': batch['y_history'].unsqueeze(0),
                'players': {},
                'seq_lens': torch.LongTensor([batch['seq_len']])
            }
            
            for p_key in batch['players'].keys():
                batch_input['players'][p_key] = {
                    g_key: batch['players'][p_key][g_key].unsqueeze(0)
                    for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']
                }
            
            # Predict next 5 steps
            pred = model(batch_input)  # [1, horizon]
            pred_denorm = pred[0].numpy() * target_std + target_mean
            
            all_preds_at_t.append(pred_denorm)
            forecast_times.append(batch['seq_len'])
    
    all_game_results.append({
        'match_id': match_id,
        'real': real_trajectory,
        'predictions': all_preds_at_t,  # List of 5-step forecasts
        'forecast_times': forecast_times,  # When each forecast was made
        'frames': frames
    })

# Create visualization - showing 1-step and 5-step ahead predictions
fig = plt.figure(figsize=(24, 20))

for idx, game in enumerate(all_game_results):
    ax = plt.subplot(6, 5, idx + 1)
    
    real = game['real']
    frames = game['frames']
    predictions = game['predictions']
    forecast_times = game['forecast_times']
    
    # Plot real trajectory
    ax.plot(frames, real, 'b-', label='Real Gold', linewidth=3, marker='o', markersize=4, alpha=0.8, zorder=3)
    
    # Plot 1-step ahead predictions
    pred_1step = []
    pred_1step_frames = []
    for i, preds in enumerate(predictions):
        if len(preds) >= 1:
            pred_1step.append(preds[0])  # First prediction (1 step ahead)
            pred_1step_frames.append(frames[forecast_times[i]])  # Frame where prediction was made
    
    if len(pred_1step) > 0:
        ax.plot(pred_1step_frames, pred_1step, 'r--', label='Predicted (1-step ahead)', 
               linewidth=2.5, marker='s', markersize=5, alpha=0.7, zorder=2)
    
    # Plot 5-step ahead predictions (if available)
    pred_5step = []
    pred_5step_frames = []
    for i, preds in enumerate(predictions):
        if len(preds) >= 5:
            pred_5step.append(preds[4])  # Fifth prediction (5 steps ahead)
            pred_5step_frames.append(frames[forecast_times[i]] if forecast_times[i] < len(frames) else frames[-1])
    
    if len(pred_5step) > 0:
        ax.plot(pred_5step_frames, pred_5step, 'g:', label='Predicted (5-step ahead)', 
               linewidth=2.5, marker='^', markersize=5, alpha=0.7, zorder=1)
    
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    
    # Calculate metrics for 1-step ahead
    if len(pred_1step) > 0:
        # Align predictions with real values
        aligned_real = []
        aligned_pred = []
        for i, frame_idx in enumerate(pred_1step_frames):
            frame_pos = np.where(frames == frame_idx)[0]
            if len(frame_pos) > 0:
                aligned_real.append(real[frame_pos[0]])
                aligned_pred.append(pred_1step[i])
        
        if len(aligned_real) > 0:
            rmse_1step = np.sqrt(np.mean((np.array(aligned_real) - np.array(aligned_pred)) ** 2))
        else:
            rmse_1step = 0
    else:
        rmse_1step = 0
    
    ax.set_xlabel('Frame', fontsize=8)
    ax.set_ylabel('Gold', fontsize=8)
    
    match_display = game['match_id'][:12] + '...' if len(game['match_id']) > 12 else game['match_id']
    ax.set_title(f'G{idx+1}: RMSE={rmse_1step:.0f}', 
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

plt.suptitle('Autoregressive LSTM Model: 30 Test Games - Multi-Step Forecasting (y_hat = Ax₁ + Bx₂ + Cx₃ + Dy_history)', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/autoregressive_test_games.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Multi-game visualization saved to: results/autoregressive_test_games.png")
plt.close()

# Create detailed summary for one game
if len(all_game_results) > 0:
    game = all_game_results[2]  # Pick game 3 for detailed analysis
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    real = game['real']
    frames = game['frames']
    predictions = game['predictions']
    forecast_times = game['forecast_times']
    
    # Plot 1: All forecast horizons
    ax1 = axes[0, 0]
    ax1.plot(frames, real, 'b-', label='Real Gold', linewidth=3, marker='o', markersize=6, alpha=0.9)
    
    colors = ['red', 'orange', 'yellow', 'green', 'purple']
    for h in range(min(5, forecast_horizon)):
        pred_h = []
        pred_h_frames = []
        for i, preds in enumerate(predictions):
            if len(preds) > h:
                pred_h.append(preds[h])
                # Prediction is for h+1 steps ahead
                target_frame_idx = forecast_times[i] + h
                if target_frame_idx < len(frames):
                    pred_h_frames.append(frames[target_frame_idx])
        
        if len(pred_h) > 0:
            ax1.plot(pred_h_frames, pred_h, '--', color=colors[h], 
                    label=f'{h+1}-step ahead', linewidth=2, marker='s', markersize=4, alpha=0.6)
    
    ax1.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Gold Difference', fontsize=12, fontweight='bold')
    ax1.set_title('All Forecast Horizons', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 1-step ahead scatter
    ax2 = axes[0, 1]
    pred_1step = [preds[0] for preds in predictions if len(preds) >= 1]
    real_1step = [real[forecast_times[i]] for i in range(len(predictions)) if len(predictions[i]) >= 1 and forecast_times[i] < len(real)]
    
    if len(pred_1step) > 0 and len(real_1step) > 0:
        min_len = min(len(pred_1step), len(real_1step))
        pred_1step = np.array(pred_1step[:min_len])
        real_1step = np.array(real_1step[:min_len])
        
        ax2.scatter(real_1step, pred_1step, alpha=0.6, s=80, edgecolors='k', linewidth=1, c=range(len(real_1step)), cmap='viridis')
        min_val = min(real_1step.min(), pred_1step.min())
        max_val = max(real_1step.max(), pred_1step.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect')
        
        rmse_1 = np.sqrt(np.mean((real_1step - pred_1step) ** 2))
        ax2.text(0.05, 0.95, f'RMSE: {rmse_1:.0f}', transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax2.set_xlabel('Real Gold (1-step ahead)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Gold (1-step ahead)', fontsize=12, fontweight='bold')
    ax2.set_title('1-Step Ahead Forecast Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Forecast error by horizon
    ax3 = axes[1, 0]
    horizon_rmses = []
    for h in range(forecast_horizon):
        pred_h = [preds[h] for preds in predictions if len(preds) > h]
        real_h = [real[forecast_times[i] + h] for i in range(len(predictions)) 
                 if len(predictions[i]) > h and forecast_times[i] + h < len(real)]
        
        if len(pred_h) > 0 and len(real_h) > 0:
            min_len = min(len(pred_h), len(real_h))
            rmse_h = np.sqrt(np.mean((np.array(real_h[:min_len]) - np.array(pred_h[:min_len])) ** 2))
            horizon_rmses.append(rmse_h)
        else:
            horizon_rmses.append(0)
    
    ax3.bar(range(1, len(horizon_rmses) + 1), horizon_rmses, color='steelblue', 
           edgecolor='k', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Forecast Horizon (steps ahead)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('RMSE (Gold)', fontsize=12, fontweight='bold')
    ax3.set_title('Forecast Error by Horizon', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(1, forecast_horizon + 1))
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, rmse in enumerate(horizon_rmses):
        ax3.text(i + 1, rmse, f'{rmse:.0f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # Plot 4: Example forecast trajectory
    ax4 = axes[1, 1]
    # Pick a moment in the game (middle)
    mid_idx = len(predictions) // 2
    if mid_idx < len(predictions):
        forecast_time = forecast_times[mid_idx]
        forecast_pred = predictions[mid_idx]
        
        # Real trajectory for next 5 frames
        real_next_5 = real[forecast_time:forecast_time + forecast_horizon]
        pred_next_5 = forecast_pred[:len(real_next_5)]
        time_indices = list(range(1, len(real_next_5) + 1))
        
        ax4.plot(time_indices, real_next_5, 'b-', label='Real', linewidth=3, marker='o', markersize=8, alpha=0.8)
        ax4.plot(time_indices, pred_next_5, 'r--', label='Forecasted', linewidth=3, marker='s', markersize=8, alpha=0.8)
        
        for i in range(len(time_indices)):
            ax4.plot([time_indices[i], time_indices[i]], [real_next_5[i], pred_next_5[i]], 
                    'gray', linewidth=1.5, alpha=0.5)
        
        ax4.set_xlabel('Steps Ahead', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Gold Difference', fontsize=12, fontweight='bold')
        ax4.set_title(f'Example Forecast at Frame {frames[forecast_time]}', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(time_indices)
    
    plt.suptitle(f'Detailed Analysis: Game {game["match_id"][:30]}', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('results/autoregressive_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Detailed analysis saved to: results/autoregressive_detailed_analysis.png")
    plt.close()

print(f"\n{'='*60}")
print("VISUALIZATION COMPLETE")
print(f"{'='*60}")
print("\nKey insights:")
print("  - Model forecasts 1-5 steps into the future")
print("  - Uses past gold differences (y_history) to capture momentum")
print("  - Architecture: y_hat = Ax₁ + Bx₂ + Cx₃ + Dy_history")
print("  - D component learns gold trends and mean reversion")
print("\nGenerated files:")
print("  1. results/autoregressive_test_games.png - 6 test games with forecasts")
print("  2. results/autoregressive_detailed_analysis.png - Detailed horizon analysis")

