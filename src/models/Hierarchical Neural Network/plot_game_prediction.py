#!/usr/bin/env python3
"""
Plot real vs predicted gold difference for a single game
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append('src')

# Import from NN module
import importlib.util
spec = importlib.util.spec_from_file_location("NN", "src/models/NN.py")
NN_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NN_module)

HierarchicalLoLDataset = NN_module.HierarchicalLoLDataset
HierarchicalGoldPredictor = NN_module.HierarchicalGoldPredictor
FeatureGroups = NN_module.FeatureGroups

print("="*60)
print("LOADING DATA AND MODEL")
print("="*60)

# Load the parquet data
print("Loading data...")
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

# Get one complete game (pick a match with many frames)
match_ids = df_clean['match_id'].unique()
match_frame_counts = df_clean.groupby('match_id').size()
# Pick a match with a good number of frames for visualization
selected_match = match_frame_counts[match_frame_counts > 20].index[10]  # Get 11th match with >20 frames

print(f"Selected match_id: {selected_match}")

game_data = df_clean[df_clean['match_id'] == selected_match].copy()
print(f"Number of frames in this game: {len(game_data)}")
print(f"Time range: frame {game_data['frame_idx'].min()} to {game_data['frame_idx'].max()}")

# Create feature groups and dataset
fg = FeatureGroups()
scaler_dict = {}
game_dataset = HierarchicalLoLDataset(game_data, fg, scaler_dict=scaler_dict, fit_scaler=True)

# Create model
print("\nCreating model...")
model = HierarchicalGoldPredictor(fg, hidden_dim=64)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Get predictions
print("\nGenerating predictions...")
model.eval()

real_values = []
predicted_values = []
frame_indices = []

with torch.no_grad():
    for idx in range(len(game_dataset)):
        batch = game_dataset[idx]
        
        # Prepare batch (add batch dimension)
        batch_input = {
            'x1': batch['x1'].unsqueeze(0),
            'x2': batch['x2'].unsqueeze(0),
            'players': {}
        }
        
        for p_key in batch['players'].keys():
            batch_input['players'][p_key] = {
                'g1': batch['players'][p_key]['g1'].unsqueeze(0),
                'g2': batch['players'][p_key]['g2'].unsqueeze(0),
                'g3': batch['players'][p_key]['g3'].unsqueeze(0),
                'g4': batch['players'][p_key]['g4'].unsqueeze(0),
                'g5': batch['players'][p_key]['g5'].unsqueeze(0)
            }
        
        # Get prediction
        pred = model(batch_input)
        
        # Denormalize
        target_denorm = batch['target'].item() * game_dataset.target_std + game_dataset.target_mean
        pred_denorm = pred.item() * game_dataset.target_std + game_dataset.target_mean
        
        real_values.append(target_denorm)
        predicted_values.append(pred_denorm)
        frame_indices.append(game_data.iloc[idx]['frame_idx'])

# Convert to arrays
real_values = np.array(real_values)
predicted_values = np.array(predicted_values)
frame_indices = np.array(frame_indices)

# Calculate metrics
mse = np.mean((real_values - predicted_values) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(real_values - predicted_values))

print(f"\n{'='*60}")
print(f"PREDICTION METRICS FOR MATCH {selected_match}")
print(f"{'='*60}")
print(f"RMSE: {rmse:.2f} gold")
print(f"MAE: {mae:.2f} gold")
print(f"Mean Real Gold Diff: {np.mean(real_values):.2f}")
print(f"Mean Predicted Gold Diff: {np.mean(predicted_values):.2f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Real vs Predicted over time
ax1 = axes[0, 0]
ax1.plot(frame_indices, real_values, 'b-', label='Real Gold Difference', linewidth=2, marker='o', markersize=4)
ax1.plot(frame_indices, predicted_values, 'r--', label='Predicted Gold Difference', linewidth=2, marker='s', markersize=4)
ax1.fill_between(frame_indices, real_values, predicted_values, alpha=0.3, color='gray', label='Prediction Error')
ax1.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
ax1.set_xlabel('Frame Index (time)', fontsize=12)
ax1.set_ylabel('Total Gold Difference', fontsize=12)
ax1.set_title(f'Match {selected_match}: Real vs Predicted Gold Difference Over Time', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Scatter - Predicted vs Real
ax2 = axes[0, 1]
ax2.scatter(real_values, predicted_values, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
min_val = min(real_values.min(), predicted_values.min())
max_val = max(real_values.max(), predicted_values.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Real Gold Difference', fontsize=12)
ax2.set_ylabel('Predicted Gold Difference', fontsize=12)
ax2.set_title('Predicted vs Real Values', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Add metrics text
metrics_text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}'
ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 3: Error over time
ax3 = axes[1, 0]
errors = predicted_values - real_values
ax3.plot(frame_indices, errors, 'g-', linewidth=2, marker='o', markersize=4)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
ax3.fill_between(frame_indices, 0, errors, alpha=0.3, color='green')
ax3.set_xlabel('Frame Index (time)', fontsize=12)
ax3.set_ylabel('Prediction Error (Predicted - Real)', fontsize=12)
ax3.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Error distribution
ax4 = axes[1, 1]
ax4.hist(errors, bins=20, alpha=0.7, color='purple', edgecolor='black')
ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
ax4.axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2, label=f'Mean Error: {np.mean(errors):.2f}')
ax4.set_xlabel('Prediction Error', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Error Distribution', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/single_game_prediction.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: results/single_game_prediction.png")

# Print some sample predictions
print(f"\n{'='*60}")
print("SAMPLE PREDICTIONS (First 10 frames)")
print(f"{'='*60}")
print(f"{'Frame':<8} {'Real':>12} {'Predicted':>12} {'Error':>12}")
print("-"*60)
for i in range(min(10, len(frame_indices))):
    print(f"{frame_indices[i]:<8} {real_values[i]:>12.2f} {predicted_values[i]:>12.2f} {errors[i]:>12.2f}")

print(f"\n{'='*60}")
print("VISUALIZATION COMPLETE")
print(f"{'='*60}")

plt.show()

