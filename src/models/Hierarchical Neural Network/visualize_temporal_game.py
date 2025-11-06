#!/usr/bin/env python3
"""
Visualize trained temporal model predictions on a single game
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
print("LOADING TRAINED TEMPORAL MODEL")
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

# Load saved model checkpoint
checkpoint = torch.load('results/temporal_hierarchical_model.pth', weights_only=False)
sequence_length = checkpoint['sequence_length']
scaler_dict = checkpoint['scaler_dict']

print(f"✓ Model checkpoint loaded")
print(f"  Sequence length: {sequence_length}")

# Select a game with many frames
match_frame_counts = df_clean.groupby('match_id').size()
good_matches = match_frame_counts[match_frame_counts >= sequence_length + 10].index
selected_match = good_matches[15]  # Pick a match

print(f"\nSelected match: {selected_match}")

game_data = df_clean[df_clean['match_id'] == selected_match].copy().sort_values('frame_idx').reset_index(drop=True)
print(f"Number of frames: {len(game_data)}")

# Create dataset for this game
fg = FeatureGroups()
game_dataset = TemporalHierarchicalDataset(
    game_data, fg,
    sequence_length=sequence_length,
    stride=1,  # stride=1 to get prediction at every frame
    scaler_dict=scaler_dict,
    fit_scaler=False
)

print(f"Created {len(game_dataset)} sequences from this game")

# Load model
model = TemporalHierarchicalGoldPredictor(fg, hidden_dim=32, lstm_layers=1)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# Get predictions
print("\nGenerating predictions...")
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
        # Frame index corresponds to the LAST frame of the sequence
        frame_indices.append(game_data.iloc[idx + sequence_length - 1]['frame_idx'])

real_values = np.array(real_values)
predicted_values = np.array(predicted_values)
frame_indices = np.array(frame_indices)

# Calculate metrics
mse = np.mean((real_values - predicted_values) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(real_values - predicted_values))
r2 = 1 - (np.sum((real_values - predicted_values) ** 2) / np.sum((real_values - np.mean(real_values)) ** 2))

print(f"\n{'='*60}")
print(f"PREDICTION METRICS FOR MATCH {selected_match}")
print(f"{'='*60}")
print(f"RMSE: {rmse:.2f} gold")
print(f"MAE: {mae:.2f} gold")
print(f"R² Score: {r2:.4f}")
print(f"Mean Real Gold Diff: {np.mean(real_values):.2f}")
print(f"Mean Predicted Gold Diff: {np.mean(predicted_values):.2f}")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Real vs Predicted over time
ax1 = axes[0, 0]
ax1.plot(frame_indices, real_values, 'b-', label='Real Gold Difference', linewidth=2.5, marker='o', markersize=5, alpha=0.8)
ax1.plot(frame_indices, predicted_values, 'r--', label='Predicted Gold Difference', linewidth=2.5, marker='s', markersize=5, alpha=0.8)
ax1.fill_between(frame_indices, real_values, predicted_values, alpha=0.2, color='purple', label='Prediction Error')
ax1.axhline(y=0, color='k', linestyle=':', linewidth=1.5, alpha=0.5, label='Even Gold')
ax1.set_xlabel('Frame Index (Game Time)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Total Gold Difference', fontsize=13, fontweight='bold')
ax1.set_title(f'Match {selected_match}: Temporal LSTM Model\nReal vs Predicted Gold Difference Over Time', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Scatter - Predicted vs Real
ax2 = axes[0, 1]
ax2.scatter(real_values, predicted_values, alpha=0.7, s=80, edgecolors='k', linewidth=1, c=frame_indices, cmap='viridis')
min_val = min(real_values.min(), predicted_values.min())
max_val = max(real_values.max(), predicted_values.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction', alpha=0.8)
ax2.set_xlabel('Real Gold Difference', fontsize=13, fontweight='bold')
ax2.set_ylabel('Predicted Gold Difference', fontsize=13, fontweight='bold')
ax2.set_title('Predicted vs Real Values\n(color = game time)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
cbar = plt.colorbar(ax2.collections[0], ax=ax2)
cbar.set_label('Frame Index', fontsize=10)

# Add metrics text
metrics_text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.4f}'
ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Plot 3: Error over time
ax3 = axes[1, 0]
errors = predicted_values - real_values
ax3.plot(frame_indices, errors, 'g-', linewidth=2.5, marker='D', markersize=5, alpha=0.8)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
ax3.fill_between(frame_indices, 0, errors, alpha=0.3, color='green')
ax3.set_xlabel('Frame Index (Game Time)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Prediction Error (Predicted - Real)', fontsize=13, fontweight='bold')
ax3.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Add error statistics
error_text = f'Mean Error: {np.mean(errors):.2f}\nStd Error: {np.std(errors):.2f}'
ax3.text(0.05, 0.05, error_text, transform=ax3.transAxes, fontsize=11,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Plot 4: Error distribution
ax4 = axes[1, 1]
n, bins, patches = ax4.hist(errors, bins=15, alpha=0.7, color='purple', edgecolor='black', linewidth=1.5)
ax4.axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
ax4.axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2.5, label=f'Mean: {np.mean(errors):.2f}')
ax4.set_xlabel('Prediction Error (Gold)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax4.set_title('Error Distribution', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Temporal LSTM Hierarchical Model - Single Game Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/temporal_single_game_prediction.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: results/temporal_single_game_prediction.png")

# Print detailed predictions
print(f"\n{'='*60}")
print("SAMPLE PREDICTIONS")
print(f"{'='*60}")
print(f"{'Frame':<8} {'Real':>12} {'Predicted':>12} {'Error':>12} {'%Error':>10}")
print("-"*60)
for i in range(0, len(frame_indices), max(1, len(frame_indices)//15)):  # Show ~15 samples
    pct_error = 100 * errors[i] / (abs(real_values[i]) + 1) if abs(real_values[i]) > 100 else 0
    print(f"{frame_indices[i]:<8} {real_values[i]:>12.2f} {predicted_values[i]:>12.2f} {errors[i]:>12.2f} {pct_error:>9.1f}%")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"\nThe temporal LSTM model:")
print(f"  - Processes sequences of {sequence_length} frames")
print(f"  - Uses LSTM to capture trends in damage, vision, and player stats")
print(f"  - Predicts gold difference at each frame based on recent history")
print(f"  - Achieved RMSE of {rmse:.2f} gold on this match")

plt.show()

