#!/usr/bin/env python3
"""
Visualize traditional hierarchical LSTM model predictions on 30 test games
Model: y_hat = Ax_1 + Bx_2 + Cx_3 (NO autoregressive component)
Uses the model trained in NN.py (150 epochs)
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

# Import from NN.py module
import importlib.util
spec = importlib.util.spec_from_file_location("NN", "src/models/Hierarchical Neural Network/NN.py")
NN_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NN_module)

FullGameTemporalDataset = NN_module.FullGameTemporalDataset
TemporalHierarchicalGoldPredictor = NN_module.TemporalHierarchicalGoldPredictor
FeatureGroups = NN_module.FeatureGroups
collate_variable_length = NN_module.collate_variable_length

print("="*60)
print("TRADITIONAL HIERARCHICAL LSTM MODEL - TEST GAMES")
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

# Load model checkpoint (150 epoch model)
checkpoint_path = 'results/Model_20251027_150Epoch/temporal_hierarchical_model.pth'
print(f"\nLoading model from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, weights_only=False)
scaler_dict = checkpoint['scaler_dict']

print(f"✓ Model checkpoint loaded")

# Create model
fg = FeatureGroups()
model = TemporalHierarchicalGoldPredictor(fg, hidden_dim=32, lstm_layers=1)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# Select 30 test games
print("\nSelecting test games...")
test_match_frame_counts = test_df.groupby('match_id').size()
good_test_matches = test_match_frame_counts[test_match_frame_counts >= 25].index

np.random.seed(150)
selected_matches = np.random.choice(good_test_matches, size=min(30, len(good_test_matches)), replace=False)

print(f"Selected {len(selected_matches)} test games")

# Get scaler parameters
target_mean = scaler_dict['target_mean']
target_std = scaler_dict['target_std']

# Evaluate each game
all_game_results = []

for match_idx, match_id in enumerate(selected_matches):
    print(f"Processing game {match_idx + 1}/30: {match_id}...")
    
    game_data = test_df[test_df['match_id'] == match_id].copy().sort_values('frame_idx').reset_index(drop=True)
    
    real_trajectory = game_data['Total_Gold_Difference'].values
    frames = game_data['frame_idx'].values
    
    # Create dataset for this match
    single_match_df = test_df[test_df['match_id'] == match_id].copy()
    
    game_dataset = FullGameTemporalDataset(
        single_match_df, fg,
        scaler_dict=scaler_dict,
        fit_scaler=False,
        min_frames=10
    )
    
    if len(game_dataset) == 0:
        print(f"  Skipping - no valid sequences")
        continue
    
    print(f"  Game length: {len(game_data)} frames")
    
    # Make predictions frame by frame
    predictions = []
    
    with torch.no_grad():
        for frame_idx in range(len(game_data)):
            # Use all frames up to current frame
            partial_data = game_data.iloc[:frame_idx+1].copy()
            
            if len(partial_data) < 10:
                predictions.append(np.nan)
                continue
            
            # Create dataset with partial data
            partial_match_df = pd.DataFrame(partial_data)
            partial_match_df['match_id'] = match_id
            
            partial_dataset = FullGameTemporalDataset(
                partial_match_df, fg,
                scaler_dict=scaler_dict,
                fit_scaler=False,
                min_frames=1
            )
            
            if len(partial_dataset) == 0:
                predictions.append(np.nan)
                continue
            
            # Get the batch
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
            
            # Predict
            pred = model(batch_input)
            pred_denorm = pred[0].item() * target_std + target_mean
            predictions.append(pred_denorm)
    
    predictions = np.array(predictions)
    
    # Calculate RMSE for valid predictions
    valid_mask = ~np.isnan(predictions)
    if valid_mask.sum() > 0:
        rmse = np.sqrt(np.mean((real_trajectory[valid_mask] - predictions[valid_mask]) ** 2))
        mae = np.mean(np.abs(real_trajectory[valid_mask] - predictions[valid_mask]))
    else:
        rmse = np.nan
        mae = np.nan
    
    all_game_results.append({
        'match_id': match_id,
        'real': real_trajectory,
        'predicted': predictions,
        'frames': frames,
        'rmse': rmse,
        'mae': mae
    })
    
    print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Create visualization - 30 games in 6x5 grid
fig = plt.figure(figsize=(24, 20))

for idx, game in enumerate(all_game_results):
    ax = plt.subplot(6, 5, idx + 1)
    
    real = game['real']
    pred = game['predicted']
    frames = game['frames']
    
    # Plot real trajectory
    ax.plot(frames, real, 'b-', label='Real Gold', linewidth=2.5, marker='o', markersize=4, alpha=0.8, zorder=3)
    
    # Plot predicted trajectory (only valid predictions)
    valid_mask = ~np.isnan(pred)
    if valid_mask.sum() > 0:
        ax.plot(frames[valid_mask], pred[valid_mask], 'r--', label='Predicted', 
               linewidth=2.5, marker='s', markersize=4, alpha=0.7, zorder=2)
    
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Frame', fontsize=8)
    ax.set_ylabel('Gold', fontsize=8)
    
    match_display = game['match_id'][:12] + '...' if len(game['match_id']) > 12 else game['match_id']
    ax.set_title(f'G{idx+1}: RMSE={game["rmse"]:.0f}', 
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

plt.suptitle('Traditional Hierarchical LSTM Model: 30 Test Games (y_hat = Ax₁ + Bx₂ + Cx₃) - 150 Epochs', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/traditional_nn_test_games.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: results/traditional_nn_test_games.png")
plt.close()

# Print summary statistics
print(f"\n{'='*60}")
print("TEST GAME SUMMARY")
print(f"{'='*60}")
print(f"{'Game':<6} {'Match ID':<25} {'Frames':<8} {'RMSE':>10} {'MAE':>10}")
print("-"*60)

for idx, game in enumerate(all_game_results):
    match_display = game['match_id'][:25]
    total = len(game['real'])
    rmse_val = game['rmse'] if not np.isnan(game['rmse']) else 0
    mae_val = game['mae'] if not np.isnan(game['mae']) else 0
    print(f"{idx+1:<6} {match_display:<25} {total:<8} {rmse_val:>10.2f} {mae_val:>10.2f}")

# Overall statistics
valid_games = [g for g in all_game_results if not np.isnan(g['rmse'])]

if len(valid_games) > 0:
    all_rmses = [g['rmse'] for g in valid_games]
    all_maes = [g['mae'] for g in valid_games]
    
    print(f"\n{'='*60}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Average RMSE: {np.mean(all_rmses):.2f} gold")
    print(f"Average MAE: {np.mean(all_maes):.2f} gold")
    print(f"Best RMSE: {np.min(all_rmses):.2f} gold")
    print(f"Worst RMSE: {np.max(all_rmses):.2f} gold")
    print(f"Std Dev RMSE: {np.std(all_rmses):.2f} gold")

print(f"\n{'='*60}")
print("MODEL ARCHITECTURE")
print(f"{'='*60}")
print("Formula: y_hat = Ax₁ + Bx₂ + Cx₃")
print("\nComponents:")
print("  A: Damage LSTM (6 damage features)")
print("  B: Vision LSTM (3 vision features)")
print("  C: Team LSTM (hierarchical player aggregation)")
print("     - 10 players × 5 stat groups")
print("     - G1: Offensive stats (10 features)")
print("     - G2: Defensive stats (4 features)")
print("     - G3: Vamp stats (4 features)")
print("     - G4: Resource stats (2 features)")
print("     - G5: Mobility stats (3 features)")
print("\nTemporal Processing:")
print("  - Bidirectional LSTM for each component")
print("  - Processes full game as variable-length sequence")
print("  - No autoregressive component (pure exogenous)")

print(f"\n{'='*60}")
print("EVALUATION COMPLETE")
print(f"{'='*60}")
print("\nGenerated file:")
print("  - results/traditional_nn_test_games.png")

