#!/usr/bin/env python3
"""
Analyze RMSE distribution across ALL test games
Supports multiple model types (LSTM, Autoregressive, etc.)
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import argparse
import os
sys.path.append('src')

from models.lstm_model import LSTM, get_specified_features

print("="*60)
print("ANALYZING ALL TEST GAMES")
print("="*60)

# Parse arguments
parser = argparse.ArgumentParser(description='Analyze test games with different models')
parser.add_argument('--model_path', type=str, default='models/lstm_model.pth',
                    help='Path to model checkpoint (default: models/lstm_model.pth)')
parser.add_argument('--model_type', type=str, default='auto',
                    choices=['auto', 'lstm', 'autoregressive'],
                    help='Model type: auto (detect from checkpoint), lstm, or autoregressive')
parser.add_argument('--data_path', type=str, default='data/processed/featured_data_with_scores.parquet',
                    help='Path to featured data (default: data/processed/featured_data_with_scores.parquet)')
parser.add_argument('--feature_list', type=str, default=None,
                    help='Path to CSV file with feature list, or "specified" to use get_specified_features()')
parser.add_argument('--target_col', type=str, default='Total_Gold_Difference',
                    help='Target column name (default: Total_Gold_Difference)')
parser.add_argument('--sequence_length', type=int, default=15,
                    help='Sequence length for LSTM models (default: 15)')
parser.add_argument('--min_game_length', type=int, default=None,
                    help='Minimum number of timestamps required for a game to be processed (default: sequence_length)')
parser.add_argument('--output_path', type=str, default=None,
                    help='Output path for plot (default: auto-generated based on model name)')

args = parser.parse_args()

# Set min_game_length default to sequence_length if not provided
if args.min_game_length is None:
    args.min_game_length = args.sequence_length

# Load data
print(f"\nLoading data from {args.data_path}...")
if args.data_path.endswith('.parquet'):
    df = pd.read_parquet(args.data_path)
else:
    df = pd.read_csv(args.data_path)

print(f"✓ Loaded {len(df)} rows")

# Load model checkpoint
print(f"\nLoading model from {args.model_path}...")
checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
print(f"✓ Checkpoint loaded")

# Detect model type if auto
if args.model_type == 'auto':
    # Check checkpoint keys to determine model type
    if 'forecast_horizon' in checkpoint or 'scaler_dict' in checkpoint:
        args.model_type = 'autoregressive'
    elif 'input_size' in checkpoint:
        args.model_type = 'lstm'
    else:
        raise ValueError(f"Could not auto-detect model type from checkpoint. Keys: {list(checkpoint.keys())}")

print(f"✓ Model type: {args.model_type}")

# Get feature list
if args.feature_list is None:
    # Try to get from checkpoint
    if 'feature_list' in checkpoint:
        feature_cols = checkpoint['feature_list']
        print(f"✓ Using feature list from checkpoint ({len(feature_cols)} features)")
    elif args.model_type == 'lstm':
        # For LSTM, use get_specified_features
        feature_cols = get_specified_features()
        print(f"✓ Using specified features from LSTM model ({len(feature_cols)} features)")
    else:
        raise ValueError("No feature list provided and cannot auto-detect. Use --feature_list")
elif args.feature_list == 'specified':
    feature_cols = get_specified_features()
    print(f"✓ Using specified features ({len(feature_cols)} features)")
elif args.feature_list.endswith('.csv'):
    # Load from CSV file
    feature_df = pd.read_csv(args.feature_list)
    if 'feature' in feature_df.columns:
        feature_cols = feature_df['feature'].tolist()
    elif 'Feature' in feature_df.columns:
        feature_cols = feature_df['Feature'].tolist()
    else:
        feature_cols = feature_df.iloc[:, 0].tolist()
    print(f"✓ Loaded {len(feature_cols)} features from {args.feature_list}")
else:
    # Comma-separated list
    feature_cols = [f.strip() for f in args.feature_list.split(',')]
    print(f"✓ Using provided feature list ({len(feature_cols)} features)")

# Filter features that exist in data
available_features = [f for f in feature_cols if f in df.columns]
missing_features = [f for f in feature_cols if f not in df.columns]
if missing_features:
    print(f"⚠ Warning: {len(missing_features)} features missing from data: {missing_features[:5]}...")
    feature_cols = available_features

print(f"✓ Using {len(feature_cols)} available features")

# Split data (use same split as training if available, otherwise recreate)
# Check if data_path is already a test split file
is_test_split = 'test.parquet' in args.data_path or '/test.parquet' in args.data_path

if is_test_split:
    # Data is already the test split - use all match_ids
    test_matches = df['match_id'].unique()
    test_df = df.copy()
    print(f"✓ Using test split directly: {len(test_matches)} games")
elif 'test_matches' in checkpoint:
    # Use test matches from checkpoint
    test_matches = checkpoint['test_matches']
    test_df = df[df['match_id'].isin(test_matches)].copy()
    print(f"✓ Using test matches from checkpoint ({len(test_matches)} games)")
else:
    # Recreate test split from full dataset
    print("Creating test split...")
    unique_matches = df['match_id'].unique()
    np.random.seed(42)
    shuffled_matches = np.random.permutation(unique_matches)
    
    n_train = int(len(unique_matches) * 0.7)
    n_val = int(len(unique_matches) * 0.15)
    
    test_matches = shuffled_matches[n_train + n_val:]
    test_df = df[df['match_id'].isin(test_matches)].copy()
    print(f"✓ Created test split: {len(test_matches)} games")

# Initialize model based on type
if args.model_type == 'lstm':
    # Load LSTM model
    input_size = checkpoint['input_size']
    hidden_size = checkpoint.get('hidden_size', 128)
    num_layers = checkpoint.get('num_layers', 2)
    dropout = checkpoint.get('dropout', 0.4)
    
    model = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load feature scaler if available
    feature_scaler = None
    if 'feature_scaler' in checkpoint:
        import pickle
        import io
        scaler_buffer = io.BytesIO(checkpoint['feature_scaler'])
        feature_scaler = pickle.load(scaler_buffer)
        print(f"✓ Feature scaler loaded from checkpoint")
    else:
        print(f"⚠ Warning: No feature scaler found in checkpoint. Features will not be scaled!")
        print(f"   This may cause incorrect predictions if model was trained with scaled features.")
    
    print(f"✓ LSTM model loaded: input_size={input_size}, hidden_size={hidden_size}, layers={num_layers}")
    
elif args.model_type == 'autoregressive':
    # Load autoregressive model
    import importlib.util
    spec = importlib.util.spec_from_file_location("NN_auto", "src/models/NN_autoregressive.py")
    NN_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(NN_module)
    
    AutoregressiveHierarchicalPredictor = NN_module.AutoregressiveHierarchicalPredictor
    FeatureGroups = NN_module.FeatureGroups
    
    scaler_dict = checkpoint['scaler_dict']
    forecast_horizon = checkpoint['forecast_horizon']
    
    fg = FeatureGroups()
    model = AutoregressiveHierarchicalPredictor(fg, hidden_dim=32, lstm_layers=1, forecast_horizon=forecast_horizon)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    target_mean = scaler_dict['target_mean']
    target_std = scaler_dict['target_std']
    print(f"✓ Autoregressive model loaded: forecast_horizon={forecast_horizon}")

print(f"\nProcessing all {len(test_matches)} test games...")
print(f"Minimum game length filter: {args.min_game_length} timestamps (games shorter than this will be skipped)")

# Filter out games that are too short before processing
print(f"\nFiltering games by minimum length ({args.min_game_length} timestamps)...")
games_by_length = test_df.groupby('match_id').size()
valid_matches = games_by_length[games_by_length >= args.min_game_length].index.tolist()
short_matches = games_by_length[games_by_length < args.min_game_length].index.tolist()

if len(short_matches) > 0:
    print(f"  ⚠ Skipping {len(short_matches)} games that are too short (< {args.min_game_length} timestamps)")
    print(f"  ✓ Keeping {len(valid_matches)} games with sufficient length")
else:
    print(f"  ✓ All {len(valid_matches)} games meet the minimum length requirement")

test_matches = valid_matches

# Evaluate ALL test games
all_rmses = []
all_maes = []
skipped = 0

for match_idx, match_id in enumerate(test_matches):
    if (match_idx + 1) % 50 == 0:
        print(f"  Processed {match_idx + 1}/{len(test_matches)} games...")
    
    try:
        game_data = test_df[test_df['match_id'] == match_id].copy().sort_values('frame_idx').reset_index(drop=True)
        
        total_frames = len(game_data)
        
        # Double-check length (should already be filtered, but safety check)
        if total_frames < args.min_game_length:
            skipped += 1
            continue
        
        # For LSTM, also need at least sequence_length frames
        if args.model_type == 'lstm' and total_frames < args.sequence_length:
            skipped += 1
            continue
        
        # Predict based on model type
        if args.model_type == 'lstm':
            # LSTM: Create sequences of length args.sequence_length
            # Predict the last frame of each sequence
            predictions_list = []
            actuals_list = []
            
            for start_idx in range(total_frames - args.sequence_length + 1):
                end_idx = start_idx + args.sequence_length
                sequence_data = game_data.iloc[start_idx:end_idx]
                
                # Extract features
                feature_values = sequence_data[feature_cols].values
                
                # Check for NaN
                if np.isnan(feature_values).any():
                    feature_values = np.nan_to_num(feature_values, nan=0.0)
                
                # Apply feature scaling if scaler is available (IMPORTANT: model was trained on scaled features!)
                if feature_scaler is not None:
                    # Reshape for scaling: (seq_len, features) -> (seq_len * features,)
                    original_shape = feature_values.shape
                    reshaped = feature_values.reshape(-1, feature_values.shape[-1])
                    scaled = feature_scaler.transform(reshaped)
                    feature_values = scaled.reshape(original_shape)
                
                # Convert to tensor: (1, seq_len, features)
                sequence_tensor = torch.FloatTensor(feature_values).unsqueeze(0)
                length_tensor = torch.LongTensor([args.sequence_length])
                
                # Predict
                with torch.no_grad():
                    pred = model(sequence_tensor, length_tensor).squeeze().item()
                
                # Get actual value (last frame of sequence)
                actual = sequence_data[args.target_col].iloc[-1]
                
                predictions_list.append(pred)
                actuals_list.append(actual)
            
            if len(predictions_list) == 0:
                skipped += 1
                continue
            
            predictions = np.array(predictions_list)
            actuals = np.array(actuals_list)
            
            # Calculate metrics for this game
            rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
            mae = np.mean(np.abs(actuals - predictions))
            
            all_rmses.append(rmse)
            all_maes.append(mae)
            
        elif args.model_type == 'autoregressive':
            # Autoregressive: Use all frames except last forecast_horizon as input
            cutoff_frame = total_frames - forecast_horizon
            if cutoff_frame < 1:
                skipped += 1
                continue
                
            input_data = game_data.iloc[:cutoff_frame].copy()
            target_data = game_data.iloc[cutoff_frame:].copy()
            
            # Extract features
            x1_seq = input_data[fg.damage_features].values
            x2_seq = input_data[fg.vision_features].values
            y_history = input_data[args.target_col].values
            
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
                    ('g1', fg.offensive_stats),
                    ('g2', fg.defensive_stats),
                    ('g3', fg.vamp_stats),
                    ('g4', fg.resource_stats),
                    ('g5', fg.mobility_stats)
                ]:
                    cols = [f'Player{p_idx}_{stat}' for stat in g_features]
                    g_data = input_data[cols].values
                    g_norm = normalize_feature(g_data, f'p{p_idx}_{g_key}')
                    player_features[f'p{p_idx}'][g_key] = torch.FloatTensor(g_norm)
            
            # Create batch
            batch_input = {
                'x1': torch.FloatTensor(x1_norm).unsqueeze(0),
                'x2': torch.FloatTensor(x2_norm).unsqueeze(0),
                'y_history': torch.FloatTensor(y_hist_norm).unsqueeze(0),
                'players': {},
                'seq_lens': torch.LongTensor([len(input_data)])
            }
            
            for p_key in player_features.keys():
                batch_input['players'][p_key] = {
                    g_key: player_features[p_key][g_key].unsqueeze(0)
                    for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']
                }
            
            # Predict
            with torch.no_grad():
                predictions = model(batch_input)
                predictions_denorm = predictions[0].numpy() * target_std + target_mean
            
            # Get real values
            actuals = target_data[args.target_col].values
            
            # Calculate metrics
            min_len = min(len(actuals), len(predictions_denorm))
            rmse = np.sqrt(np.mean((actuals[:min_len] - predictions_denorm[:min_len]) ** 2))
            mae = np.mean(np.abs(actuals[:min_len] - predictions_denorm[:min_len]))
            
            all_rmses.append(rmse)
            all_maes.append(mae)
        
    except Exception as e:
        print(f"  ⚠ Error processing game {match_id}: {e}")
        skipped += 1
        continue

print(f"\n✓ Processed {len(all_rmses)} games (skipped {skipped} too short)")

# Check if we have any results
if len(all_rmses) == 0:
    print("\n❌ Error: No games were successfully processed!")
    print("   All games were either too short or encountered errors.")
    print("   Please check your data and model configuration.")
    exit(1)

# Convert to numpy arrays
all_rmses = np.array(all_rmses)
all_maes = np.array(all_maes)

# Calculate statistics
print(f"\n{'='*60}")
print("RMSE DISTRIBUTION ANALYSIS")
print(f"{'='*60}")

# Count by threshold
count_under_1000 = np.sum(all_rmses < 1000)
count_under_2000 = np.sum(all_rmses < 2000)
count_under_3000 = np.sum(all_rmses < 3000)
count_over_3000 = np.sum(all_rmses > 3000)
count_over_5000 = np.sum(all_rmses > 5000)

total_games = len(all_rmses)

print(f"\nTotal test games analyzed: {total_games}")
print(f"\nRMSE Thresholds:")
if total_games > 0:
    print(f"  RMSE < 1,000 gold:  {count_under_1000:4d} games ({count_under_1000/total_games*100:5.1f}%)")
    print(f"  RMSE < 2,000 gold:  {count_under_2000:4d} games ({count_under_2000/total_games*100:5.1f}%)")
    print(f"  RMSE < 3,000 gold:  {count_under_3000:4d} games ({count_under_3000/total_games*100:5.1f}%)")
    print(f"  RMSE > 3,000 gold:  {count_over_3000:4d} games ({count_over_3000/total_games*100:5.1f}%)")
    print(f"  RMSE > 5,000 gold:  {count_over_5000:4d} games ({count_over_5000/total_games*100:5.1f}%)")

    print(f"\nDetailed Statistics:")
    print(f"  Mean RMSE:   {np.mean(all_rmses):8.2f} gold")
    print(f"  Median RMSE: {np.median(all_rmses):8.2f} gold")
    print(f"  Std Dev:     {np.std(all_rmses):8.2f} gold")
    print(f"  Min RMSE:    {np.min(all_rmses):8.2f} gold")
    print(f"  Max RMSE:    {np.max(all_rmses):8.2f} gold")
    print(f"  25th %ile:   {np.percentile(all_rmses, 25):8.2f} gold")
    print(f"  75th %ile:   {np.percentile(all_rmses, 75):8.2f} gold")

# Create histogram
print(f"\nCreating RMSE distribution plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1 = axes[0]
if len(all_rmses) > 0:
    max_rmse = all_rmses.max()
    bins = np.arange(0, min(15000, max_rmse + 500), 500)
    counts, edges, patches = ax1.hist(all_rmses, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
else:
    # Empty histogram if no data
    ax1.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax1.transAxes)
    patches = []

# Color code by threshold
for i, patch in enumerate(patches):
    if edges[i] < 1000:
        patch.set_facecolor('green')
        patch.set_alpha(0.7)
    elif edges[i] < 3000:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    else:
        patch.set_facecolor('coral')
        patch.set_alpha(0.7)

ax1.axvline(x=1000, color='green', linestyle='--', linewidth=2, label='RMSE = 1,000')
ax1.axvline(x=3000, color='red', linestyle='--', linewidth=2, label='RMSE = 3,000')
ax1.axvline(x=np.mean(all_rmses), color='purple', linestyle='-', linewidth=2.5, label=f'Mean = {np.mean(all_rmses):.0f}')
ax1.axvline(x=np.median(all_rmses), color='orange', linestyle='-', linewidth=2.5, label=f'Median = {np.median(all_rmses):.0f}')

ax1.set_xlabel('RMSE (Gold)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Games', fontsize=12, fontweight='bold')
model_name = os.path.basename(args.model_path).replace('.pth', '').replace('_', ' ').title()
ax1.set_title(f'RMSE Distribution - All {total_games} Test Games\n({model_name})', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Cumulative distribution
ax2 = axes[1]
sorted_rmses = np.sort(all_rmses)
cumulative = np.arange(1, len(sorted_rmses) + 1) / len(sorted_rmses) * 100

ax2.plot(sorted_rmses, cumulative, linewidth=2.5, color='steelblue')
ax2.axvline(x=1000, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax2.axvline(x=3000, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.axhline(y=50, color='orange', linestyle=':', linewidth=2, alpha=0.7)

ax2.set_xlabel('RMSE (Gold)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Cumulative RMSE Distribution', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add annotations
ax2.text(1000, 5, '1K', fontsize=10, color='green', fontweight='bold')
ax2.text(3000, 5, '3K', fontsize=10, color='red', fontweight='bold')

plt.suptitle(f'{model_name}: Test Set Performance Analysis', 
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()

# Determine output path
if args.output_path:
    output_path = args.output_path
else:
    model_basename = os.path.basename(args.model_path).replace('.pth', '')
    output_path = f'results/{model_basename}_test_rmse_distribution.png'

os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Distribution plot saved to: {output_path}")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}")
