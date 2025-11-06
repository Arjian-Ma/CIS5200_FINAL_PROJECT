#!/usr/bin/env python3
"""
Visualize games from each performance category:
- Excellent: RMSE < 1,000
- Good: 1,000 <= RMSE < 2,000
- Moderate: 2,000 <= RMSE < 3,000
- Challenging: RMSE > 3,000
Supports both LSTM and Autoregressive models
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
print("CATEGORIZING TEST GAMES BY PERFORMANCE")
print("="*60)

# Parse arguments
parser = argparse.ArgumentParser(description='Visualize games by performance category')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model checkpoint')
parser.add_argument('--model_type', type=str, default='auto',
                    choices=['auto', 'lstm', 'autoregressive'],
                    help='Model type: auto (detect from checkpoint), lstm, or autoregressive')
parser.add_argument('--data_path', type=str, default='data/processed/featured_data_with_scores.parquet',
                    help='Path to featured data (default: data/processed/featured_data_with_scores.parquet)')
parser.add_argument('--feature_list', type=str, default=None,
                    help='Feature list: "specified", CSV path, or comma-separated (default: auto-detect)')
parser.add_argument('--target_col', type=str, default='Total_Gold_Difference',
                    help='Target column name (default: Total_Gold_Difference)')
parser.add_argument('--sequence_length', type=int, default=15,
                    help='Sequence length for LSTM models (default: 15)')
parser.add_argument('--games_per_category', type=int, default=20,
                    help='Number of games to visualize per category (default: 20)')
parser.add_argument('--forecast_horizon', type=int, default=None,
                    help='Forecast horizon for autoregressive models (default: from checkpoint)')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory for plots (default: results/)')

args = parser.parse_args()

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
    if 'forecast_horizon' in checkpoint or 'scaler_dict' in checkpoint:
        args.model_type = 'autoregressive'
    elif 'input_size' in checkpoint:
        args.model_type = 'lstm'
    else:
        raise ValueError(f"Could not auto-detect model type from checkpoint. Keys: {list(checkpoint.keys())}")

print(f"✓ Model type: {args.model_type}")

# Get feature list for LSTM
if args.model_type == 'lstm':
    if args.feature_list is None:
        # Priority: checkpoint > get_specified_features()
        if 'feature_list' in checkpoint:
            feature_cols = checkpoint['feature_list']
            print(f"✓ Using feature list from checkpoint ({len(feature_cols)} features)")
        else:
            feature_cols = get_specified_features()
            print(f"✓ Using specified features from LSTM model ({len(feature_cols)} features)")
    elif args.feature_list == 'specified':
        feature_cols = get_specified_features()
        print(f"✓ Using specified features ({len(feature_cols)} features)")
    elif args.feature_list.endswith('.csv'):
        # Load from CSV using same logic as get_specified_features()
        feature_df = pd.read_csv(args.feature_list)
        if 'feature' in feature_df.columns:
            feature_col = 'feature'
        elif 'Feature' in feature_df.columns:
            feature_col = 'Feature'
        else:
            feature_col = feature_df.columns[0]
        feature_cols = feature_df[feature_col].dropna().tolist()
        print(f"✓ Loaded {len(feature_cols)} features from CSV: {args.feature_list}")
        print(f"  Features in order: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"  Features: {feature_cols}")
    else:
        feature_cols = [f.strip() for f in args.feature_list.split(',')]
        print(f"✓ Using provided feature list ({len(feature_cols)} features)")
    
    # Filter features that exist in data, maintaining order
    available_features = [f for f in feature_cols if f in df.columns]
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"⚠ Warning: {len(missing_features)} features missing from data: {missing_features[:5]}...")
    feature_cols = available_features
    print(f"✓ Using {len(feature_cols)} available features (order maintained)")

# Split data
if 'test_matches' in checkpoint:
    test_matches = checkpoint['test_matches']
    print(f"✓ Using test matches from checkpoint ({len(test_matches)} games)")
else:
    print("Creating test split...")
    unique_matches = df['match_id'].unique()
np.random.seed(42)
shuffled_matches = np.random.permutation(unique_matches)

n_train = int(len(unique_matches) * 0.7)
n_val = int(len(unique_matches) * 0.15)

test_matches = shuffled_matches[n_train + n_val:]
    print(f"✓ Created test split: {len(test_matches)} games")

test_df = df[df['match_id'].isin(test_matches)].copy()

# Initialize feature scaler (will be set for LSTM models)
feature_scaler = None

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
    forecast_horizon = args.forecast_horizon if args.forecast_horizon else checkpoint.get('forecast_horizon', 5)

fg = FeatureGroups()
model = AutoregressiveHierarchicalPredictor(fg, hidden_dim=32, lstm_layers=1, forecast_horizon=forecast_horizon)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

target_mean = scaler_dict['target_mean']
target_std = scaler_dict['target_std']
    print(f"✓ Autoregressive model loaded: forecast_horizon={forecast_horizon}")

# Helper function to predict a game
def predict_game(match_id, game_data):
    total_frames = len(game_data)
    
    if args.model_type == 'lstm':
        if total_frames < args.sequence_length:
            return None
        
        # LSTM: Create sliding window predictions
        predictions_list = []
        actuals_list = []
        frames_list = []
        
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
            
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(feature_values).unsqueeze(0)
            length_tensor = torch.LongTensor([args.sequence_length])
            
            # Predict
            with torch.no_grad():
                pred = model(sequence_tensor, length_tensor).squeeze().item()
            
            # Get actual value (last frame of sequence)
            actual = sequence_data[args.target_col].iloc[-1]
            frame_idx = sequence_data['frame_idx'].iloc[-1]
            
            predictions_list.append(pred)
            actuals_list.append(actual)
            frames_list.append(frame_idx)
        
        if len(predictions_list) == 0:
            return None
        
        predictions = np.array(predictions_list)
        actuals = np.array(actuals_list)
        frames = np.array(frames_list)
        
        # Calculate RMSE for this game
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        
        return {
            'match_id': match_id,
            'real_full': game_data[args.target_col].values,
            'frames_full': game_data['frame_idx'].values,
            'cutoff_frame': None,
            'predicted': predictions,
            'actuals': actuals,
            'frames': frames,
            'rmse': rmse,
            'model_type': 'lstm'
        }
    
    elif args.model_type == 'autoregressive':
        if total_frames < forecast_horizon + 10:
        return None
    
    cutoff_frame = total_frames - forecast_horizon
    input_data = game_data.iloc[:cutoff_frame].copy()
    target_data = game_data.iloc[cutoff_frame:].copy()
    
    # Extract and normalize features
    x1_seq = input_data[fg.damage_features].values
    x2_seq = input_data[fg.vision_features].values
        y_history = input_data[args.target_col].values
    
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
    
        real_last = target_data[args.target_col].values
    min_len = min(len(real_last), len(predictions_denorm))
    rmse = np.sqrt(np.mean((real_last[:min_len] - predictions_denorm[:min_len]) ** 2))
    
    return {
        'match_id': match_id,
            'real_full': game_data[args.target_col].values,
        'frames_full': game_data['frame_idx'].values,
        'cutoff_frame': cutoff_frame,
        'predicted_last': predictions_denorm,
        'real_last': real_last,
        'frames_last': target_data['frame_idx'].values,
            'rmse': rmse,
            'model_type': 'autoregressive'
    }

# Categorize all test games
print(f"\nCategorizing all test games by RMSE...")
categories = {
    'excellent': [],  # < 1000
    'good': [],       # 1000-2000
    'moderate': [],   # 2000-3000
    'challenging': [] # > 3000
}

for match_idx, match_id in enumerate(test_matches):
    if (match_idx + 1) % 100 == 0:
        print(f"  Categorized {match_idx + 1}/{len(test_matches)} games...")
    
    game_data = test_df[test_df['match_id'] == match_id].copy().sort_values('frame_idx').reset_index(drop=True)
    result = predict_game(match_id, game_data)
    
    if result is None:
        continue
    
    rmse = result['rmse']
    
    if rmse < 1000:
        categories['excellent'].append(result)
    elif rmse < 2000:
        categories['good'].append(result)
    elif rmse < 3000:
        categories['moderate'].append(result)
    else:
        categories['challenging'].append(result)

print(f"\n{'='*60}")
print("CATEGORIZATION COMPLETE")
print(f"{'='*60}")
print(f"Excellent (RMSE < 1,000):      {len(categories['excellent'])} games")
print(f"Good (1,000 ≤ RMSE < 2,000):   {len(categories['good'])} games")
print(f"Moderate (2,000 ≤ RMSE < 3,000): {len(categories['moderate'])} games")
print(f"Challenging (RMSE ≥ 3,000):    {len(categories['challenging'])} games")

# Select games from each category
np.random.seed(400)

selected_by_category = {}
for cat_name, games in categories.items():
    if len(games) >= args.games_per_category:
        selected_by_category[cat_name] = np.random.choice(len(games), size=args.games_per_category, replace=False)
    else:
        selected_by_category[cat_name] = list(range(len(games)))
    print(f"Selected {len(selected_by_category[cat_name])} games from '{cat_name}' category")

# Create visualization for each category
category_titles = {
    'excellent': 'Excellent Predictions (RMSE < 1,000)',
    'good': 'Good Predictions (1,000 ≤ RMSE < 2,000)',
    'moderate': 'Moderate Predictions (2,000 ≤ RMSE < 3,000)',
    'challenging': 'Challenging Predictions (RMSE ≥ 3,000)'
}

category_colors = {
    'excellent': 'green',
    'good': 'steelblue',
    'moderate': 'orange',
    'challenging': 'red'
}

# Determine output directory
if args.output_dir:
    output_dir = args.output_dir
else:
    model_basename = os.path.basename(args.model_path).replace('.pth', '')
    output_dir = f'results/{model_basename}_by_performance'

os.makedirs(output_dir, exist_ok=True)

for cat_name, indices in selected_by_category.items():
    if len(indices) == 0:
        print(f"\nSkipping '{cat_name}' category - no games available")
        continue
    
    print(f"\nCreating visualization for '{cat_name}' category...")
    
    num_games = len(indices)
    cols = 4
    rows = (num_games + cols - 1) // cols
    
    fig = plt.figure(figsize=(24, 4*rows))
    
    for plot_idx, game_idx in enumerate(indices):
        game = categories[cat_name][game_idx]
        ax = plt.subplot(rows, cols, plot_idx + 1)
        
        if game['model_type'] == 'lstm':
            # LSTM visualization
            frames_full = game['frames_full']
            real_full = game['real_full']
            frames_pred = game['frames']
            predictions = game['predicted']
            actuals = game['actuals']
            
            # Plot full real trajectory
            ax.plot(frames_full, real_full, 'b-', 
                   label='Actual', linewidth=2, marker='o', markersize=2.5, alpha=0.6)
            
            # Plot predictions
            ax.plot(frames_pred, predictions, 'r--', 
                   label='Predicted', linewidth=2, marker='D', markersize=4, alpha=0.8)
            
            # Plot actuals at prediction points
            ax.plot(frames_pred, actuals, color=category_colors[cat_name], marker='*',
                   label='Actual (at pred)', markersize=6, alpha=0.9, zorder=5, linestyle='None')
        
        elif game['model_type'] == 'autoregressive':
            # Autoregressive visualization
        real_full = game['real_full']
        frames_full = game['frames_full']
        cutoff = game['cutoff_frame']
        pred_last = game['predicted_last']
        real_last = game['real_last']
        frames_last = game['frames_last']
        
        # Plot known history
        ax.plot(frames_full[:cutoff], real_full[:cutoff], 'b-', 
               label='Known', linewidth=2, marker='o', markersize=2.5, alpha=0.6)
        
        # Plot real endgame
        ax.plot(frames_last, real_last, color=category_colors[cat_name], linestyle='-',
               label='Real', linewidth=3, marker='*', markersize=7, alpha=0.9, zorder=5)
        
        # Plot predicted endgame
        ax.plot(frames_last[:len(pred_last)], pred_last, 'r--', 
               label='Pred', linewidth=2.5, marker='D', markersize=5, alpha=0.8, zorder=4)
        
        # Vertical line at cutoff
        ax.axvline(x=frames_full[cutoff-1], color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
        
        # Shade forecast region
        if len(frames_last) > 0:
            ax.axvspan(frames_last[0], frames_last[-1], alpha=0.12, color='yellow')
        
        ax.axhline(y=0, color='k', linestyle=':', linewidth=0.8, alpha=0.5)
        
        ax.set_xlabel('Frame', fontsize=7)
        ax.set_ylabel('Gold', fontsize=7)
        
        ax.set_title(f'G{plot_idx+1}: RMSE={game["rmse"]:.0f}', 
                     fontsize=8, fontweight='bold')
        ax.legend(fontsize=5, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
    
    model_name = os.path.basename(args.model_path).replace('.pth', '').replace('_', ' ').title()
    plt.suptitle(f'{category_titles[cat_name]} - {model_name}', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{cat_name}_games.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    plt.close()

print(f"\n{'='*60}")
print("ALL VISUALIZATIONS COMPLETE")
print(f"{'='*60}")
print(f"\nGenerated files in {output_dir}/:")
for cat_name in ['excellent', 'good', 'moderate', 'challenging']:
    if len(selected_by_category.get(cat_name, [])) > 0:
        print(f"  - {cat_name}_games.png  - {len(selected_by_category[cat_name])} games with {category_titles[cat_name]}")
