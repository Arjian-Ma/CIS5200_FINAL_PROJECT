#!/usr/bin/env python3
"""
Visualize model predictions on test games
Supports both endgame forecasting (autoregressive) and sequence-based (LSTM) models
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
print("MODEL VISUALIZATION - TEST GAMES")
print("="*60)

# Parse arguments
parser = argparse.ArgumentParser(description='Visualize model predictions on test games')
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
parser.add_argument('--num_games', type=int, default=30,
                    help='Number of games to visualize (default: 30)')
parser.add_argument('--output_path', type=str, default=None,
                    help='Output path for plot (default: auto-generated)')
parser.add_argument('--forecast_horizon', type=int, default=None,
                    help='Forecast horizon for autoregressive models (default: from checkpoint)')

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
    # Check for LSTM model first (has input_size)
    if 'input_size' in checkpoint:
        # Check if it's autoregressive LSTM
        if checkpoint.get('forecast_horizon', 1) > 1 or checkpoint.get('output_size', 1) > 1:
            args.model_type = 'lstm'  # Still use 'lstm' type, but will detect autoregressive in prediction
        else:
            args.model_type = 'lstm'
    # Check for hierarchical autoregressive model (has scaler_dict)
    elif 'scaler_dict' in checkpoint:
        args.model_type = 'autoregressive'
    else:
        raise ValueError(f"Could not auto-detect model type from checkpoint. Keys: {list(checkpoint.keys())}")

print(f"✓ Model type: {args.model_type}")

# Get feature list for LSTM
if args.model_type == 'lstm':
    # Priority: Always use checkpoint feature_list if available (to match scaler)
    if 'feature_list' in checkpoint:
        feature_cols = checkpoint['feature_list']
        print(f"✓ Using feature list from checkpoint ({len(feature_cols)} features)")
        if args.feature_list is not None and args.feature_list != 'specified':
            print(f"  ⚠ Note: Overriding with checkpoint features to match scaler (expected {len(feature_cols)} features)")
    elif args.feature_list is None:
        # Fallback: use get_specified_features() if no checkpoint feature list
        feature_cols = get_specified_features()
        print(f"✓ Using specified features from LSTM model ({len(feature_cols)} features)")
    elif args.feature_list == 'specified':
        feature_cols = get_specified_features()
        print(f"✓ Using specified features ({len(feature_cols)} features)")
    elif args.feature_list.endswith('.csv'):
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
    
    # Filter features that exist in data
    available_features = [f for f in feature_cols if f in df.columns]
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"⚠ Warning: {len(missing_features)} features missing from data: {missing_features[:5]}...")
    feature_cols = available_features
    print(f"✓ Using {len(feature_cols)} available features")

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
    
    # Check if model is autoregressive
    lstm_forecast_horizon = checkpoint.get('forecast_horizon', 1)
    lstm_output_size = checkpoint.get('output_size', 1)
    is_lstm_autoregressive = lstm_forecast_horizon > 1 or lstm_output_size > 1
    
    if is_lstm_autoregressive:
        output_size = lstm_output_size if lstm_output_size > 1 else lstm_forecast_horizon
        print(f"🔄 Detected LSTM autoregressive mode: forecast_horizon={lstm_forecast_horizon}, output_size={output_size}")
    else:
        output_size = 1
        print(f"✓ LSTM many-to-one mode: output_size={output_size}")
    
    model = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=output_size
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
    
    print(f"✓ LSTM model loaded: input_size={input_size}, hidden_size={hidden_size}, layers={num_layers}, output_size={output_size}")
    
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

# Select test games
print(f"\nSelecting {args.num_games} test games...")
test_match_frame_counts = test_df.groupby('match_id').size()

# Determine minimum frames needed based on model type
if args.model_type == 'autoregressive':
    min_frames = forecast_horizon + args.sequence_length
elif args.model_type == 'lstm':
    # Check if LSTM is autoregressive
    lstm_forecast_horizon = checkpoint.get('forecast_horizon', 1)
    if lstm_forecast_horizon > 1:
        min_frames = args.sequence_length + lstm_forecast_horizon
    else:
        min_frames = args.sequence_length + 5
else:
    min_frames = args.sequence_length + 5

good_test_matches = test_match_frame_counts[test_match_frame_counts >= min_frames].index

np.random.seed(300)
selected_matches = np.random.choice(good_test_matches, size=min(args.num_games, len(good_test_matches)), replace=False)

print(f"✓ Selected {len(selected_matches)} test games")

# Evaluate each game
all_game_results = []

for match_idx, match_id in enumerate(selected_matches):
    print(f"Processing game {match_idx + 1}/{len(selected_matches)}: {match_id}...")
    
    game_data = test_df[test_df['match_id'] == match_id].copy().sort_values('frame_idx').reset_index(drop=True)
    total_frames = len(game_data)
    
    if args.model_type == 'lstm':
        # Check if LSTM is autoregressive
        is_lstm_autoregressive = checkpoint.get('forecast_horizon', 1) > 1 or checkpoint.get('output_size', 1) > 1
        lstm_forecast_horizon = checkpoint.get('forecast_horizon', 1)
        
        if is_lstm_autoregressive:
            # LSTM Autoregressive mode: Use cutoff approach
            if total_frames < args.sequence_length + lstm_forecast_horizon:
                print(f"  Skipping - too short (need at least {args.sequence_length + lstm_forecast_horizon} frames)")
                continue
            
            forecast_horizon_actual = lstm_forecast_horizon
            cutoff_frame = total_frames - forecast_horizon_actual
            
            if cutoff_frame < args.sequence_length:
                print(f"  Skipping - invalid forecast horizon")
                continue
            
            input_data = game_data.iloc[:cutoff_frame].copy()
            target_data = game_data.iloc[cutoff_frame:].copy()
            
            # Use last sequence_length frames from known history
            start_idx = max(0, cutoff_frame - args.sequence_length)
            sequence_data = game_data.iloc[start_idx:cutoff_frame].copy()
            
            # Extract X features and add Y history for autoregressive mode
            X_features = sequence_data[feature_cols].values
            y_history = sequence_data[args.target_col].values.reshape(-1, 1)
            feature_values = np.concatenate([X_features, y_history], axis=1)
            
            # Check for NaN
            if np.isnan(feature_values).any():
                feature_values = np.nan_to_num(feature_values, nan=0.0)
            
            # Apply feature scaling
            if feature_scaler is not None:
                original_shape = feature_values.shape
                reshaped = feature_values.reshape(-1, feature_values.shape[-1])
                scaled = feature_scaler.transform(reshaped)
                feature_values = scaled.reshape(original_shape)
            
            # Ensure exactly sequence_length frames
            if len(feature_values) < args.sequence_length:
                padding_needed = args.sequence_length - len(feature_values)
                first_frame = feature_values[0:1].repeat(padding_needed, axis=0)
                feature_values = np.vstack([first_frame, feature_values])
            elif len(feature_values) > args.sequence_length:
                feature_values = feature_values[-args.sequence_length:]
            
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(feature_values).unsqueeze(0)
            length_tensor = torch.LongTensor([args.sequence_length])
            
            # Predict (output shape: (1, forecast_horizon))
            with torch.no_grad():
                predictions_raw = model(sequence_tensor, length_tensor)
                predictions_array = predictions_raw.squeeze().cpu().numpy()
            
            # Handle forecast_horizon vs actual available future frames
            if len(predictions_array) > forecast_horizon_actual:
                predictions_array = predictions_array[:forecast_horizon_actual]
            elif len(predictions_array) < forecast_horizon_actual:
                last_pred = predictions_array[-1] if len(predictions_array) > 0 else 0
                predictions_array = np.pad(predictions_array, (0, forecast_horizon_actual - len(predictions_array)), 
                                           constant_values=last_pred)
            
            # Get actual values for future frames
            actuals = target_data[args.target_col].values[:forecast_horizon_actual]
            frames = target_data['frame_idx'].values[:forecast_horizon_actual]
            
            # Calculate metrics
            min_len = min(len(actuals), len(predictions_array))
            rmse = np.sqrt(np.mean((actuals[:min_len] - predictions_array[:min_len]) ** 2))
            mae = np.mean(np.abs(actuals[:min_len] - predictions_array[:min_len]))
            
            # Store results
            all_game_results.append({
                'match_id': match_id,
                'real_full': game_data[args.target_col].values,
                'frames_full': game_data['frame_idx'].values,
                'cutoff_frame': cutoff_frame,
                'predicted_last': predictions_array,
                'real_last': actuals,
                'frames_last': frames,
                'rmse': rmse,
                'mae': mae,
                'model_type': 'lstm_autoregressive'
            })
        else:
            # LSTM Many-to-One mode: Create sliding window predictions
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
        print(f"  Skipping - too short")
        continue
    
            predictions = np.array(predictions_list)
            actuals = np.array(actuals_list)
            frames = np.array(frames_list)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
            mae = np.mean(np.abs(actuals - predictions))
            
            # Store results
            all_game_results.append({
                'match_id': match_id,
                'real_full': game_data[args.target_col].values,
                'frames_full': game_data['frame_idx'].values,
                'cutoff_frame': None,
                'predicted': predictions,
                'actuals': actuals,
                'frames': frames,
                'rmse': rmse,
                'mae': mae,
                'model_type': 'lstm'
            })
        
    elif args.model_type == 'autoregressive':
        # Autoregressive: Predict last forecast_horizon frames
        if total_frames < forecast_horizon + 10:
            print(f"  Skipping - too short (need at least {forecast_horizon + 10} frames)")
            continue
        
    cutoff_frame = total_frames - forecast_horizon
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
        real_full = game_data[args.target_col].values
    frames_full = game_data['frame_idx'].values
        real_last = target_data[args.target_col].values
    frames_last = target_data['frame_idx'].values
    
    # Calculate metrics
    min_len = min(len(real_last), len(predictions_denorm))
    rmse = np.sqrt(np.mean((real_last[:min_len] - predictions_denorm[:min_len]) ** 2))
    mae = np.mean(np.abs(real_last[:min_len] - predictions_denorm[:min_len]))
    
    all_game_results.append({
        'match_id': match_id,
        'real_full': real_full,
        'frames_full': frames_full,
        'cutoff_frame': cutoff_frame,
        'predicted_last': predictions_denorm,
        'real_last': real_last,
        'frames_last': frames_last,
        'rmse': rmse,
            'mae': mae,
            'model_type': 'autoregressive'
    })
    
    print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Create visualization
num_games = len(all_game_results)
if num_games == 0:
    print("❌ No games to visualize!")
    sys.exit(1)

# Calculate grid dimensions
cols = 5
rows = (num_games + cols - 1) // cols

fig = plt.figure(figsize=(24, 4*rows))

for idx, game in enumerate(all_game_results):
    ax = plt.subplot(rows, cols, idx + 1)
    
    if game['model_type'] == 'lstm':
        # LSTM many-to-one visualization: show sliding window predictions
        frames_full = game['frames_full']
        real_full = game['real_full']
        frames_pred = game['frames']
        predictions = game['predicted']
        actuals = game['actuals']
        
        # Plot full real trajectory
        ax.plot(frames_full, real_full, 'b-', 
               label='Actual', linewidth=2, marker='o', markersize=3, alpha=0.6)
        
        # Plot predictions
        ax.plot(frames_pred, predictions, 'r--', 
               label='Predicted', linewidth=2, marker='D', markersize=4, alpha=0.8)
        
        # Plot actuals at prediction points
        ax.plot(frames_pred, actuals, 'g*', 
               label='Actual (at pred)', markersize=6, alpha=0.9, zorder=5)
        
        ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Frame', fontsize=8)
        ax.set_ylabel('Gold Diff', fontsize=8)
        match_display = game['match_id'][:12] + '...' if len(game['match_id']) > 12 else game['match_id']
        ax.set_title(f'G{idx+1}: RMSE={game["rmse"]:.0f}', fontsize=9, fontweight='bold')
        ax.legend(fontsize=5, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    
    elif game['model_type'] == 'lstm_autoregressive':
        # LSTM autoregressive visualization: show endgame forecasting
    real_full = game['real_full']
    frames_full = game['frames_full']
    cutoff = game['cutoff_frame']
    pred_last = game['predicted_last']
    real_last = game['real_last']
    frames_last = game['frames_last']
    
        # Plot known history
    ax.plot(frames_full[:cutoff], real_full[:cutoff], 'b-', 
           label='Known History', linewidth=2, marker='o', markersize=3, alpha=0.6)
    
    # Plot real endgame (ground truth)
    ax.plot(frames_last, real_last, 'g-', 
           label='Real Endgame', linewidth=3, marker='*', markersize=8, alpha=0.9, zorder=5)
    
    # Plot predicted endgame
    ax.plot(frames_last[:len(pred_last)], pred_last, 'r--', 
           label='Predicted', linewidth=3, marker='D', markersize=6, alpha=0.8, zorder=4)
    
    # Vertical line at cutoff
        ax.axvline(x=frames_full[cutoff-1], color='orange', linestyle=':', linewidth=2, alpha=0.7)
    
    # Shade the forecast region
    if len(frames_last) > 0:
        ax.axvspan(frames_last[0], frames_last[-1], alpha=0.15, color='yellow')
    
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Frame', fontsize=8)
        ax.set_ylabel('Gold Diff', fontsize=8)
        match_display = game['match_id'][:12] + '...' if len(game['match_id']) > 12 else game['match_id']
        ax.set_title(f'G{idx+1}: RMSE={game["rmse"]:.0f}', fontsize=9, fontweight='bold')
        ax.legend(fontsize=5, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
    elif game['model_type'] == 'autoregressive':
        # Autoregressive visualization: show endgame forecasting
        real_full = game['real_full']
        frames_full = game['frames_full']
        cutoff = game['cutoff_frame']
        pred_last = game['predicted_last']
        real_last = game['real_last']
        frames_last = game['frames_last']
        
        # Plot full real trajectory
        ax.plot(frames_full[:cutoff], real_full[:cutoff], 'b-', 
               label='Known History', linewidth=2, marker='o', markersize=3, alpha=0.6)
        
        # Plot real endgame (ground truth)
        ax.plot(frames_last, real_last, 'g-', 
               label='Real Endgame', linewidth=3, marker='*', markersize=8, alpha=0.9, zorder=5)
        
        # Plot predicted endgame
        ax.plot(frames_last[:len(pred_last)], pred_last, 'r--', 
               label='Predicted', linewidth=3, marker='D', markersize=6, alpha=0.8, zorder=4)
        
        # Vertical line at cutoff
        ax.axvline(x=frames_full[cutoff-1], color='orange', linestyle=':', linewidth=2, alpha=0.7)
        
        # Shade the forecast region
        if len(frames_last) > 0:
            ax.axvspan(frames_last[0], frames_last[-1], alpha=0.15, color='yellow')
        
        ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Frame', fontsize=8)
        ax.set_ylabel('Gold Diff', fontsize=8)
    
    match_display = game['match_id'][:12] + '...' if len(game['match_id']) > 12 else game['match_id']
        ax.set_title(f'G{idx+1}: RMSE={game["rmse"]:.0f}', fontsize=9, fontweight='bold')
    ax.legend(fontsize=5, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

# Determine title
model_name = os.path.basename(args.model_path).replace('.pth', '').replace('_', ' ').title()
if args.model_type == 'lstm':
    # Check if any games are autoregressive
    has_autoregressive = any(g.get('model_type') == 'lstm_autoregressive' for g in all_game_results)
    if has_autoregressive:
        title = f'LSTM Autoregressive Model: {num_games} Test Games (Endgame Forecasting)'
    else:
        title = f'LSTM Model: {num_games} Test Games (Sliding Window Predictions)'
elif args.model_type == 'autoregressive':
    title = f'Endgame Forecasting Model: {num_games} Test Games'

plt.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()

# Determine output path
if args.output_path:
    output_path = args.output_path
else:
    model_basename = os.path.basename(args.model_path).replace('.pth', '')
    output_path = f'results/{model_basename}_visualization_{num_games}games.png'

os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")
plt.close()

# Print summary statistics
print(f"\n{'='*60}")
print("TEST GAME SUMMARY")
print(f"{'='*60}")

if args.model_type == 'lstm':
    # Check if any games are autoregressive
    has_autoregressive = any(g.get('model_type') == 'lstm_autoregressive' for g in all_game_results)
    if has_autoregressive:
        print(f"{'Game':<6} {'Match ID':<25} {'Total':<8} {'Cutoff':<8} {'RMSE':>10} {'MAE':>10}")
        print("-"*65)
        for idx, game in enumerate(all_game_results):
            match_display = game['match_id'][:25]
            total = len(game['real_full'])
            cutoff = game.get('cutoff_frame', 0)
            print(f"{idx+1:<6} {match_display:<25} {total:<8} {cutoff:<8} {game['rmse']:>10.2f} {game['mae']:>10.2f}")
    else:
        print(f"{'Game':<6} {'Match ID':<25} {'Frames':<8} {'RMSE':>10} {'MAE':>10}")
        print("-"*60)
        for idx, game in enumerate(all_game_results):
            match_display = game['match_id'][:25]
            total = len(game['real_full'])
            print(f"{idx+1:<6} {match_display:<25} {total:<8} {game['rmse']:>10.2f} {game['mae']:>10.2f}")
elif args.model_type == 'autoregressive':
print(f"{'Game':<6} {'Match ID':<25} {'Total':<8} {'Cutoff':<8} {'RMSE':>10} {'MAE':>10}")
print("-"*60)
for idx, game in enumerate(all_game_results):
    match_display = game['match_id'][:25]
    total = len(game['real_full'])
    cutoff = game['cutoff_frame']
    print(f"{idx+1:<6} {match_display:<25} {total:<8} {cutoff:<8} {game['rmse']:>10.2f} {game['mae']:>10.2f}")

# Overall statistics
if len(all_game_results) > 0:
    all_rmses = [g['rmse'] for g in all_game_results]
    all_maes = [g['mae'] for g in all_game_results]
    
    print(f"\n{'='*60}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Average RMSE: {np.mean(all_rmses):.2f} gold")
    print(f"Average MAE: {np.mean(all_maes):.2f} gold")
    print(f"Best RMSE: {np.min(all_rmses):.2f} gold")
    print(f"Worst RMSE: {np.max(all_rmses):.2f} gold")
    print(f"Std Dev RMSE: {np.std(all_rmses):.2f} gold")

print(f"\n{'='*60}")
print("VISUALIZATION COMPLETE")
print(f"{'='*60}")
