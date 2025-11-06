#!/usr/bin/env python3
"""
Autoregressive Hierarchical Neural Network
Incorporates past gold difference as an additional exogenous variable
y_hat = Ax_1 + Bx_2 + Cx_3 + Dy_history

Now gold difference has its own LSTM to capture momentum/trends
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Import base classes from NN.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import importlib.util
spec = importlib.util.spec_from_file_location("NN", "src/models/Hierarchical Neural Network/NN.py")
NN_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NN_module)

FeatureGroups = NN_module.FeatureGroups

# Load data
print("\n" + "="*60)
print("LOADING DATA FOR AUTOREGRESSIVE MODEL")
print("="*60)
print("Loading data from parquet file...")
full_data = pd.read_parquet("Data/processed/featured_data.parquet")

print(f"✓ Full data loaded: {full_data.shape}")

# Remove data leakage features (but KEEP Total_Gold_Difference_Last_Time_Frame for autoregressive use!)
leakage_features = [
    # 'Total_Gold_Difference_Last_Time_Frame',  # KEEP THIS - we'll use it as lagged y
    'Total_Minions_Killed_Difference',
    'Total_Jungle_Minions_Killed_Difference',
    'Total_Kill_Difference',
    'Total_Assist_Difference',
    'Elite_Monster_Killed_Difference',
    'Buildings_Taken_Difference',
    'Total_Xp_Difference_Last_Time_Frame',
]

print(f"\nRemoving {len(leakage_features)} data leakage features (keeping gold history)...")
df_clean = full_data.drop(columns=[f for f in leakage_features if f in full_data.columns], errors='ignore')
print(f"Cleaned data shape: {df_clean.shape}")

# Split data
unique_matches = df_clean['match_id'].unique()
n_matches = len(unique_matches)
print(f"\nTotal unique matches: {n_matches}")

np.random.seed(42)
shuffled_matches = np.random.permutation(unique_matches)

n_train = int(n_matches * 0.7)
n_val = int(n_matches * 0.15)

train_matches = shuffled_matches[:n_train]
val_matches = shuffled_matches[n_train:n_train + n_val]
test_matches = shuffled_matches[n_train + n_val:]

train_data = df_clean[df_clean['match_id'].isin(train_matches)].copy()
val_data = df_clean[df_clean['match_id'].isin(val_matches)].copy()
test_data = df_clean[df_clean['match_id'].isin(test_matches)].copy()

print(f"\nData split:")
print(f"Train: {train_data.shape} ({len(train_matches)} matches)")
print(f"Val: {val_data.shape} ({len(val_matches)} matches)")
print(f"Test: {test_data.shape} ({len(test_matches)} matches)")


class AutoregressiveGameDataset(Dataset):
    """
    Dataset for endgame forecasting
    - Uses frames [0, N-forecast_horizon] to predict frames [N-forecast_horizon+1, ..., N]
    - ONE sequence per game (predicts only the last 'forecast_horizon' frames)
    - Includes past gold difference as an input feature (y_history)
    """
    
    def __init__(self, dataframe, feature_groups, forecast_horizon=5, scaler_dict=None, fit_scaler=False, min_frames=10):
        self.data = dataframe
        self.fg = feature_groups
        self.forecast_horizon = forecast_horizon
        self.min_frames = min_frames
        self.scaler_dict = scaler_dict if scaler_dict is not None else {}
        
        # Store sequences
        self.sequences = {
            'x1': [], 'x2': [], 'y_history': [],  # Added y_history!
            'players': {f'p{i}': {'g1': [], 'g2': [], 'g3': [], 'g4': [], 'g5': []} for i in range(1, 11)}
        }
        self.targets = []  # Will be multi-step targets (last forecast_horizon frames)
        self.match_ids = []
        self.sequence_lengths = []
        self.target_horizons = []  # How many steps ahead we can predict
        
        print(f"Creating ENDGAME forecasting sequences (predict last {forecast_horizon} frames)...")
        
        # Group by match
        for match_id, match_data in dataframe.groupby('match_id'):
            match_data = match_data.sort_values('frame_idx').reset_index(drop=True)
            
            # Need at least min_frames + forecast_horizon frames
            if len(match_data) < min_frames + forecast_horizon:
                continue
            
            # ONE SEQUENCE PER GAME:
            # Use all frames EXCEPT last forecast_horizon frames as input
            cutoff_idx = len(match_data) - forecast_horizon
            
            # Input: frames [0, cutoff_idx-1]
            input_data = match_data.iloc[:cutoff_idx]
            
            # Targets: gold difference at last forecast_horizon frames
            target_frames = match_data.iloc[cutoff_idx:]
            targets_y = target_frames['Total_Gold_Difference'].values
            
            # Extract x1: damage history [0, cutoff_idx-1]
            x1_seq = input_data[self.fg.damage_features].values
            self.sequences['x1'].append(x1_seq)
            
            # Extract x2: vision history [0, cutoff_idx-1]
            x2_seq = input_data[self.fg.vision_features].values
            self.sequences['x2'].append(x2_seq)
            
            # Extract y_history: past gold differences [0, cutoff_idx-1]
            y_history = input_data['Total_Gold_Difference'].values
            self.sequences['y_history'].append(y_history)
            
            # Extract player features history [0, cutoff_idx-1]
            for player_idx in range(1, 11):
                for g_key, g_features in [
                    ('g1', self.fg.offensive_stats),
                    ('g2', self.fg.defensive_stats),
                    ('g3', self.fg.vamp_stats),
                    ('g4', self.fg.resource_stats),
                    ('g5', self.fg.mobility_stats)
                ]:
                    cols = [f'Player{player_idx}_{stat}' for stat in g_features]
                    g_seq = input_data[cols].values
                    self.sequences['players'][f'p{player_idx}'][g_key].append(g_seq)
            
            # Store multi-step targets (last forecast_horizon frames)
            self.targets.append(targets_y)
            self.match_ids.append(match_id)
            self.sequence_lengths.append(len(input_data))
            self.target_horizons.append(len(targets_y))
        
        print(f"  Created {len(self.targets)} endgame sequences (1 per game)")
        print(f"  Input lengths range: {min(self.sequence_lengths)} to {max(self.sequence_lengths)} frames")
        print(f"  Each sequence predicts last {forecast_horizon} frames of the game")
        
        # Convert to tensors and normalize
        self._convert_and_normalize(fit_scaler)
    
    def _convert_and_normalize(self, fit_scaler):
        """Convert to tensors and apply normalization"""
        
        # Normalize helper
        def normalize(data_list, key, fit=False):
            all_data = np.concatenate(data_list, axis=0)
            
            if fit:
                mean = all_data.mean(axis=0)
                std = all_data.std(axis=0) + 1e-8
                self.scaler_dict[f'{key}_mean'] = mean
                self.scaler_dict[f'{key}_std'] = std
            else:
                mean = self.scaler_dict.get(f'{key}_mean', 0)
                std = self.scaler_dict.get(f'{key}_std', 1)
            
            normalized = [(seq - mean) / std for seq in data_list]
            return [torch.FloatTensor(seq) for seq in normalized]
        
        # Normalize x1, x2, and y_history
        self.sequences['x1'] = normalize(self.sequences['x1'], 'x1', fit=fit_scaler)
        self.sequences['x2'] = normalize(self.sequences['x2'], 'x2', fit=fit_scaler)
        self.sequences['y_history'] = normalize(self.sequences['y_history'], 'y_history', fit=fit_scaler)
        
        # Normalize player features
        for p_idx in range(1, 11):
            p_key = f'p{p_idx}'
            for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']:
                self.sequences['players'][p_key][g_key] = normalize(
                    self.sequences['players'][p_key][g_key],
                    f'{p_key}_{g_key}',
                    fit=fit_scaler
                )
        
        # Normalize multi-step targets
        all_targets = np.concatenate(self.targets, axis=0)
        if fit_scaler:
            self.target_mean = all_targets.mean()
            self.target_std = all_targets.std() + 1e-8
            self.scaler_dict['target_mean'] = self.target_mean
            self.scaler_dict['target_std'] = self.target_std
        else:
            self.target_mean = self.scaler_dict.get('target_mean', 0)
            self.target_std = self.scaler_dict.get('target_std', 1)
        
        # Normalize each target sequence
        self.targets = [torch.FloatTensor((t - self.target_mean) / self.target_std) for t in self.targets]
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'x1': self.sequences['x1'][idx],
            'x2': self.sequences['x2'][idx],
            'y_history': self.sequences['y_history'][idx],  # Past gold differences
            'players': {
                p_key: {
                    g_key: self.sequences['players'][p_key][g_key][idx]
                    for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']
                }
                for p_key in [f'p{i}' for i in range(1, 11)]
            },
            'target': self.targets[idx],  # Multi-step targets [horizon]
            'seq_len': self.sequence_lengths[idx],
            'horizon': self.target_horizons[idx]
        }


class AutoregressiveHierarchicalPredictor(nn.Module):
    """
    Autoregressive Hierarchical Model
    y_hat = Ax_1 + Bx_2 + Cx_3 + Dy_history
    
    D: New LSTM component that processes past gold difference
    Allows model to learn gold momentum, trends, and mean reversion
    """
    
    def __init__(self, feature_groups, hidden_dim=32, lstm_layers=1, forecast_horizon=5):
        super(AutoregressiveHierarchicalPredictor, self).__init__()
        self.fg = feature_groups
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # A: LSTM for damage sequences
        self.A_lstm = nn.LSTM(
            input_size=len(self.fg.damage_features),
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.A_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, forecast_horizon)  # Output for each future step
        )
        
        # B: LSTM for vision sequences
        self.B_lstm = nn.LSTM(
            input_size=len(self.fg.vision_features),
            hidden_size=hidden_dim // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.B_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )
        
        # D: NEW - LSTM for gold difference history (autoregressive component)
        self.D_lstm = nn.LSTM(
            input_size=1,  # Just the gold difference value
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.D_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, forecast_horizon)  # Predict momentum
        )
        
        # G1-G5: LSTMs for player stat groups (same as before)
        self.G1_lstm = nn.LSTM(len(self.fg.offensive_stats), hidden_dim, lstm_layers, batch_first=True, bidirectional=True)
        self.G1_head = nn.Linear(hidden_dim * 2, 8)
        
        self.G2_lstm = nn.LSTM(len(self.fg.defensive_stats), hidden_dim // 2, lstm_layers, batch_first=True, bidirectional=True)
        self.G2_head = nn.Linear(hidden_dim, 8)
        
        self.G3_lstm = nn.LSTM(len(self.fg.vamp_stats), hidden_dim // 2, lstm_layers, batch_first=True, bidirectional=True)
        self.G3_head = nn.Linear(hidden_dim, 8)
        
        self.G4_lstm = nn.LSTM(len(self.fg.resource_stats), hidden_dim // 4, lstm_layers, batch_first=True, bidirectional=True)
        self.G4_head = nn.Linear(hidden_dim // 2, 8)
        
        self.G5_lstm = nn.LSTM(len(self.fg.mobility_stats), hidden_dim // 4, lstm_layers, batch_first=True, bidirectional=True)
        self.G5_head = nn.Linear(hidden_dim // 2, 8)
        
        # F: Player aggregator
        self.F = nn.Sequential(
            nn.Linear(5 * 8, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 16)
        )
        
        # D & E: Team aggregation with attention
        self.team_D_query = nn.Linear(16, 16)
        self.team_D_key = nn.Linear(16, 16)
        self.team_D_value = nn.Linear(16, 16)
        self.team_D_out = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 16)
        )
        
        self.team_E_query = nn.Linear(16, 16)
        self.team_E_key = nn.Linear(16, 16)
        self.team_E_value = nn.Linear(16, 16)
        self.team_E_out = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 16)
        )
        
        # C: Team interaction processor
        self.C = nn.Sequential(
            nn.Linear(32, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, forecast_horizon)
        )
        
        # Final combiner: y_hat = Ax_1 + Bx_2 + Cx_3 + Dy_history
        self.final_combiner = nn.Sequential(
            nn.Linear(4 * forecast_horizon, hidden_dim * 2),  # 4 components now
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, forecast_horizon)  # Multi-step output
        )
    
    def process_variable_lstm(self, lstm, head, input_seq, seq_lens):
        """Process variable-length sequences through LSTM"""
        batch_size = input_seq.shape[0]
        
        # Sort by length
        seq_lens_sorted, perm_idx = seq_lens.sort(descending=True)
        input_sorted = input_seq[perm_idx]
        
        # Pack padded sequences
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            input_sorted, seq_lens_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # LSTM forward
        packed_output, (h_n, c_n) = lstm(packed_input)
        
        # Unpack
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Get last valid timestep for each sequence
        last_outputs = []
        for i, length in enumerate(seq_lens_sorted):
            last_outputs.append(output[i, length-1, :])
        last_outputs = torch.stack(last_outputs)
        
        # Restore original order
        _, unperm_idx = perm_idx.sort()
        last_outputs = last_outputs[unperm_idx]
        
        # Process through head
        return head(last_outputs)
    
    def attention_pool(self, player_embeddings, query_layer, key_layer, value_layer, out_layer):
        """Attention pooling for team aggregation"""
        Q = query_layer(player_embeddings)
        K = key_layer(player_embeddings)
        V = value_layer(player_embeddings)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / (16 ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attended = torch.bmm(attn_weights, V)
        pooled = attended.mean(dim=1)
        return out_layer(pooled)
    
    def forward(self, batch):
        """
        Forward pass with autoregressive component
        Predicts multiple steps ahead: [t+1, t+2, ..., t+horizon]
        """
        batch_size = batch['x1'].shape[0]
        seq_lens = batch['seq_lens']
        
        # A: Damage contribution (multi-step forecast)
        damage_scores = self.process_variable_lstm(
            self.A_lstm, self.A_head, batch['x1'], seq_lens
        )  # [batch, horizon]
        
        # B: Vision contribution (multi-step forecast)
        vision_scores = self.process_variable_lstm(
            self.B_lstm, self.B_head, batch['x2'], seq_lens
        )  # [batch, horizon]
        
        # D: Gold history contribution (AUTOREGRESSIVE - captures momentum!)
        y_hist_input = batch['y_history'].unsqueeze(-1)  # [batch, seq_len, 1]
        gold_momentum = self.process_variable_lstm(
            self.D_lstm, self.D_head, y_hist_input, seq_lens
        )  # [batch, horizon]
        
        # Process all 10 players' sequences
        player_embeddings = []
        for player_idx in range(1, 11):
            p_key = f'p{player_idx}'
            player_data = batch['players'][p_key]
            
            # Process stat groups with LSTMs
            g1_emb = self.process_variable_lstm(self.G1_lstm, self.G1_head, player_data['g1'], seq_lens)
            g2_emb = self.process_variable_lstm(self.G2_lstm, self.G2_head, player_data['g2'], seq_lens)
            g3_emb = self.process_variable_lstm(self.G3_lstm, self.G3_head, player_data['g3'], seq_lens)
            g4_emb = self.process_variable_lstm(self.G4_lstm, self.G4_head, player_data['g4'], seq_lens)
            g5_emb = self.process_variable_lstm(self.G5_lstm, self.G5_head, player_data['g5'], seq_lens)
            
            # Concatenate and aggregate
            g_concat = torch.cat([g1_emb, g2_emb, g3_emb, g4_emb, g5_emb], dim=1)
            player_emb = self.F(g_concat)
            player_embeddings.append(player_emb)
        
        # Stack and split into teams
        all_players = torch.stack(player_embeddings, dim=1)  # [batch, 10, 16]
        team1_players = all_players[:, :5, :]
        team2_players = all_players[:, 5:, :]
        
        # Team aggregation
        t1 = self.attention_pool(team1_players, self.team_D_query, self.team_D_key, self.team_D_value, self.team_D_out)
        t2 = self.attention_pool(team2_players, self.team_E_query, self.team_E_key, self.team_E_value, self.team_E_out)
        
        # C: Team interaction
        team_concat = torch.cat([t1, t2], dim=1)
        team_scores = self.C(team_concat)  # [batch, horizon]
        
        # Final combination: y_hat = Ax_1 + Bx_2 + Cx_3 + Dy_history
        # Concatenate all 4 components
        all_scores = torch.cat([damage_scores, vision_scores, team_scores, gold_momentum], dim=1)  # [batch, 4*horizon]
        y_hat = self.final_combiner(all_scores)  # [batch, horizon]
        
        return y_hat


def collate_autoregressive(batch):
    """Custom collate for autoregressive variable-length sequences"""
    max_len = max([b['seq_len'] for b in batch])
    batch_size = len(batch)
    forecast_horizon = batch[0]['target'].shape[0]
    
    # Get dimensions
    x1_dim = batch[0]['x1'].shape[1]
    x2_dim = batch[0]['x2'].shape[1]
    
    # Initialize padded tensors
    x1_padded = torch.zeros(batch_size, max_len, x1_dim)
    x2_padded = torch.zeros(batch_size, max_len, x2_dim)
    y_history_padded = torch.zeros(batch_size, max_len, 1)
    
    # Player features
    player_padded = {}
    for p_key in [f'p{i}' for i in range(1, 11)]:
        player_padded[p_key] = {}
        for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']:
            g_dim = batch[0]['players'][p_key][g_key].shape[1]
            player_padded[p_key][g_key] = torch.zeros(batch_size, max_len, g_dim)
    
    targets = []
    seq_lens = []
    
    # Fill in the data
    for i, sample in enumerate(batch):
        seq_len = sample['seq_len']
        seq_lens.append(seq_len)
        
        # Pad inputs
        x1_padded[i, :seq_len, :] = sample['x1']
        x2_padded[i, :seq_len, :] = sample['x2']
        y_history_padded[i, :seq_len, :] = sample['y_history'].unsqueeze(-1)
        
        # Pad player features
        for p_key in [f'p{j}' for j in range(1, 11)]:
            for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']:
                player_padded[p_key][g_key][i, :seq_len, :] = sample['players'][p_key][g_key]
        
        targets.append(sample['target'])
    
    return {
        'x1': x1_padded,
        'x2': x2_padded,
        'y_history': y_history_padded.squeeze(-1),  # [batch, seq_len]
        'players': player_padded,
        'target': torch.stack(targets),  # [batch, horizon]
        'seq_lens': torch.LongTensor(seq_lens)
    }


def train_autoregressive_model(model, train_loader, val_loader, epochs=50, lr=0.001, target_std=1.0):
    """Train the endgame forecasting model"""
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print(f"\n{'='*60}")
    print("TRAINING ENDGAME FORECASTING MODEL")
    print(f"{'='*60}")
    print(f"Optimizer: Adam with weight decay=1e-5")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Loss function: Huber (delta=1.0)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Task: Predict last {model.forecast_horizon} frames of each game")
    
    train_losses = []
    val_losses = []
    
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # Training
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, unit="batch")
        for batch in train_pbar:
            optimizer.zero_grad()
            
            predictions = model(batch)
            loss = criterion(predictions, batch['target'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, unit="batch")
        with torch.no_grad():
            for batch in val_pbar:
                predictions = model(batch)
                loss = criterion(predictions, batch['target'])
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Denormalize for display
        train_loss_denorm = avg_train_loss * (target_std ** 2)
        val_loss_denorm = avg_val_loss * (target_std ** 2)
        
        epoch_pbar.set_postfix({
            'train_rmse': f'{np.sqrt(train_loss_denorm):.2f}',
            'val_rmse': f'{np.sqrt(val_loss_denorm):.2f}'
        })
    
    return train_losses, val_losses


if __name__ == "__main__":
    fg = FeatureGroups()
    
    # Create autoregressive datasets
    print("\nCreating autoregressive datasets...")
    forecast_horizon = 5  # Predict 5 steps ahead
    
    train_dataset = AutoregressiveGameDataset(
        train_data, fg,
        forecast_horizon=forecast_horizon,
        scaler_dict=None,
        fit_scaler=True,
        min_frames=10
    )
    val_dataset = AutoregressiveGameDataset(
        val_data, fg,
        forecast_horizon=forecast_horizon,
        scaler_dict=train_dataset.scaler_dict,
        fit_scaler=False,
        min_frames=10
    )
    test_dataset = AutoregressiveGameDataset(
        test_data, fg,
        forecast_horizon=forecast_horizon,
        scaler_dict=train_dataset.scaler_dict,
        fit_scaler=False,
        min_frames=10
    )
    
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_autoregressive)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_autoregressive)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_autoregressive)
    
    # Create model
    model = AutoregressiveHierarchicalPredictor(fg, hidden_dim=32, lstm_layers=1, forecast_horizon=forecast_horizon)
    
    print(f"\nEndgame forecasting model created:")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Architecture: y_hat = Ax_1 + Bx_2 + Cx_3 + Dy_history")
    print(f"  - D component: LSTM on past gold differences (captures momentum)")
    print(f"  - Task: Predict last {forecast_horizon} frames using all prior data")
    print(f"  - Training: {len(train_dataset)} games, Val: {len(val_dataset)} games, Test: {len(test_dataset)} games")
    
    # Train
    target_std = train_dataset.target_std
    print(f"  - Target std: {target_std:.2f}")
    
    train_losses, val_losses = train_autoregressive_model(
        model, train_loader, val_loader,
        epochs=100, lr=0.001, target_std=target_std
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_dict': train_dataset.scaler_dict,
        'forecast_horizon': forecast_horizon
    }, 'results/autoregressive_hierarchical_model.pth')
    
    print("\n✓ Model saved to: results/autoregressive_hierarchical_model.pth")
    
    # Evaluate
    print("\nEvaluating on test set...")
    model.eval()
    
    criterion = nn.HuberLoss(delta=1.0)
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            predictions = model(batch)
            loss = criterion(predictions, batch['target'])
            test_loss += loss.item()
            
            all_predictions.append(predictions.numpy())
            all_targets.append(batch['target'].numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    avg_test_loss = test_loss / len(test_loader)
    test_rmse = np.sqrt(avg_test_loss * (target_std ** 2))
    
    print(f"\n{'='*60}")
    print("TEST RESULTS (Multi-Step Forecasting)")
    print(f"{'='*60}")
    print(f"Test RMSE: {test_rmse:.2f} gold")
    
    # Calculate RMSE for each horizon
    for h in range(forecast_horizon):
        h_rmse = np.sqrt(np.mean((all_predictions[:, h] - all_targets[:, h]) ** 2)) * target_std
        print(f"  Step {h+1} ahead RMSE: {h_rmse:.2f} gold")
    
    # Plot loss curves
    print("\nPlotting loss curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0]
    epochs_range = range(1, len(train_losses) + 1)
    
    # Denormalize losses for better interpretability
    train_rmse = [np.sqrt(loss * (target_std ** 2)) for loss in train_losses]
    val_rmse = [np.sqrt(loss * (target_std ** 2)) for loss in val_losses]
    
    ax1.plot(epochs_range, train_rmse, 'b-', label='Training RMSE', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs_range, val_rmse, 'r-', label='Validation RMSE', linewidth=2, marker='s', markersize=4)
    ax1.axhline(y=test_rmse, color='g', linestyle='--', linewidth=2, label=f'Test RMSE: {test_rmse:.2f}')
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE (Gold)', fontsize=12, fontweight='bold')
    ax1.set_title('Endgame Forecasting: Training & Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RMSE by Forecast Horizon
    ax2 = axes[1]
    horizon_rmses = []
    for h in range(forecast_horizon):
        h_rmse = np.sqrt(np.mean((all_predictions[:, h] - all_targets[:, h]) ** 2)) * target_std
        horizon_rmses.append(h_rmse)
    
    horizon_labels = [f'N-{forecast_horizon-h-1}' for h in range(forecast_horizon)]
    colors = plt.cm.viridis(np.linspace(0, 1, forecast_horizon))
    
    bars = ax2.bar(range(1, forecast_horizon + 1), horizon_rmses, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('Frame Position (Endgame)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE (Gold)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Test RMSE by Frame (Last {forecast_horizon} Frames)', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(1, forecast_horizon + 1))
    ax2.set_xticklabels(horizon_labels)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, rmse) in enumerate(zip(bars, horizon_rmses)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Endgame Forecasting Model (y_hat = Ax₁ + Bx₂ + Cx₃ + Dy_history)', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('results/endgame_forecasting_loss.png', dpi=300, bbox_inches='tight')
    print(f"✓ Loss plot saved to: results/endgame_forecasting_loss.png")
    plt.close()
    
    print(f"\n{'='*60}")
    print("ENDGAME FORECASTING MODEL COMPLETE")
    print(f"{'='*60}")
    print("\nKey features:")
    print("  ✓ Uses past gold differences as input (D component)")
    print("  ✓ Predicts last 5 frames of each game using all prior data")
    print("  ✓ Captures gold momentum and trends toward endgame")
    print("  ✓ Maintains hierarchical structure: y_hat = Ax_1 + Bx_2 + Cx_3 + Dy_history")
    print("  ✓ 1 training sequence per game (not rolling windows)")
    print(f"\nGenerated files:")
    print(f"  - results/autoregressive_hierarchical_model.pth (model checkpoint)")
    print(f"  - results/endgame_forecasting_loss.png (loss curves)")

