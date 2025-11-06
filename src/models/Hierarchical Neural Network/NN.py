#!/usr/bin/env python3
"""
Hierarchical Neural Network for League of Legends Gold Difference Prediction
Custom architecture: y_hat = Ax_1 + Bx_2 + Cx_3
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the parquet data
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)
print("Loading data from parquet file...")
full_data = pd.read_parquet("Data/processed/featured_data.parquet")

print(f"✓ Full data loaded: {full_data.shape}")
print(f"  Columns: {list(full_data.columns[:10])}... (showing first 10)")

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

print(f"\nRemoving {len(leakage_features)} data leakage features...")
full_data_clean = full_data.drop(columns=[f for f in leakage_features if f in full_data.columns], errors='ignore')
print(f"Cleaned data shape: {full_data_clean.shape}")

# Split data by match_id (70% train, 15% val, 15% test)
unique_matches = full_data_clean['match_id'].unique()
n_matches = len(unique_matches)
print(f"\nTotal unique matches: {n_matches}")

np.random.seed(42)
shuffled_matches = np.random.permutation(unique_matches)

n_train = int(n_matches * 0.7)
n_val = int(n_matches * 0.15)

train_matches = shuffled_matches[:n_train]
val_matches = shuffled_matches[n_train:n_train + n_val]
test_matches = shuffled_matches[n_train + n_val:]

train_data = full_data_clean[full_data_clean['match_id'].isin(train_matches)].copy()
val_data = full_data_clean[full_data_clean['match_id'].isin(val_matches)].copy()
test_data = full_data_clean[full_data_clean['match_id'].isin(test_matches)].copy()

print(f"\nData split:")
print(f"Train: {train_data.shape} ({len(train_matches)} matches)")
print(f"Val: {val_data.shape} ({len(val_matches)} matches)")
print(f"Test: {test_data.shape} ({len(test_matches)} matches)")

# Define feature groups according to the hierarchical structure
class FeatureGroups:
    """Define all feature groups for hierarchical model"""
    
    # x_1: Damage features (Team-level)
    damage_features = [
        'Magic_Damage_Done_Diff',
        'Magic_Damage_Done_To_Champions_Diff',
        'Physical_Damage_Done_Diff',
        'Physical_Damage_Done_To_Champions_Diff',
        'True_Damage_Done_Diff',
        'True_Damage_Done_To_Champions_Diff'
    ]
    
    # x_2: Vision/Control features (Team-level)
    vision_features = [
        'Total_Ward_Placed_Difference',
        'Total_Ward_Killed_Difference',
        'Time_Enemy_Spent_Controlled_Difference'
    ]
    
    # For each player: 5 groups (g_1 to g_5)
    # g_1: Offensive stats
    offensive_stats = [
        'Attack_Damage', 'Attack_Speed', 'Ability_Power', 'Ability_Haste',
        'Armor_Pen_Percent', 'Armor_Pen', 'Bonus_Armor_Pen_Percent',
        'Magic_Pen', 'Magic_Pen_Percent', 'Bonus_Magic_Pen_Percent'
    ]
    
    # g_2: Defensive stats
    defensive_stats = [
        'Armor', 'Magic_Resist', 'Health_Percentage', 'Health_Regen'
    ]
    
    # g_3: Vampirism stats
    vamp_stats = [
        'Life_Steal', 'Omnivamp', 'Physical_Vamp', 'Spell_Vamp'
    ]
    
    # g_4: Resource stats
    resource_stats = [
        'Power_Percent', 'Power_Regen'
    ]
    
    # g_5: Mobility/Position stats
    mobility_stats = [
        'Movement_Speed', 'X_Position', 'Y_Position'
    ]

class FullGameTemporalDataset(Dataset):
    """
    Dataset where each sample is a FULL GAME (variable-length sequences)
    Uses entire match as one sequence, handles variable game lengths
    """
    
    def __init__(self, dataframe, feature_groups, scaler_dict=None, fit_scaler=False, min_frames=5):
        self.data = dataframe
        self.fg = feature_groups
        self.min_frames = min_frames
        self.scaler_dict = scaler_dict if scaler_dict is not None else {}
        
        # Store sequences (each item is a full game)
        self.sequences = {
            'x1': [], 'x2': [],
            'players': {f'p{i}': {'g1': [], 'g2': [], 'g3': [], 'g4': [], 'g5': []} for i in range(1, 11)}
        }
        self.targets = []
        self.match_ids = []
        self.sequence_lengths = []
        
        # Group by match - each match is ONE sequence
        print(f"Creating full-game sequences (variable length)...")
        for match_id, match_data in dataframe.groupby('match_id'):
            match_data = match_data.sort_values('frame_idx').reset_index(drop=True)
            
            # Skip very short games
            if len(match_data) < min_frames:
                continue
            
            # Use ENTIRE game as one sequence
            # Extract x1 features for full game
            x1_seq = match_data[self.fg.damage_features].values
            self.sequences['x1'].append(x1_seq)
            
            # Extract x2 features for full game
            x2_seq = match_data[self.fg.vision_features].values
            self.sequences['x2'].append(x2_seq)
            
            # Extract player features for full game
            for player_idx in range(1, 11):
                for g_key, g_features in [
                    ('g1', self.fg.offensive_stats),
                    ('g2', self.fg.defensive_stats),
                    ('g3', self.fg.vamp_stats),
                    ('g4', self.fg.resource_stats),
                    ('g5', self.fg.mobility_stats)
                ]:
                    cols = [f'Player{player_idx}_{stat}' for stat in g_features]
                    g_seq = match_data[cols].values
                    self.sequences['players'][f'p{player_idx}'][g_key].append(g_seq)
            
            # Target: gold difference at FINAL frame (who won?)
            target = match_data['Total_Gold_Difference'].iloc[-1]
            self.targets.append(target)
            
            self.match_ids.append(match_id)
            self.sequence_lengths.append(len(match_data))
        
        print(f"  Created {len(self.targets)} full-game sequences")
        print(f"  Game lengths range: {min(self.sequence_lengths)} to {max(self.sequence_lengths)} frames")
        
        # Convert to tensors and normalize
        self._convert_and_normalize(fit_scaler)
    
    def _convert_and_normalize(self, fit_scaler):
        """Convert to tensors and apply normalization"""
        
        # Normalize helper - handles variable-length sequences
        def normalize(data_list, key, fit=False):
            # Concatenate all sequences to compute global statistics
            all_data = np.concatenate(data_list, axis=0)  # [total_frames, features]
            
            if fit:
                mean = all_data.mean(axis=0)
                std = all_data.std(axis=0) + 1e-8
                self.scaler_dict[f'{key}_mean'] = mean
                self.scaler_dict[f'{key}_std'] = std
            else:
                mean = self.scaler_dict.get(f'{key}_mean', 0)
                std = self.scaler_dict.get(f'{key}_std', 1)
            
            # Normalize each sequence individually
            normalized = [(seq - mean) / std for seq in data_list]
            return [torch.FloatTensor(seq) for seq in normalized]
        
        # Normalize x1 and x2
        self.sequences['x1'] = normalize(self.sequences['x1'], 'x1', fit=fit_scaler)
        self.sequences['x2'] = normalize(self.sequences['x2'], 'x2', fit=fit_scaler)
        
        # Normalize player features
        for p_idx in range(1, 11):
            p_key = f'p{p_idx}'
            for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']:
                self.sequences['players'][p_key][g_key] = normalize(
                    self.sequences['players'][p_key][g_key],
                    f'{p_key}_{g_key}',
                    fit=fit_scaler
                )
        
        # Normalize targets
        targets_array = np.array(self.targets)
        if fit_scaler:
            self.target_mean = targets_array.mean()
            self.target_std = targets_array.std() + 1e-8
            self.scaler_dict['target_mean'] = self.target_mean
            self.scaler_dict['target_std'] = self.target_std
        else:
            self.target_mean = self.scaler_dict.get('target_mean', 0)
            self.target_std = self.scaler_dict.get('target_std', 1)
        
        targets_normalized = (targets_array - self.target_mean) / self.target_std
        self.targets = [torch.FloatTensor([t]) for t in targets_normalized]
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'x1': self.sequences['x1'][idx],  # [variable_seq_len, 6]
            'x2': self.sequences['x2'][idx],  # [variable_seq_len, 3]
            'players': {
                p_key: {
                    g_key: self.sequences['players'][p_key][g_key][idx]  # [variable_seq_len, num_features]
                    for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']
                }
                for p_key in [f'p{i}' for i in range(1, 11)]
            },
            'target': self.targets[idx],
            'seq_len': self.sequence_lengths[idx]
        }


class TemporalHierarchicalDataset(Dataset):
    """
    Temporal Dataset that creates FIXED-LENGTH sequences from match data
    Each sample is a SEQUENCE of frames, preserving time-series structure
    """
    
    def __init__(self, dataframe, feature_groups, sequence_length=10, stride=5, scaler_dict=None, fit_scaler=False):
        self.data = dataframe
        self.fg = feature_groups
        self.sequence_length = sequence_length
        self.stride = stride
        self.scaler_dict = scaler_dict if scaler_dict is not None else {}
        
        # Store sequences
        self.sequences = {
            'x1': [], 'x2': [],
            'players': {f'p{i}': {'g1': [], 'g2': [], 'g3': [], 'g4': [], 'g5': []} for i in range(1, 11)}
        }
        self.targets = []
        
        # Group by match and create sequences
        for match_id, match_data in dataframe.groupby('match_id'):
            match_data = match_data.sort_values('frame_idx').reset_index(drop=True)
            
            if len(match_data) < sequence_length:
                continue
            
            # Create overlapping sequences
            for start_idx in range(0, len(match_data) - sequence_length + 1, stride):
                seq_data = match_data.iloc[start_idx:start_idx + sequence_length]
                
                # Extract x1 features for this sequence
                x1_seq = seq_data[self.fg.damage_features].values
                self.sequences['x1'].append(x1_seq)
                
                # Extract x2 features for this sequence
                x2_seq = seq_data[self.fg.vision_features].values
                self.sequences['x2'].append(x2_seq)
                
                # Extract player features for this sequence
                for player_idx in range(1, 11):
                    for g_key, g_features in [
                        ('g1', self.fg.offensive_stats),
                        ('g2', self.fg.defensive_stats),
                        ('g3', self.fg.vamp_stats),
                        ('g4', self.fg.resource_stats),
                        ('g5', self.fg.mobility_stats)
                    ]:
                        cols = [f'Player{player_idx}_{stat}' for stat in g_features]
                        g_seq = seq_data[cols].values
                        self.sequences['players'][f'p{player_idx}'][g_key].append(g_seq)
                
                # Target: gold difference at LAST frame of sequence
                target = seq_data['Total_Gold_Difference'].iloc[-1]
                self.targets.append(target)
        
        # Convert to tensors and normalize
        self._convert_and_normalize(fit_scaler)
    
    def _convert_and_normalize(self, fit_scaler):
        """Convert to tensors and apply normalization"""
        
        # Normalize helper
        def normalize(data_list, key, fit=False):
            data_array = np.array(data_list)
            if fit:
                mean = data_array.mean(axis=(0, 1))
                std = data_array.std(axis=(0, 1)) + 1e-8
                self.scaler_dict[f'{key}_mean'] = mean
                self.scaler_dict[f'{key}_std'] = std
            else:
                mean = self.scaler_dict.get(f'{key}_mean', 0)
                std = self.scaler_dict.get(f'{key}_std', 1)
            
            normalized = (data_array - mean) / std
            return [torch.FloatTensor(seq) for seq in normalized]
        
        # Normalize x1 and x2
        self.sequences['x1'] = normalize(self.sequences['x1'], 'x1', fit=fit_scaler)
        self.sequences['x2'] = normalize(self.sequences['x2'], 'x2', fit=fit_scaler)
        
        # Normalize player features
        for p_idx in range(1, 11):
            p_key = f'p{p_idx}'
            for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']:
                self.sequences['players'][p_key][g_key] = normalize(
                    self.sequences['players'][p_key][g_key],
                    f'{p_key}_{g_key}',
                    fit=fit_scaler
                )
        
        # Normalize targets
        targets_array = np.array(self.targets)
        if fit_scaler:
            self.target_mean = targets_array.mean()
            self.target_std = targets_array.std() + 1e-8
            self.scaler_dict['target_mean'] = self.target_mean
            self.scaler_dict['target_std'] = self.target_std
        else:
            self.target_mean = self.scaler_dict.get('target_mean', 0)
            self.target_std = self.scaler_dict.get('target_std', 1)
        
        targets_normalized = (targets_array - self.target_mean) / self.target_std
        self.targets = [torch.FloatTensor([t]) for t in targets_normalized]
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'x1': self.sequences['x1'][idx],  # [seq_len, 6]
            'x2': self.sequences['x2'][idx],  # [seq_len, 3]
            'players': {
                p_key: {
                    g_key: self.sequences['players'][p_key][g_key][idx]  # [seq_len, num_features]
                    for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']
                }
                for p_key in [f'p{i}' for i in range(1, 11)]
            },
            'target': self.targets[idx]
        }


class HierarchicalLoLDataset(Dataset):
    """Custom Dataset that organizes features hierarchically (single frame, non-temporal)"""
    
    def __init__(self, dataframe, feature_groups, scaler_dict=None, fit_scaler=False):
        self.data = dataframe
        self.fg = feature_groups
        self.scaler_dict = scaler_dict if scaler_dict is not None else {}
        
        # Extract target (normalize it too)
        target_values = dataframe['Total_Gold_Difference'].values
        if fit_scaler:
            self.target_mean = target_values.mean()
            self.target_std = target_values.std()
        else:
            self.target_mean = scaler_dict.get('target_mean', 0)
            self.target_std = scaler_dict.get('target_std', 1)
        
        target_normalized = (target_values - self.target_mean) / (self.target_std + 1e-8)
        self.targets = torch.FloatTensor(target_normalized).unsqueeze(1)
        
        # Store normalization parameters
        self.scaler_dict['target_mean'] = self.target_mean
        self.scaler_dict['target_std'] = self.target_std
        
        # Normalize helper function
        def normalize_features(values, key, fit=False):
            if fit:
                mean = values.mean(axis=0)
                std = values.std(axis=0) + 1e-8
                self.scaler_dict[f'{key}_mean'] = mean
                self.scaler_dict[f'{key}_std'] = std
            else:
                mean = self.scaler_dict.get(f'{key}_mean', 0)
                std = self.scaler_dict.get(f'{key}_std', 1)
            return (values - mean) / std
        
        # Extract x_1: Damage features (normalized)
        x1_values = dataframe[self.fg.damage_features].values
        x1_normalized = normalize_features(x1_values, 'x1', fit=fit_scaler)
        self.x1 = torch.FloatTensor(x1_normalized)
        
        # Extract x_2: Vision features (normalized)
        x2_values = dataframe[self.fg.vision_features].values
        x2_normalized = normalize_features(x2_values, 'x2', fit=fit_scaler)
        self.x2 = torch.FloatTensor(x2_normalized)
        
        # Extract player features for 10 players
        self.player_features = []
        for player_idx in range(1, 11):
            player_data = {}
            
            # g_1: Offensive stats (normalized)
            g1_cols = [f'Player{player_idx}_{stat}' for stat in self.fg.offensive_stats]
            g1_values = dataframe[g1_cols].values
            g1_normalized = normalize_features(g1_values, f'p{player_idx}_g1', fit=fit_scaler)
            player_data['g1'] = torch.FloatTensor(g1_normalized)
            
            # g_2: Defensive stats (normalized)
            g2_cols = [f'Player{player_idx}_{stat}' for stat in self.fg.defensive_stats]
            g2_values = dataframe[g2_cols].values
            g2_normalized = normalize_features(g2_values, f'p{player_idx}_g2', fit=fit_scaler)
            player_data['g2'] = torch.FloatTensor(g2_normalized)
            
            # g_3: Vamp stats (normalized)
            g3_cols = [f'Player{player_idx}_{stat}' for stat in self.fg.vamp_stats]
            g3_values = dataframe[g3_cols].values
            g3_normalized = normalize_features(g3_values, f'p{player_idx}_g3', fit=fit_scaler)
            player_data['g3'] = torch.FloatTensor(g3_normalized)
            
            # g_4: Resource stats (normalized)
            g4_cols = [f'Player{player_idx}_{stat}' for stat in self.fg.resource_stats]
            g4_values = dataframe[g4_cols].values
            g4_normalized = normalize_features(g4_values, f'p{player_idx}_g4', fit=fit_scaler)
            player_data['g4'] = torch.FloatTensor(g4_normalized)
            
            # g_5: Mobility stats (normalized)
            g5_cols = [f'Player{player_idx}_{stat}' for stat in self.fg.mobility_stats]
            g5_values = dataframe[g5_cols].values
            g5_normalized = normalize_features(g5_values, f'p{player_idx}_g5', fit=fit_scaler)
            player_data['g5'] = torch.FloatTensor(g5_normalized)
            
            # Team assignment
            player_data['team'] = dataframe[f'Player{player_idx}_Team'].values[0]
            
            self.player_features.append(player_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return all feature groups for this sample
        player_data = {}
        for i, player in enumerate(self.player_features):
            player_data[f'p{i+1}'] = {
                'g1': player['g1'][idx],
                'g2': player['g2'][idx],
                'g3': player['g3'][idx],
                'g4': player['g4'][idx],
                'g5': player['g5'][idx]
            }
        
        return {
            'x1': self.x1[idx],
            'x2': self.x2[idx],
            'players': player_data,
            'target': self.targets[idx]
        }


class TemporalHierarchicalGoldPredictor(nn.Module):
    """
    Temporal Hierarchical Neural Network with LSTM
    Architecture: y_hat = Ax_1 + Bx_2 + Cx_3
    where each component processes SEQUENCES with LSTM
    
    Key features:
    - LSTM for temporal modeling of each exogenous variable group
    - Hierarchical structure preserved: damage, vision, player stats
    - Attention mechanism for team aggregation
    - Final linear combination of temporal features
    """
    
    def __init__(self, feature_groups, hidden_dim=32, lstm_layers=1):
        super(TemporalHierarchicalGoldPredictor, self).__init__()
        self.fg = feature_groups
        self.hidden_dim = hidden_dim
        
        # A: LSTM for damage features (x_1 sequences)
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
            nn.Linear(hidden_dim, 1)
        )
        
        # B: LSTM for vision features (x_2 sequences)
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
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # G1-G5: LSTMs for player stat groups
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
        
        # F: Aggregates player stat groups into player representation
        self.F = nn.Sequential(
            nn.Linear(5 * 8, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 16)
        )
        
        # D & E: Team aggregation with attention
        self.D_attn_query = nn.Linear(16, 16)
        self.D_attn_key = nn.Linear(16, 16)
        self.D_attn_value = nn.Linear(16, 16)
        self.D_out = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 16)
        )
        
        self.E_attn_query = nn.Linear(16, 16)
        self.E_attn_key = nn.Linear(16, 16)
        self.E_attn_value = nn.Linear(16, 16)
        self.E_out = nn.Sequential(
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
            nn.Linear(hidden_dim, 1)
        )
        
        # Final combiner
        self.final_combiner = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
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
    
    def process_variable_lstm(self, lstm, head, input_seq, seq_lens):
        """
        Process variable-length sequences through LSTM
        Args:
            lstm: LSTM layer
            head: Linear head to process LSTM output
            input_seq: [batch, max_seq_len, features] (padded)
            seq_lens: [batch] - actual lengths of each sequence
        Returns:
            output: [batch, output_dim]
        """
        batch_size = input_seq.shape[0]
        
        # Sort by length (required for pack_padded_sequence)
        seq_lens_sorted, perm_idx = seq_lens.sort(descending=True)
        input_sorted = input_seq[perm_idx]
        
        # Pack padded sequences
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            input_sorted, seq_lens_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # Process with LSTM
        packed_output, (h_n, c_n) = lstm(packed_input)
        
        # Unpack
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Get the last valid timestep for each sequence
        last_outputs = []
        for i, length in enumerate(seq_lens_sorted):
            last_outputs.append(output[i, length-1, :])
        last_outputs = torch.stack(last_outputs)  # [batch, hidden*2]
        
        # Restore original order
        _, unperm_idx = perm_idx.sort()
        last_outputs = last_outputs[unperm_idx]
        
        # Process through head
        return head(last_outputs)
    
    def forward(self, batch):
        """
        Forward pass with LSTM processing of VARIABLE-LENGTH sequences
        Input sequences: [batch_size, variable_seq_len, features] (padded)
        seq_lens: [batch_size] - actual length of each game
        """
        batch_size = batch['x1'].shape[0]
        seq_lens = batch['seq_lens']
        
        # A: Process damage sequences with LSTM (variable length)
        damage_score = self.process_variable_lstm(
            self.A_lstm, self.A_head, batch['x1'], seq_lens
        )  # [batch, 1]
        
        # B: Process vision sequences with LSTM (variable length)
        vision_score = self.process_variable_lstm(
            self.B_lstm, self.B_head, batch['x2'], seq_lens
        )  # [batch, 1]
        
        # Process all 10 players' sequences with LSTMs
        player_embeddings = []
        for player_idx in range(1, 11):
            p_key = f'p{player_idx}'
            player_data = batch['players'][p_key]
            
            # Process each stat group with LSTM (variable length)
            g1_emb = self.process_variable_lstm(
                self.G1_lstm, self.G1_head, player_data['g1'], seq_lens
            )  # [batch, 8]
            
            g2_emb = self.process_variable_lstm(
                self.G2_lstm, self.G2_head, player_data['g2'], seq_lens
            )  # [batch, 8]
            
            g3_emb = self.process_variable_lstm(
                self.G3_lstm, self.G3_head, player_data['g3'], seq_lens
            )  # [batch, 8]
            
            g4_emb = self.process_variable_lstm(
                self.G4_lstm, self.G4_head, player_data['g4'], seq_lens
            )  # [batch, 8]
            
            g5_emb = self.process_variable_lstm(
                self.G5_lstm, self.G5_head, player_data['g5'], seq_lens
            )  # [batch, 8]
            
            # Concatenate all group embeddings
            g_concat = torch.cat([g1_emb, g2_emb, g3_emb, g4_emb, g5_emb], dim=1)  # [batch, 40]
            
            # F: Aggregate into player representation
            player_emb = self.F(g_concat)  # [batch, 16]
            player_embeddings.append(player_emb)
        
        # Stack player embeddings
        all_players = torch.stack(player_embeddings, dim=1)  # [batch, 10, 16]
        
        # Split into teams
        team1_players = all_players[:, :5, :]  # [batch, 5, 16]
        team2_players = all_players[:, 5:, :]  # [batch, 5, 16]
        
        # D & E: Aggregate teams with attention
        t1 = self.attention_pool(team1_players, self.D_attn_query, self.D_attn_key, self.D_attn_value, self.D_out)
        t2 = self.attention_pool(team2_players, self.E_attn_query, self.E_attn_key, self.E_attn_value, self.E_out)
        
        # C: Process team interactions
        team_concat = torch.cat([t1, t2], dim=1)  # [batch, 32]
        team_score = self.C(team_concat)  # [batch, 1]
        
        # Final combination: y_hat = Ax_1 + Bx_2 + Cx_3
        combined_scores = torch.cat([damage_score, vision_score, team_score], dim=1)  # [batch, 3]
        y_hat = self.final_combiner(combined_scores)  # [batch, 1]
        
        return y_hat


class HierarchicalGoldPredictor(nn.Module):
    """
    Enhanced Hierarchical Neural Network with Multi-Layer Perceptrons
    Architecture: y_hat = Ax_1 + Bx_2 + Cx_3
    where x_3 = t_1 - t_2 (team difference)
    
    Key improvements:
    - Multi-layer MLPs for each component (2-3 hidden layers)
    - ReLU/LeakyReLU for better gradient flow
    - Interaction terms within player groups
    - Attention-like mechanism for team aggregation
    - Biases everywhere
    - Huber loss for robustness
    """
    
    def __init__(self, feature_groups, hidden_dim=64):
        super(HierarchicalGoldPredictor, self).__init__()
        self.fg = feature_groups
        self.hidden_dim = hidden_dim
        
        # A: Damage features MLP (deeper network)
        self.A = nn.Sequential(
            nn.Linear(len(self.fg.damage_features), hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # B: Vision features MLP
        self.B = nn.Sequential(
            nn.Linear(len(self.fg.vision_features), hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # G1-G5: Feature group MLPs (one for each stat category)
        # G1: Offensive stats (10 features -> complex interactions)
        self.G1 = nn.Sequential(
            nn.Linear(len(self.fg.offensive_stats), hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 8)  # Output 8 features for interaction
        )
        
        # G2: Defensive stats
        self.G2 = nn.Sequential(
            nn.Linear(len(self.fg.defensive_stats), hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 8)
        )
        
        # G3: Vamp stats
        self.G3 = nn.Sequential(
            nn.Linear(len(self.fg.vamp_stats), hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 8)
        )
        
        # G4: Resource stats
        self.G4 = nn.Sequential(
            nn.Linear(len(self.fg.resource_stats), hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 8)
        )
        
        # G5: Mobility stats
        self.G5 = nn.Sequential(
            nn.Linear(len(self.fg.mobility_stats), hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 8)
        )
        
        # F: Player score aggregator (combines 5 groups × 8 features = 40)
        # Now with cross-group interactions
        self.F = nn.Sequential(
            nn.Linear(5 * 8, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 16)  # 16-dim player representation
        )
        
        # Attention mechanism for team aggregation
        # D: Team 1 aggregator with attention
        self.D_query = nn.Linear(16, 16)
        self.D_key = nn.Linear(16, 16)
        self.D_value = nn.Linear(16, 16)
        self.D_out = nn.Sequential(
            nn.Linear(16, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 16)
        )
        
        # E: Team 2 aggregator with attention
        self.E_query = nn.Linear(16, 16)
        self.E_key = nn.Linear(16, 16)
        self.E_value = nn.Linear(16, 16)
        self.E_out = nn.Sequential(
            nn.Linear(16, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 16)
        )
        
        # C: Team difference processor (with interaction modeling)
        self.C = nn.Sequential(
            nn.Linear(32, hidden_dim),  # Concat both teams + difference
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Final combination layer (learns to weight A, B, C contributions)
        self.final_combiner = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def attention_pool(self, player_embeddings, query_layer, key_layer, value_layer, out_layer):
        """
        Attention-based pooling for team aggregation
        player_embeddings: [batch_size, 5, 16] - 5 players, 16-dim each
        """
        batch_size = player_embeddings.shape[0]
        
        # Compute query, key, value
        Q = query_layer(player_embeddings)  # [batch_size, 5, 16]
        K = key_layer(player_embeddings)    # [batch_size, 5, 16]
        V = value_layer(player_embeddings)  # [batch_size, 5, 16]
        
        # Attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (16 ** 0.5)  # [batch_size, 5, 5]
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention: weighted sum of values
        attended = torch.bmm(attn_weights, V)  # [batch_size, 5, 16]
        
        # Pool across players: mean pooling
        pooled = attended.mean(dim=1)  # [batch_size, 16]
        
        # Final transformation
        output = out_layer(pooled)  # [batch_size, 16]
        return output
    
    def forward(self, batch):
        batch_size = batch['x1'].shape[0]
        
        # Compute Ax_1: Damage contribution (NO sigmoid, MLP does the work)
        damage_score = self.A(batch['x1'])  # [batch_size, 1]
        
        # Compute Bx_2: Vision contribution (NO sigmoid)
        vision_score = self.B(batch['x2'])  # [batch_size, 1]
        
        # Compute player representations for all 10 players
        player_embeddings = []
        for player_idx in range(1, 11):
            p_key = f'p{player_idx}'
            player_data = batch['players'][p_key]
            
            # Compute g_1 to g_5 embeddings (NO sigmoid, outputs 8-dim each)
            g1_emb = self.G1(player_data['g1'])  # [batch_size, 8]
            g2_emb = self.G2(player_data['g2'])  # [batch_size, 8]
            g3_emb = self.G3(player_data['g3'])  # [batch_size, 8]
            g4_emb = self.G4(player_data['g4'])  # [batch_size, 8]
            g5_emb = self.G5(player_data['g5'])  # [batch_size, 8]
            
            # Concatenate all group embeddings
            g_concat = torch.cat([g1_emb, g2_emb, g3_emb, g4_emb, g5_emb], dim=1)  # [batch_size, 40]
            
            # Combine using F: creates 16-dim player representation
            player_emb = self.F(g_concat)  # [batch_size, 16]
            player_embeddings.append(player_emb)
        
        # Stack all player embeddings
        all_players = torch.stack(player_embeddings, dim=1)  # [batch_size, 10, 16]
        
        # Split into team 1 (players 1-5) and team 2 (players 6-10)
        team1_players = all_players[:, :5, :]  # [batch_size, 5, 16]
        team2_players = all_players[:, 5:, :]  # [batch_size, 5, 16]
        
        # Compute team representations with attention
        t1 = self.attention_pool(team1_players, self.D_query, self.D_key, self.D_value, self.D_out)  # [batch_size, 16]
        t2 = self.attention_pool(team2_players, self.E_query, self.E_key, self.E_value, self.E_out)  # [batch_size, 16]
        
        # Compute team difference AND concatenate both teams for richer interaction
        team_concat = torch.cat([t1, t2], dim=1)  # [batch_size, 32]
        
        # Process team interactions through C
        team_score = self.C(team_concat)  # [batch_size, 1]
        
        # Combine all three scores (A, B, C) with learned weighting
        combined_scores = torch.cat([damage_score, vision_score, team_score], dim=1)  # [batch_size, 3]
        y_hat = self.final_combiner(combined_scores)  # [batch_size, 1]
        
        return y_hat


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, target_std=1.0):
    """Train the enhanced hierarchical model using Adam + Huber loss"""
    
    # Huber loss is more robust to outliers than MSE
    criterion = nn.HuberLoss(delta=1.0)
    # Adam optimizer for better convergence with complex model
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print(f"\n{'='*60}")
    print("TRAINING ENHANCED HIERARCHICAL NEURAL NETWORK")
    print(f"{'='*60}")
    print(f"Optimizer: Adam with weight decay=1e-5")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Loss function: Huber (delta=1.0)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    train_losses = []
    val_losses = []
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, unit="batch")
        for batch in train_pbar:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch)
            loss = criterion(predictions, batch['target'])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        # Progress bar for validation batches
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, unit="batch")
        with torch.no_grad():
            for batch in val_pbar:
                predictions = model(batch)
                loss = criterion(predictions, batch['target'])
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress (denormalize losses for interpretability)
        train_loss_denorm = avg_train_loss * (target_std ** 2)
        val_loss_denorm = avg_val_loss * (target_std ** 2)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_rmse': f'{np.sqrt(train_loss_denorm):.2f}',
            'val_rmse': f'{np.sqrt(val_loss_denorm):.2f}'
        })
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, target_std=1.0):
    """Evaluate the model on test set"""
    
    model.eval()
    criterion = nn.MSELoss()
    
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    print("\nEvaluating on test set...")
    test_pbar = tqdm(test_loader, desc="Testing", unit="batch")
    
    with torch.no_grad():
        for batch in test_pbar:
            predictions = model(batch)
            loss = criterion(predictions, batch['target'])
            test_loss += loss.item()
            
            all_predictions.extend(predictions.numpy())
            all_targets.extend(batch['target'].numpy())
            
            test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_test_loss = test_loss / len(test_loader)
    test_loss_denorm = avg_test_loss * (target_std ** 2)
    
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Loss (MSE): {avg_test_loss:.4f}")
    print(f"Test RMSE: {np.sqrt(test_loss_denorm):.2f} gold")
    
    return avg_test_loss, all_predictions, all_targets


def plot_losses(train_losses, val_losses, test_loss, target_std=1.0, save_path="results/training_loss.png"):
    """Plot training, validation, and test losses"""
    
    # Denormalize losses to gold scale
    train_losses_denorm = [loss * (target_std ** 2) for loss in train_losses]
    val_losses_denorm = [loss * (target_std ** 2) for loss in val_losses]
    test_loss_denorm = test_loss * (target_std ** 2)
    
    # Convert to RMSE
    train_rmse = [np.sqrt(loss) for loss in train_losses_denorm]
    val_rmse = [np.sqrt(loss) for loss in val_losses_denorm]
    test_rmse = np.sqrt(test_loss_denorm)
    
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_rmse, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_rmse, 'r-', label='Validation Loss', linewidth=2)
    plt.axhline(y=test_rmse, color='g', linestyle='--', linewidth=2, label=f'Test Loss: {test_rmse:.2f}')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RMSE (Gold Difference)', fontsize=12)
    plt.title('Hierarchical Neural Network - Training and Testing Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add text annotation for final values
    final_train = train_rmse[-1]
    final_val = val_rmse[-1]
    plt.text(0.02, 0.98, f'Final Train RMSE: {final_train:.2f}\nFinal Val RMSE: {final_val:.2f}\nTest RMSE: {test_rmse:.2f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Loss plot saved to: {save_path}")
    plt.close()


def print_model_summary(model):
    """Print model architecture summary"""
    
    print(f"\n{'='*60}")
    print("MODEL ARCHITECTURE SUMMARY")
    print(f"{'='*60}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    print("\nComponent Breakdown:")
    print(f"  A (Damage MLP): {sum(p.numel() for p in model.A.parameters()):,} params")
    print(f"  B (Vision MLP): {sum(p.numel() for p in model.B.parameters()):,} params")
    print(f"  G1 (Offensive MLP): {sum(p.numel() for p in model.G1.parameters()):,} params")
    print(f"  G2 (Defensive MLP): {sum(p.numel() for p in model.G2.parameters()):,} params")
    print(f"  G3 (Vamp MLP): {sum(p.numel() for p in model.G3.parameters()):,} params")
    print(f"  G4 (Resource MLP): {sum(p.numel() for p in model.G4.parameters()):,} params")
    print(f"  G5 (Mobility MLP): {sum(p.numel() for p in model.G5.parameters()):,} params")
    print(f"  F (Player aggregation): {sum(p.numel() for p in model.F.parameters()):,} params")
    
    d_params = (sum(p.numel() for p in model.D_query.parameters()) + 
                sum(p.numel() for p in model.D_key.parameters()) + 
                sum(p.numel() for p in model.D_value.parameters()) + 
                sum(p.numel() for p in model.D_out.parameters()))
    print(f"  D (Team 1 attention): {d_params:,} params")
    
    e_params = (sum(p.numel() for p in model.E_query.parameters()) + 
                sum(p.numel() for p in model.E_key.parameters()) + 
                sum(p.numel() for p in model.E_value.parameters()) + 
                sum(p.numel() for p in model.E_out.parameters()))
    print(f"  E (Team 2 attention): {e_params:,} params")
    
    print(f"  C (Team interaction MLP): {sum(p.numel() for p in model.C.parameters()):,} params")
    print(f"  Final combiner: {sum(p.numel() for p in model.final_combiner.parameters()):,} params")
    
    print(f"\n{'='*60}")


def collate_variable_length(batch):
    """
    Custom collate function for variable-length full-game sequences
    Pads sequences to the longest in the batch
    """
    # Find max length in this batch
    max_len = max([b['seq_len'] for b in batch])
    batch_size = len(batch)
    
    # Get dimensions
    x1_dim = batch[0]['x1'].shape[1]
    x2_dim = batch[0]['x2'].shape[1]
    
    # Initialize padded tensors
    x1_padded = torch.zeros(batch_size, max_len, x1_dim)
    x2_padded = torch.zeros(batch_size, max_len, x2_dim)
    
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
        
        # Pad x1, x2
        x1_padded[i, :seq_len, :] = sample['x1']
        x2_padded[i, :seq_len, :] = sample['x2']
        
        # Pad player features
        for p_key in [f'p{j}' for j in range(1, 11)]:
            for g_key in ['g1', 'g2', 'g3', 'g4', 'g5']:
                player_padded[p_key][g_key][i, :seq_len, :] = sample['players'][p_key][g_key]
        
        targets.append(sample['target'])
    
    return {
        'x1': x1_padded,
        'x2': x2_padded,
        'players': player_padded,
        'target': torch.stack(targets),
        'seq_lens': torch.LongTensor(seq_lens)
    }


if __name__ == "__main__":
    # Create feature groups
    fg = FeatureGroups()
    
    # Create FULL-GAME datasets with variable-length sequences
    print("\nCreating full-game temporal datasets (variable length)...")
    
    train_dataset = FullGameTemporalDataset(
        train_data, fg, 
        scaler_dict=None, 
        fit_scaler=True,
        min_frames=10
    )
    val_dataset = FullGameTemporalDataset(
        val_data, fg,
        scaler_dict=train_dataset.scaler_dict,
        fit_scaler=False,
        min_frames=10
    )
    test_dataset = FullGameTemporalDataset(
        test_data, fg,
        scaler_dict=train_dataset.scaler_dict,
        fit_scaler=False,
        min_frames=10
    )
    
    print(f"Train games: {len(train_dataset)}")
    print(f"Val games: {len(val_dataset)}")
    print(f"Test games: {len(test_dataset)}")
    
    # Create dataloaders with custom collate function for variable-length sequences
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_variable_length)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_variable_length)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_variable_length)
    
    # Create TEMPORAL model with LSTM
    model = TemporalHierarchicalGoldPredictor(fg, hidden_dim=32, lstm_layers=1)
    
    print(f"\nFull-game temporal model architecture created:")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Using LSTM for temporal modeling of exogenous variables")
    print(f"  - Sequence type: FULL GAME (variable length)")
    
    # Train model for 150 epochs
    target_std = train_dataset.target_std
    print(f"  - Target std: {target_std:.2f}")
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=150, lr=0.001, target_std=target_std)
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_dict': train_dataset.scaler_dict,
        'sequence_length': 'FULL GAME (variable length)'
    }, 'results/temporal_hierarchical_model.pth')
    print("\n✓ Model saved to: results/temporal_hierarchical_model.pth")
    
    # Evaluate on test set
    test_loss, predictions, targets = evaluate_model(model, test_loader, target_std=target_std)
    
    # Plot losses
    plot_losses(train_losses, val_losses, test_loss, target_std=target_std, save_path="results/nn_training_loss.png")
    
    # Print model summary
    print_model_summary(model)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - LSTM-BASED TEMPORAL MODEL")
    print(f"{'='*60}")
    print("Model uses LSTM to capture temporal patterns in:")
    print("  - x_1: Damage evolution over time")
    print("  - x_2: Vision control evolution over time")
    print("  - g_1 to g_5: Player stat evolution for all 10 players")
    print("Then combines: y_hat = Ax_1 + Bx_2 + Cx_3")