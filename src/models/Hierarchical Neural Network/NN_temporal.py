#!/usr/bin/env python3
"""
Temporal Hierarchical Neural Network for League of Legends Gold Difference Prediction
With LSTM/GRU for sequential modeling
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
df_clean = full_data.drop(columns=[f for f in leakage_features if f in full_data.columns], errors='ignore')
print(f"Cleaned data shape: {df_clean.shape}")

# Prepare features and labels
label_col = 'Total_Gold_Difference'
exclude_cols = ['match_id', 'frame_idx', 'timestamp', label_col]
feature_columns = [col for col in df_clean.columns if col not in exclude_cols]

print(f"\nNumber of features: {len(feature_columns)}")


class SequentialGameDataset(Dataset):
    """
    Dataset that creates sequences from match data for temporal modeling
    Each sample is a sequence of frames from a single match
    """
    
    def __init__(self, dataframe, feature_cols, label_col, sequence_length=10, stride=5):
        """
        Args:
            dataframe: Full dataset
            feature_cols: List of feature column names
            label_col: Target column name
            sequence_length: Number of frames in each sequence
            stride: How many frames to skip between sequences (for overlapping windows)
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.sequence_length = sequence_length
        self.stride = stride
        
        self.sequences = []
        self.targets = []
        self.match_ids = []
        
        # Group by match and create sequences
        for match_id, match_data in dataframe.groupby('match_id'):
            match_data = match_data.sort_values('frame_idx').reset_index(drop=True)
            
            # Skip matches that are too short
            if len(match_data) < sequence_length:
                continue
            
            # Create overlapping sequences
            for start_idx in range(0, len(match_data) - sequence_length + 1, stride):
                seq_data = match_data.iloc[start_idx:start_idx + sequence_length]
                
                # Features: [sequence_length, num_features]
                X = seq_data[feature_cols].values
                # Target: predict gold diff at the LAST frame of sequence
                y = seq_data[label_col].iloc[-1]
                
                self.sequences.append(torch.FloatTensor(X))
                self.targets.append(torch.FloatTensor([y]))
                self.match_ids.append(match_id)
        
        # Normalize features
        all_features = torch.stack(self.sequences)  # [N, seq_len, features]
        self.feature_mean = all_features.mean(dim=(0, 1))
        self.feature_std = all_features.std(dim=(0, 1)) + 1e-8
        
        self.target_mean = torch.stack(self.targets).mean()
        self.target_std = torch.stack(self.targets).std() + 1e-8
        
        # Apply normalization
        for i in range(len(self.sequences)):
            self.sequences[i] = (self.sequences[i] - self.feature_mean) / self.feature_std
            self.targets[i] = (self.targets[i] - self.target_mean) / self.target_std
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class TemporalHierarchicalPredictor(nn.Module):
    """
    Temporal model with LSTM for sequential gold difference prediction
    Architecture:
    1. LSTM processes temporal sequences
    2. Attention mechanism over time steps
    3. MLP for final prediction
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super(TemporalHierarchicalPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism over time steps
        self.attention_query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_key = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_value = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, sequence_length, input_dim]
        Returns:
            predictions: [batch_size, 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch, seq, hidden*2]
        
        # Apply attention over time steps
        Q = self.attention_query(lstm_out)  # [batch, seq, hidden]
        K = self.attention_key(lstm_out)    # [batch, seq, hidden]
        V = self.attention_value(lstm_out)  # [batch, seq, hidden]
        
        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # [batch, seq, seq]
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.bmm(attn_weights, V)  # [batch, seq, hidden]
        
        # Use the last time step's attended representation
        final_repr = attended[:, -1, :]  # [batch, hidden]
        
        # Final prediction
        output = self.predictor(final_repr)  # [batch, 1]
        
        return output


def train_temporal_model(model, train_loader, val_loader, epochs=50, lr=0.001, target_std=1.0):
    """Train the temporal model"""
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"\n{'='*60}")
    print("TRAINING TEMPORAL HIERARCHICAL NEURAL NETWORK")
    print(f"{'='*60}")
    print(f"Optimizer: Adam with weight decay=1e-5")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Loss function: Huber (delta=1.0)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    # Training loop
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, unit="batch")
        for batch_x, batch_y in train_pbar:
            optimizer.zero_grad()
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, unit="batch")
        with torch.no_grad():
            for batch_x, batch_y in val_pbar:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Denormalize losses for interpretability
        train_loss_denorm = avg_train_loss * (target_std ** 2)
        val_loss_denorm = avg_val_loss * (target_std ** 2)
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'train_rmse': f'{np.sqrt(train_loss_denorm):.2f}',
            'val_rmse': f'{np.sqrt(val_loss_denorm):.2f}'
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'results/best_temporal_model.pth')
    
    print(f"\n✓ Best model saved to: results/best_temporal_model.pth")
    return train_losses, val_losses


def evaluate_temporal_model(model, test_loader, target_std=1.0):
    """Evaluate the temporal model"""
    
    model.eval()
    criterion = nn.HuberLoss(delta=1.0)
    
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    print("\nEvaluating on test set...")
    test_pbar = tqdm(test_loader, desc="Testing", unit="batch")
    
    with torch.no_grad():
        for batch_x, batch_y in test_pbar:
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            test_loss += loss.item()
            
            all_predictions.extend(predictions.numpy())
            all_targets.extend(batch_y.numpy())
            
            test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_test_loss = test_loss / len(test_loader)
    test_loss_denorm = avg_test_loss * (target_std ** 2)
    
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Loss (Huber): {avg_test_loss:.4f}")
    print(f"Test RMSE: {np.sqrt(test_loss_denorm):.2f} gold")
    
    return avg_test_loss, all_predictions, all_targets


def plot_temporal_losses(train_losses, val_losses, test_loss, target_std=1.0, save_path="results/temporal_nn_loss.png"):
    """Plot training and testing losses"""
    
    # Denormalize losses
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
    plt.title('Temporal LSTM Model - Training and Testing Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add text annotation
    final_train = train_rmse[-1]
    final_val = val_rmse[-1]
    plt.text(0.02, 0.98, f'Final Train RMSE: {final_train:.2f}\nFinal Val RMSE: {final_val:.2f}\nTest RMSE: {test_rmse:.2f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Loss plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Split data by match_id
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
    
    train_df = df_clean[df_clean['match_id'].isin(train_matches)].copy()
    val_df = df_clean[df_clean['match_id'].isin(val_matches)].copy()
    test_df = df_clean[df_clean['match_id'].isin(test_matches)].copy()
    
    print(f"\nData split:")
    print(f"Train: {train_df.shape} ({len(train_matches)} matches)")
    print(f"Val: {val_df.shape} ({len(val_matches)} matches)")
    print(f"Test: {test_df.shape} ({len(test_matches)} matches)")
    
    # Create sequential datasets
    print("\nCreating sequential datasets...")
    sequence_length = 10  # Use 10 frames to predict next frame
    
    train_dataset = SequentialGameDataset(train_df, feature_columns, label_col, sequence_length=sequence_length, stride=3)
    val_dataset = SequentialGameDataset(val_df, feature_columns, label_col, sequence_length=sequence_length, stride=5)
    test_dataset = SequentialGameDataset(test_df, feature_columns, label_col, sequence_length=sequence_length, stride=5)
    
    # Use normalization from train set
    val_dataset.feature_mean = train_dataset.feature_mean
    val_dataset.feature_std = train_dataset.feature_std
    val_dataset.target_mean = train_dataset.target_mean
    val_dataset.target_std = train_dataset.target_std
    
    test_dataset.feature_mean = train_dataset.feature_mean
    test_dataset.feature_std = train_dataset.feature_std
    test_dataset.target_mean = train_dataset.target_mean
    test_dataset.target_std = train_dataset.target_std
    
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Create model
    model = TemporalHierarchicalPredictor(
        input_dim=len(feature_columns),
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )
    
    print(f"\nModel architecture created:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    target_std = train_dataset.target_std.item()
    print(f"  Target std: {target_std:.2f}")
    
    train_losses, val_losses = train_temporal_model(
        model, train_loader, val_loader,
        epochs=50, lr=0.001, target_std=target_std
    )
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('results/best_temporal_model.pth'))
    
    # Evaluate
    test_loss, predictions, targets = evaluate_temporal_model(model, test_loader, target_std=target_std)
    
    # Plot losses
    plot_temporal_losses(train_losses, val_losses, test_loss, target_std=target_std)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")

