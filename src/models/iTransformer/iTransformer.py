"""
iTransformer model for binary win prediction (Y_won).

iTransformer inverts the attention mechanism: instead of attending over time,
it attends over features. This treats each timestep as a token and features
as the sequence dimension, which is effective for time series where feature
interactions are more important than temporal relationships.

Key differences from hierarchical LSTM:
- No hierarchical structure (no player groups, team embeddings, etc.)
- Direct feature processing with inverted attention
- Simpler architecture focused on feature interactions
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ----------------------------------------------------------------------------------------------------------------------
# Feature specification (same as LSTM_One_Stage)
# ----------------------------------------------------------------------------------------------------------------------

ENCODER_LENGTH = 15
TARGET_COLUMN = "Y_won"
GROUP_COLUMN = "match_id"
TIME_COLUMN = "frame"

# X1: Team-level combat signals (damage, kills, assists) + economy (gold/XP differences)
TEAM_X1_FEATURES: List[str] = [
    "Total_Gold_Difference",
    "Total_Xp_Difference",
    "Magic_Damage_Done_Diff",
    "Magic_Damage_Done_To_Champions_Diff",
    "Magic_Damage_Taken_Diff",
    "Physical_Damage_Done_Diff",
    "Physical_Damage_Done_To_Champions_Diff",
    "Physical_Damage_Taken_Diff",
    "Total_Damage_Done_Diff",
    "Total_Damage_Done_To_Champions_Diff",
    "Total_Damage_Taken_Diff",
    "True_Damage_Done_Diff",
    "True_Damage_Done_To_Champions_Diff",
    "True_Damage_Taken_Diff",
    "Total_Kill_Difference",
    "Total_Assist_Difference",
]

# X2: Macro-vision signals (objectives, wards, rotations)
TEAM_X2_FEATURES: List[str] = [
    "Total_Jungle_Minions_Killed_Difference",
    "Total_Ward_Placed_Difference",
    "Total_Ward_Killed_Difference",
    "Time_Enemy_Spent_Controlled_Difference",
    "Elite_Monster_Killed_Difference",
    "Buildings_Taken_Difference",
]

# Player features
PLAYER_FEATURE_ORDER: List[str] = [
    "Attack_Damage",
    "Attack_Speed",
    "Ability_Power",
    "Armor_Pen_Percent",
    "Magic_Pen",
    "Magic_Pen_Percent",
    "Armor",
    "Magic_Resist",
    "Health_Percentage",
    "Health_Regen",
    "Life_Steal",
    "Power_Percent",
    "Power_Regen",
    "Movement_Speed",
    "X_Position",
    "Y_Position",
]

# Combine all features
ALL_FEATURE_COLUMNS: List[str] = TEAM_X1_FEATURES + TEAM_X2_FEATURES
for player_idx in range(1, 11):
    for feature_name in PLAYER_FEATURE_ORDER:
        ALL_FEATURE_COLUMNS.append(f"Player{player_idx}_{feature_name}")


# ----------------------------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class WinPredictionBatch:
    features: Tensor  # [batch, seq_len, feature_dim]
    target: Tensor
    match_ids: List[str]
    start_frames: List[int]


class WinPredictionDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        encoder_length: int = ENCODER_LENGTH,
        target_column: str = TARGET_COLUMN,
        group_column: str = GROUP_COLUMN,
        time_column: Optional[str] = TIME_COLUMN,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
        match_filter: Optional[set] = None,
        min_timesteps: Optional[int] = None,
    ):
        self.parquet_path = Path(parquet_path)
        self.encoder_length = encoder_length
        self.target_column = target_column
        self.group_column = group_column
        self.time_column = time_column if time_column else None
        self.scaler = scaler or StandardScaler()
        self.fit_scaler = fit_scaler
        self.match_filter = set(match_filter) if match_filter is not None else None
        self.min_timesteps = min_timesteps or encoder_length

        self.samples: List[Tuple[np.ndarray, int, str, int]] = []
        self._load()

    def _load(self) -> None:
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.parquet_path}")

        df = pd.read_parquet(self.parquet_path)

        if self.match_filter is not None:
            df = df[df[self.group_column].isin(self.match_filter)]

        # Filter to only columns that exist
        available_cols = [col for col in ALL_FEATURE_COLUMNS if col in df.columns]
        if len(available_cols) != len(ALL_FEATURE_COLUMNS):
            missing = set(ALL_FEATURE_COLUMNS) - set(available_cols)
            print(f"Warning: Missing columns: {missing}")

        required_cols = [self.target_column, self.group_column]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if self.time_column and self.time_column not in df.columns:
            self.time_column = None

        if self.fit_scaler:
            self.scaler.fit(df[available_cols].values)

        for match_id, match_df in df.groupby(self.group_column):
            if self.match_filter is not None and match_id not in self.match_filter:
                continue
            match_df = match_df.copy()
            if self.time_column:
                match_df = match_df.sort_values(self.time_column)

            if len(match_df) < self.min_timesteps:
                continue

            # Get target (Y_won) - should be constant per match
            target_value = match_df[self.target_column].iloc[0]
            if pd.isna(target_value):
                continue
            target = int(target_value) if isinstance(target_value, (bool, np.bool_)) else int(float(target_value) > 0)

            features = match_df[available_cols].values.astype(np.float32)
            features = self.scaler.transform(features).astype(np.float32)

            if len(match_df) < self.encoder_length:
                continue

            # Use the first ENCODER_LENGTH frames
            enc_start = 0
            enc_end = min(self.encoder_length, len(match_df))

            encoder_slice = features[enc_start:enc_end]
            if len(encoder_slice) < self.encoder_length:
                # Pad at the end if necessary (match ended early)
                padding = np.zeros((self.encoder_length - len(encoder_slice), features.shape[1]), dtype=np.float32)
                encoder_slice = np.vstack([encoder_slice, padding])

            start_frame = (
                int(match_df.iloc[enc_start][self.time_column])
                if self.time_column and enc_start < len(match_df)
                else enc_start
            )

            self.samples.append(
                (
                    encoder_slice.astype(np.float32),
                    target,
                    str(match_id),
                    start_frame,
                )
            )

        if not self.samples:
            raise RuntimeError("No samples were generated; adjust window parameters or verify dataset coverage.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        features, target, match_id, start_frame = self.samples[idx]
        return {
            "features": torch.from_numpy(features),
            "target": torch.tensor(target, dtype=torch.float32),
            "match_id": match_id,
            "start_frame": start_frame,
        }


def collate_fn(batch: List[Dict[str, Tensor]]) -> WinPredictionBatch:
    features = torch.stack([item["features"] for item in batch], dim=0)
    target = torch.stack([item["target"] for item in batch], dim=0)
    match_ids = [item["match_id"] for item in batch]
    start_frames = [item["start_frame"] for item in batch]
    return WinPredictionBatch(features, target, match_ids, start_frames)


# ----------------------------------------------------------------------------------------------------------------------
# iTransformer Model
# ----------------------------------------------------------------------------------------------------------------------

class InvertedAttention(nn.Module):
    """Inverted attention: attends over features instead of time."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq_len, d_model]
        # In iTransformer, we transpose to [batch, d_model, seq_len] to attend over features
        # But actually, we keep it as [batch, seq_len, d_model] and attend over the feature dimension
        # by treating each timestep as a token and features as the sequence
        
        residual = x
        x = self.layer_norm(x)
        
        batch_size, seq_len, d_model = x.shape
        
        # Reshape for multi-head attention
        # We'll attend over the feature dimension by transposing
        # Original: [batch, seq_len, d_model] -> treat as [batch, d_model, seq_len]
        # But we need to work with the feature dimension, so we transpose
        x_t = x.transpose(1, 2)  # [batch, d_model, seq_len]
        
        # Now apply attention over the "time" dimension (which is now the last dim)
        # But we want to attend over features, so we need to think differently
        # Actually, in iTransformer, we transpose the input so features become the sequence
        # Input: [batch, seq_len, n_features] -> transpose to [batch, n_features, seq_len]
        # Then apply attention over the seq_len dimension
        
        # For now, let's implement it correctly:
        # We want to attend over features, so we need the input as [batch, n_features, seq_len]
        # But our input is [batch, seq_len, n_features]
        # So we transpose: [batch, n_features, seq_len]
        # Then apply standard attention over the last dimension (time)
        
        # Actually, let me re-read the iTransformer paper approach:
        # They transpose the input so that features become the sequence dimension
        # Then apply standard transformer attention
        
        # Reshape back to [batch, seq_len, d_model] for standard attention
        # But apply it in a way that captures feature interactions
        
        # Standard multi-head attention
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.w_o(out)
        out = self.dropout(out)
        
        return residual + out


class InvertedTransformerBlock(nn.Module):
    """Transformer block with inverted attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = InvertedAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: Tensor) -> Tensor:
        # Attention
        x = self.attention(x)
        # Feed-forward
        residual = x
        x = self.layer_norm(x)
        x = self.ff(x)
        return residual + x


class iTransformerWinPredictor(nn.Module):
    """
    iTransformer model for win prediction.
    
    Architecture (iTransformer inverts attention):
    1. Input: [batch, seq_len, n_features]
    2. Transpose: [batch, n_features, seq_len] - features become tokens
    3. Project each feature's time series: [batch, n_features, seq_len] -> [batch, n_features, d_model]
    4. Add positional encoding for features
    5. Apply transformer blocks (attention over N features, not time)
    6. Pool over features: [batch, n_features, d_model] -> [batch, d_model]
    7. Classification head
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = None,
        dropout: float = 0.1,
        pool_method: str = "mean",  # "mean", "max", "last"
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.pool_method = pool_method
        
        # Project each feature's time series to d_model
        # Input: [batch, n_features, seq_len] -> Output: [batch, n_features, d_model]
        # Use 1D convolution to process the time series
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        
        # Positional encoding for features (not time)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_features, d_model))
        
        # Transformer blocks (attention over features)
        self.blocks = nn.ModuleList([
            InvertedTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq_len, n_features]
        batch_size, seq_len, n_features = x.shape
        
        # Step 1: Transpose so features become tokens
        x = x.transpose(1, 2)  # [batch, n_features, seq_len]
        
        # Step 2: Project each feature's time series to d_model
        # Reshape for conv1d: [batch, n_features, seq_len] -> [batch * n_features, 1, seq_len]
        # Use reshape instead of view to handle non-contiguous tensors after transpose
        x = x.reshape(batch_size * n_features, 1, seq_len)
        
        # Project: [batch * n_features, 1, seq_len] -> [batch * n_features, d_model, seq_len]
        x = self.input_proj(x)  # [batch * n_features, d_model, seq_len]
        
        # Pool over time dimension: [batch * n_features, d_model, seq_len] -> [batch * n_features, d_model]
        # Use mean pooling over time
        x = x.mean(dim=2)  # [batch * n_features, d_model]
        
        # Reshape back: [batch * n_features, d_model] -> [batch, n_features, d_model]
        x = x.reshape(batch_size, n_features, self.d_model)
        
        # Step 3: Add positional encoding for features
        x = x + self.pos_encoding
        
        # Step 4: Apply transformer blocks (attention over features)
        for block in self.blocks:
            x = block(x)  # [batch, n_features, d_model]
        
        # Step 5: Pool over features
        if self.pool_method == "mean":
            x = x.mean(dim=1)  # [batch, d_model]
        elif self.pool_method == "max":
            x = x.max(dim=1)[0]  # [batch, d_model]
        elif self.pool_method == "last":
            x = x[:, -1, :]  # [batch, d_model]
        else:
            raise ValueError(f"Unknown pool_method: {self.pool_method}")
        
        # Step 6: Classify
        logits = self.classifier(x)  # [batch, 1]
        return logits.squeeze(-1)  # [batch]


# ----------------------------------------------------------------------------------------------------------------------
# Main training/evaluation functions
# ----------------------------------------------------------------------------------------------------------------------

def train(
    model: iTransformerWinPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    early_stop_patience: int = 15,
    device: str = "cuda",
) -> Dict:
    """Train the model and return training history."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "learning_rate": [],
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            features = batch.features.to(device)
            target = batch.target.to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, target)
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch.features.to(device)
                target = batch.target.to(device)
                
                logits = model(features)
                loss = criterion(logits, target)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == target).sum().item()
                val_total += target.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Test
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch.features.to(device)
                target = batch.target.to(device)
                
                logits = model(features)
                loss = criterion(logits, target)
                test_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                test_correct += (preds == target).sum().item()
                test_total += target.size(0)
        
        test_loss /= len(test_loader)
        test_acc = test_correct / test_total if test_total > 0 else 0.0
        
        scheduler.step(1 - val_acc)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        
        # Early stopping
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if best_state is not None:
                    model.load_state_dict(best_state)
                break
        
        tqdm.write(
            f"[Epoch {epoch:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history


@torch.no_grad()
def evaluate(model: iTransformerWinPredictor, loader: DataLoader, device: str = "cuda"):
    """Evaluate model and return predictions."""
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []
    
    for batch in loader:
        features = batch.features.to(device)
        target = batch.target.to(device)
        
        logits = model(features)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    return np.array(all_probs), np.array(all_preds), np.array(all_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Data/processed/featured_data.parquet")
    parser.add_argument("--encoder_length", type=int, default=15)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    df = pd.read_parquet(args.data_path)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=df[TARGET_COLUMN])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed, stratify=temp_df[TARGET_COLUMN])
    
    # Create datasets
    train_dataset = WinPredictionDataset(
        parquet_path=args.data_path,
        encoder_length=args.encoder_length,
        match_filter=set(train_df[GROUP_COLUMN].unique()),
    )
    val_dataset = WinPredictionDataset(
        parquet_path=args.data_path,
        encoder_length=args.encoder_length,
        scaler=train_dataset.scaler,
        fit_scaler=False,
        match_filter=set(val_df[GROUP_COLUMN].unique()),
    )
    test_dataset = WinPredictionDataset(
        parquet_path=args.data_path,
        encoder_length=args.encoder_length,
        scaler=train_dataset.scaler,
        fit_scaler=False,
        match_filter=set(test_df[GROUP_COLUMN].unique()),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    n_features = len(train_dataset.samples[0][0][0]) if train_dataset.samples else len(ALL_FEATURE_COLUMNS)
    model = iTransformerWinPredictor(
        n_features=n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    
    # Train
    history = train(model, train_loader, val_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)
    
    # Evaluate
    train_probs, train_preds, train_targets = evaluate(model, train_loader, device)
    val_probs, val_preds, val_targets = evaluate(model, val_loader, device)
    test_probs, test_preds, test_targets = evaluate(model, test_loader, device)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print("\nFinal Metrics:")
    for name, preds, targets in [("Train", train_preds, train_targets), ("Val", val_preds, val_targets), ("Test", test_preds, test_targets)]:
        acc = accuracy_score(targets, preds)
        prec = precision_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        print(f"{name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

