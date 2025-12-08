#!/usr/bin/env python3
"""
Temporal Transformer model for League of Legends Gold Difference Prediction.

This script mirrors the autoregressive LSTM pipeline by:
  - Using the same data preparation utilities (`create_dataloaders`)
  - Supporting autoregressive multi-step forecasting with configurable horizon
  - Saving feature metadata and fitted scalers inside checkpoints
  - Providing a CLI for training/inference parity with lstm_model.py
"""

import argparse
import io
import math
import os
import pickle
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataloader import create_dataloaders, load_data_splits


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """Transformer encoder tailored for temporal forecasting."""

    def __init__(
        self,
        input_size: int,
        forecast_horizon: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # We will use (seq_len, batch, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_horizon),
        )

    def _generate_key_padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Create mask where True indicates padding positions.

        Args:
            lengths: Tensor of shape (batch,) containing valid lengths
            max_len: Maximum sequence length in the batch
        """
        batch_size = lengths.size(0)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=lengths.device)
        for i in range(batch_size):
            valid_len = int(lengths[i].item())
            if valid_len < max_len:
                mask[i, valid_len:] = True
        return mask

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_size)
            lengths: Tensor of shape (batch,) with valid lengths

        Returns:
            Tensor of shape (batch, forecast_horizon)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        lengths = lengths.to(device)
        padding_mask = self._generate_key_padding_mask(lengths, seq_len)  # (batch, seq_len)

        # Project inputs and apply positional encoding
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)

        encoded = self.transformer_encoder(x, src_key_padding_mask=padding_mask)  # (seq_len, batch, d_model)
        encoded = encoded.permute(1, 0, 2)  # (batch, seq_len, d_model)
        encoded = self.layer_norm(encoded)

        # Gather the last valid timestep for each sequence
        gather_indices = (lengths - 1).clamp(min=0)
        batch_indices = torch.arange(batch_size, device=device)
        last_tokens = encoded[batch_indices, gather_indices, :]  # (batch, d_model)

        return self.regressor(last_tokens)  # (batch, forecast_horizon)


def prepare_data_loaders(
    feature_list: Optional[List[str]] = None,
    batch_size: int = 32,
    sequence_length: int = 15,
    forecast_horizon: int = 5,
    autoregressive: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, List[str]]:
    """
    Wrapper matching the LSTM pipeline to create train/val/test loaders.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_splits_path = os.path.join(project_root, "data", "splits")
    train_data, val_data, test_data = load_data_splits(data_splits_path)

    # Determine feature columns (mirrors lstm_model.prepare_data_loaders)
    if isinstance(feature_list, str) and feature_list.endswith(".csv"):
        feature_list = get_specified_features(csv_path=feature_list)
    elif feature_list == "specified":
        feature_list = get_specified_features()

    y_targets = ["Total_Gold_Difference"]
    metadata_cols = ["match_id", "frame_idx", "timestamp"]
    exclude_cols = y_targets + metadata_cols

    if feature_list is not None:
        available_features = set(train_data.columns)
        verified_features = [feat for feat in feature_list if feat in available_features]
        missing = [feat for feat in feature_list if feat not in available_features]
        if missing:
            print(f"⚠ Warning: {len(missing)} features not found in train.parquet (showing up to 5): {missing[:5]}")
        feature_cols = verified_features
    else:
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        model_type="transformer",
        batch_size=batch_size,
        sequence_length=sequence_length,
        target_cols=y_targets,
        feature_cols=feature_cols,
        remove_leakage=False,  # Already handled in dataset
        forecast_horizon=forecast_horizon if autoregressive else 1,
        autoregressive=autoregressive,
    )

    sample_batch = next(iter(train_loader))
    input_size = sample_batch["sequences"].shape[-1]
    return train_loader, val_loader, test_loader, input_size, feature_cols


def get_specified_features(csv_path: Optional[str] = None) -> List[str]:
    """
    Reuse the hard-coded feature selection used in lstm_model.py.
    """
    if csv_path is not None:
        import pandas as pd

        if not os.path.exists(csv_path):
            print(f"❌ Warning: CSV file not found: {csv_path}")
        else:
            try:
                df = pd.read_csv(csv_path)
                if "feature" in df.columns:
                    col = "feature"
                elif "Feature" in df.columns:
                    col = "Feature"
                else:
                    col = df.columns[0]
                return df[col].dropna().tolist()
            except Exception as exc:
                print(f"❌ Error loading CSV ({csv_path}): {exc}")

    # Fallback: match lstm_model.py defaults
    return [
        "Total_Gold_Difference_Last_Time_Frame",
        "Total_Xp_Difference_Last_Time_Frame",
        "Total_Minions_Killed_Difference",
        "Total_Jungle_Minions_Killed_Difference",
        "Total_Kill_Difference",
        "Total_Assist_Difference",
        "Elite_Monster_Killed_Difference",
        "Buildings_Taken_Difference",
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
        "Total_Ward_Placed_Difference",
        "Total_Ward_Killed_Difference",
        "Time_Enemy_Spent_Controlled_Difference",
        "CentroidDist",
        "MinInterTeamDist",
        "EngagedDiff",
        "FrontlineOverlap",
        "RadialVelocityDiff",
        "Blue_Team_Offensive_Score",
        "Blue_Team_Defensive_Score",
        "Blue_Team_Overall_Score",
        "Red_Team_Offensive_Score",
        "Red_Team_Defensive_Score",
        "Red_Team_Overall_Score",
        "Team_Offensive_Score_Diff",
        "Team_Defensive_Score_Diff",
        "Team_Overall_Score_Diff",
    ]


def train_transformer(
    train_loader,
    val_loader,
    test_loader,
    input_size: int,
    forecast_horizon: int = 5,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    learning_rate: float = 1e-4,
    num_epochs: int = 50,
    patience: int = 5,
    model_save_dir: Optional[str] = None,
    checkpoint_name: str = "temporal_transformer.pth",
    best_checkpoint_name: str = "temporal_transformer_best.pth",
    curve_save_path: Optional[str] = None,
    feature_list: Optional[List[str]] = None,
    feature_scaler=None,
):
    """
    Train the transformer model with early stopping and checkpointing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformer(
        input_size=input_size,
        forecast_horizon=forecast_horizon,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, test_losses = [], [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    if model_save_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_save_dir = os.path.join(project_root, "models", "temporal_transformer")
    os.makedirs(model_save_dir, exist_ok=True)
    best_path = os.path.join(model_save_dir, best_checkpoint_name)
    final_path = os.path.join(model_save_dir, checkpoint_name)

    print(f"Training Temporal Transformer on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Train", leave=False):
            sequences = batch["sequences"].to(device)
            targets = batch["targets"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad()
            outputs = model(sequences, lengths)  # (batch, forecast_horizon)

            if forecast_horizon > 1:
                if targets.ndim == 1:
                    targets = targets.view(outputs.size(0), forecast_horizon)
                elif targets.shape[1] == 1:
                    targets = targets.repeat(1, forecast_horizon)

            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Val", leave=False):
                sequences = batch["sequences"].to(device)
                targets = batch["targets"].to(device)
                lengths = batch["lengths"].to(device)

                outputs = model(sequences, lengths)
                if forecast_horizon > 1:
                    if targets.ndim == 1:
                        targets = targets.view(outputs.size(0), forecast_horizon)
                    elif targets.shape[1] == 1:
                        targets = targets.repeat(1, forecast_horizon)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

        # Test evaluation (for monitoring, not used for early stopping)
        with torch.no_grad():
            running_test_loss = 0.0
            for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Test", leave=False):
                sequences = batch["sequences"].to(device)
                targets = batch["targets"].to(device)
                lengths = batch["lengths"].to(device)

                outputs = model(sequences, lengths)
                if forecast_horizon > 1:
                    if targets.ndim == 1:
                        targets = targets.view(outputs.size(0), forecast_horizon)
                    elif targets.shape[1] == 1:
                        targets = targets.repeat(1, forecast_horizon)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item()

            avg_test_loss = running_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)

        train_rmse = math.sqrt(avg_train_loss)
        val_rmse = math.sqrt(avg_val_loss)
        test_rmse = math.sqrt(avg_test_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} → Train Loss: {avg_train_loss:.6f} (RMSE {train_rmse:.4f}) | "
              f"Val Loss: {avg_val_loss:.6f} (RMSE {val_rmse:.4f}) | Test Loss: {avg_test_loss:.6f} (RMSE {test_rmse:.4f})")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
            }

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "test_losses": test_losses,
                "input_size": input_size,
                "forecast_horizon": forecast_horizon,
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
                "feature_list": feature_list,
            }
            if feature_scaler is not None:
                buffer = io.BytesIO()
                pickle.dump(feature_scaler, buffer)
                checkpoint["feature_scaler"] = buffer.getvalue()

            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved new best model to {best_path} (Val Loss {best_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{patience} epochs")

        if patience_counter >= patience:
            print("⏹ Early stopping triggered.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict["model"])
        optimizer.load_state_dict(best_state_dict["optimizer"])

    # Save final checkpoint
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
        "input_size": input_size,
        "forecast_horizon": forecast_horizon,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "feature_list": feature_list,
    }
    if feature_scaler is not None:
        buffer = io.BytesIO()
        pickle.dump(feature_scaler, buffer)
        final_checkpoint["feature_scaler"] = buffer.getvalue()

    torch.save(final_checkpoint, final_path)
    print(f"✓ Final model saved to {final_path}")

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Temporal Transformer Training Progress")
    plt.legend()
    plt.grid(True)

    if curve_save_path is None:
        curve_save_path = os.path.join(model_save_dir, "temporal_transformer_training_curves.png")
    os.makedirs(os.path.dirname(curve_save_path), exist_ok=True)
    plt.savefig(curve_save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Training curves saved to {curve_save_path}")
    plt.close()

    return model, train_losses, val_losses, test_losses


def main(args):
    print("🚀 Temporal Transformer Training")
    print("=" * 60)

    train_loader, val_loader, test_loader, input_size, feature_cols = prepare_data_loaders(
        feature_list=args.feature_list,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        autoregressive=args.autoregressive,
    )

    train_dataset = train_loader.dataset
    feature_scaler = getattr(train_dataset, "scaler", None)

    model, train_losses, val_losses, test_losses = train_transformer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_size=input_size,
        forecast_horizon=args.forecast_horizon,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        model_save_dir=args.model_save_dir,
        checkpoint_name=args.checkpoint_name,
        best_checkpoint_name=args.best_checkpoint_name,
        curve_save_path=args.curve_save_path,
        feature_list=feature_cols,
        feature_scaler=feature_scaler,
    )

    print("\n✅ Training completed!")
    print(f"Final Train RMSE: {math.sqrt(train_losses[-1]):.4f}")
    print(f"Final Val RMSE: {math.sqrt(val_losses[-1]):.4f}")
    print(f"Final Test RMSE: {math.sqrt(test_losses[-1]):.4f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Temporal Transformer for League of Legends Gold Difference Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python temporal_transformer.py --feature_list data/Feature_Selection_RFE/best_features_rfe.csv
  python temporal_transformer.py --sequence_length 20 --forecast_horizon 5 --autoregressive
""",
    )

    parser.add_argument("--feature_list", type=str, default=None,
                        help='Feature list: CSV path, "specified", "None", or comma-separated list')
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--sequence_length", type=int, default=15, help="Input sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--forecast_horizon", type=int, default=5, help="Forecast horizon (autoregressive)")
    parser.add_argument("--autoregressive", action="store_true", default=True, help="Enable autoregressive mode")
    parser.add_argument("--no_autoregressive", dest="autoregressive", action="store_false",
                        help="Disable autoregressive mode")

    parser.add_argument("--d_model", type=int, default=256, help="Transformer hidden size")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer encoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="Feedforward dimension")

    parser.add_argument("--model_save_dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="temporal_transformer.pth",
                        help="Final checkpoint name")
    parser.add_argument("--best_checkpoint_name", type=str, default="temporal_transformer_best.pth",
                        help="Best checkpoint name")
    parser.add_argument("--curve_save_path", type=str, default=None, help="Path to save training curves plot")

    args = parser.parse_args()

    feature_arg = args.feature_list
    if feature_arg is not None:
        if feature_arg.lower() == "none":
            args.feature_list = None
        elif feature_arg.lower() == "specified":
            args.feature_list = "specified"
        elif "," in feature_arg and not feature_arg.endswith(".csv"):
            args.feature_list = [token.strip() for token in feature_arg.split(",")]

    main(args)

