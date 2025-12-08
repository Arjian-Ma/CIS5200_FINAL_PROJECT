#!/usr/bin/env python3
"""
Temporal Transformer-based win probability classifier.

This script mirrors the LSTM win classifier pipeline while swapping the backbone
for a transformer encoder:
  - Reuses the same variable-length sequence dataset (`WinRateSequenceDataset`)
  - Handles padding via attention masks (key padding mask)
  - Trains with BCEWithLogitsLoss, tracking accuracy/AUC
  - Saves checkpoints with feature metadata and fitted scalers
  - Plots training/validation accuracy curves
"""

from __future__ import annotations

import argparse
import io
import math
import os
import pickle
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import load_data_splits  # noqa: E402
from models.lstm_win_classifier import (  # noqa: E402
    WinRateSequenceDataset,
    build_dataloaders,
    build_datasets,
    get_specified_features,
)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding shared with regression transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class TemporalTransformerClassifier(nn.Module):
    """Transformer encoder that yields a single win-probability logit."""

    def __init__(
        self,
        input_size: int,
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
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _build_padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Create boolean mask compatible with `src_key_padding_mask`.
        True entries correspond to padding positions.
        """
        batch_size = lengths.size(0)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=lengths.device)
        for i in range(batch_size):
            valid_len = int(lengths[i].item())
            if valid_len < max_len:
                mask[i, valid_len:] = True
        return mask

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequences: Tensor of shape (batch, seq_len, input_size)
            lengths: Tensor of shape (batch,)

        Returns:
            logits of shape (batch,)
        """
        batch_size, seq_len, _ = sequences.shape
        device = sequences.device

        lengths = lengths.to(device)
        key_padding_mask = self._build_padding_mask(lengths, seq_len)

        x = self.input_projection(sequences)  # (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)

        encoded = self.transformer_encoder(
            x, src_key_padding_mask=key_padding_mask
        )  # (seq_len, batch, d_model)
        encoded = encoded.permute(1, 0, 2)  # (batch, seq_len, d_model)
        encoded = self.layer_norm(encoded)

        gather_indices = (lengths - 1).clamp(min=0)
        batch_indices = torch.arange(batch_size, device=device)
        last_tokens = encoded[batch_indices, gather_indices, :]

        logits = self.head(last_tokens)
        return logits.squeeze(-1)


def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            sequences = batch["sequences"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device).squeeze()

            logits = model(sequences, lengths)
            loss = criterion(logits, labels)
            losses.append(loss.item())

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    avg_loss = np.mean(losses) if losses else 0.0
    if all_probs:
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(labels, preds)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float("nan")
    else:
        acc = 0.0
        auc = float("nan")

    return avg_loss, acc, auc


def train_model(
    train_loader,
    val_loader,
    test_loader,
    input_size: int,
    args,
    feature_cols,
    scaler,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformerClassifier(
        input_size=input_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_auc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "test_loss": [],
        "test_acc": [],
        "test_auc": [],
    }

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        train_probs_epoch = []
        train_labels_epoch = []

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=False
        ):
            sequences = batch["sequences"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device).squeeze()

            optimizer.zero_grad()
            logits = model(sequences, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                probs_batch = torch.sigmoid(logits).cpu().numpy()
                train_probs_epoch.append(probs_batch)
                train_labels_epoch.append(labels.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        if train_probs_epoch:
            train_probs = np.concatenate(train_probs_epoch)
            train_labels = np.concatenate(train_labels_epoch)
            train_preds = (train_probs >= 0.5).astype(int)
            train_acc = accuracy_score(train_labels, train_preds)
            try:
                train_auc = roc_auc_score(train_labels, train_probs)
            except ValueError:
                train_auc = float("nan")
        else:
            train_acc = 0.0
            train_auc = float("nan")

        history["train_acc"].append(train_acc)
        history["train_auc"].append(train_auc)

        val_loss, val_acc, val_auc = evaluate(model, val_loader, device)
        test_loss, test_acc, test_auc = evaluate(model, test_loader, device)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_auc"].append(test_auc)

        print(
            f"Epoch {epoch}/{args.num_epochs} "
            f"| Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f} "
            f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f} "
            f"| Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "feature_cols": feature_cols,
                "input_size": input_size,
                "args": vars(args),
            }

            buffer = io.BytesIO()
            pickle.dump(scaler, buffer)
            best_state["feature_scaler"] = buffer.getvalue()

            save_path = os.path.join(args.model_save_dir, args.best_checkpoint_name)
            torch.save(best_state, save_path)
            print(f"  ✓ Saved new best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("⏹ Early stopping triggered.")
                break

    if best_state is not None:
        final_path = os.path.join(args.model_save_dir, args.checkpoint_name)
        torch.save(best_state, final_path)
        print(f"✓ Final checkpoint saved to {final_path}")

    return model, history


def plot_history(history, save_path: Optional[str] = None):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Training curves saved to {save_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Temporal Transformer win probability classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--feature_list",
        type=str,
        default=None,
        help='Feature list: CSV path, "specified", "None", or comma-separated list',
    )
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)

    parser.add_argument("--max_sequence_length", type=int, default=40)
    parser.add_argument("--min_sequence_length", type=int, default=5)
    parser.add_argument("--use_prefix_data", action="store_true",
                        help="If set, create multiple sequences per game with random cutoffs (data augmentation). Otherwise, use one sequence per game with random cutoff.")
    parser.add_argument("--min_cutoff_ratio", type=float, default=0.5,
                        help="Minimum cutoff ratio for random cutoff (default: 0.5, i.e., 50%% of game length)")
    parser.add_argument("--max_cutoff_ratio", type=float, default=0.9,
                        help="Maximum cutoff ratio for random cutoff (default: 0.9, i.e., 90%% of game length)")
    parser.add_argument("--num_prefix_sequences", type=int, default=3,
                        help="Number of sequences to create per game in prefix mode (default: 3)")
    parser.add_argument("--train_data_path", type=str, default=None,
                        help="Path to train parquet file (if not specified, uses default data/splits/train.parquet)")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to val parquet file (if not specified, uses default data/splits/val.parquet)")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Path to test parquet file (if not specified, uses default data/splits/test.parquet)")

    parser.add_argument(
        "--model_save_dir", type=str, default="models/win_transformer_classifier"
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="temporal_transformer_win_classifier.pth",
    )
    parser.add_argument(
        "--best_checkpoint_name",
        type=str,
        default="temporal_transformer_win_classifier_best.pth",
    )
    parser.add_argument(
        "--curve_save_path",
        type=str,
        default="results/Model_Transformer/win_classifier_training_curves.png",
    )

    args = parser.parse_args()

    feature_arg = args.feature_list
    if feature_arg is not None:
        if feature_arg.lower() == "none":
            args.feature_list = None
        elif feature_arg.lower() == "specified":
            args.feature_list = "specified"
        elif "," in feature_arg and not feature_arg.endswith(".csv"):
            args.feature_list = [token.strip() for token in feature_arg.split(",")]

    return args


def main():
    args = parse_args()
    os.makedirs(args.model_save_dir, exist_ok=True)

    print("🚀 Training Temporal Transformer Win Probability Classifier")
    print("=" * 60)
    
    if args.use_prefix_data:
        print(f"📊 Dataset mode: Prefix data ({args.num_prefix_sequences} sequences per game with random cutoffs)")
        print(f"   Cutoff range: {args.min_cutoff_ratio:.1%} - {args.max_cutoff_ratio:.1%} of game length")
    else:
        print("📊 Dataset mode: Full game sequences (1 sequence per game with random cutoff)")
        print(f"   Cutoff range: {args.min_cutoff_ratio:.1%} - {args.max_cutoff_ratio:.1%} of game length")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        feature_cols,
        scaler,
    ) = build_datasets(
        feature_list=args.feature_list,
        max_sequence_length=args.max_sequence_length,
        min_sequence_length=args.min_sequence_length,
        use_prefix_data=args.use_prefix_data,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        test_data_path=args.test_data_path,
        min_cutoff_ratio=args.min_cutoff_ratio,
        max_cutoff_ratio=args.max_cutoff_ratio,
        num_prefix_sequences=args.num_prefix_sequences,
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=args.batch_size
    )

    input_size = train_dataset.sequences[0].shape[-1]
    print(
        f"Feature count: {input_size} | "
        f"Train sequences: {len(train_dataset)} | "
        f"Val sequences: {len(val_dataset)} | "
        f"Test sequences: {len(test_dataset)}"
    )

    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_size=input_size,
        args=args,
        feature_cols=feature_cols,
        scaler=scaler,
    )

    if history["train_loss"]:
        plot_history(history, save_path=args.curve_save_path)

    print("\n✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


