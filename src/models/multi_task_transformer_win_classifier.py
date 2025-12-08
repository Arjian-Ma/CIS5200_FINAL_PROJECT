#!/usr/bin/env python3
"""
Multi-task Temporal Transformer for predicting:
1. Win probability (binary classification: Blue wins = 1, Red wins = 0)
2. Elite_Monster_Killed_Difference (binary classification: Blue ahead = 1, Red ahead/tied = 0)
3. Buildings_Taken_Difference (binary classification: Blue ahead = 1, Red ahead/tied = 0)

Architecture:
- Shared Transformer encoder
- Three separate heads for each task (all binary classification)
- Multi-task loss: λ_win * L_win + λ_elite * L_elite + λ_buildings * L_buildings
- All losses use BCEWithLogitsLoss (no need for loss normalization)
"""

from __future__ import annotations

import argparse
import io
import math
import os
import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_win_classifier import (
    WinRateSequenceDataset,
    build_dataloaders,
    get_specified_features,
)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

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


class MultiTaskTemporalTransformer(nn.Module):
    """
    Multi-task transformer with shared encoder and three task-specific heads:
    All three tasks are binary classification:
    1. Win probability (Blue wins = 1, Red wins = 0)
    2. Elite_Monster_Killed_Difference (Blue ahead = 1, Red ahead/tied = 0)
    3. Buildings_Taken_Difference (Blue ahead = 1, Red ahead/tied = 0)
    
    Supports adaptive loss weighting via uncertainty-based weighting (Kendall et al., 2018).
    When use_adaptive_loss=True, learns log variance parameters for each task.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_adaptive_loss: bool = False,
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

        # Shared hidden dimension for heads
        hidden_dim = d_model // 2

        # Head 1: Win probability (binary classification)
        self.win_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Head 2: Elite_Monster_Killed_Difference (binary classification: > 0 = 1, <= 0 = 0)
        self.elite_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Head 3: Buildings_Taken_Difference (binary classification: > 0 = 1, <= 0 = 0)
        self.buildings_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # Adaptive loss weighting: learn log variance for each task
        # We learn log(σ²) for numerical stability, then σ² = exp(log_var)
        self.use_adaptive_loss = use_adaptive_loss
        if use_adaptive_loss:
            # Initialize log variance to 0 (σ² = 1, so initial weight = 1/(2*1) = 0.5)
            self.log_var_win = nn.Parameter(torch.zeros(1))
            self.log_var_elite = nn.Parameter(torch.zeros(1))
            self.log_var_buildings = nn.Parameter(torch.zeros(1))

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

    def forward(
        self, sequences: torch.Tensor, lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sequences: Tensor of shape (batch, seq_len, input_size)
            lengths: Tensor of shape (batch,)

        Returns:
            Dictionary with:
                - 'win_logit': (batch,) - logits for win probability (Blue wins)
                - 'elite_logit': (batch,) - logits for Elite_Monster_Killed_Difference (Blue ahead)
                - 'buildings_logit': (batch,) - logits for Buildings_Taken_Difference (Blue ahead)
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

        # Extract last valid timestep for each sequence
        gather_indices = (lengths - 1).clamp(min=0)
        batch_indices = torch.arange(batch_size, device=device)
        last_tokens = encoded[batch_indices, gather_indices, :]  # (batch, d_model)

        # Forward through each head (all output logits for binary classification)
        win_logit = self.win_head(last_tokens).squeeze(-1)  # (batch,)
        elite_logit = self.elite_head(last_tokens).squeeze(-1)  # (batch,)
        buildings_logit = self.buildings_head(last_tokens).squeeze(-1)  # (batch,)

        return {
            "win_logit": win_logit,
            "elite_logit": elite_logit,
            "buildings_logit": buildings_logit,
        }


class MultiTaskSequenceDataset(WinRateSequenceDataset):
    """
    Extended dataset that provides multiple targets:
    - Y_won (binary)
    - Elite_Monster_Killed_Difference (regression)
    - Buildings_Taken_Difference (regression)
    """

    def __init__(
        self,
        data,
        feature_cols: List[str],
        target_col: str = "Y_won",
        max_sequence_length: int = 30,
        min_sequence_length: int = 5,
        scaler=None,
        fit_scaler: bool = True,
        use_prefix_data: bool = False,
        min_cutoff_ratio: float = 0.5,
        max_cutoff_ratio: float = 0.9,
        num_prefix_sequences: int = 3,
    ):
        # Store additional target columns
        self.elite_target_col = "Elite_Monster_Killed_Difference"
        self.buildings_target_col = "Buildings_Taken_Difference"

        # Set feature_cols and other attributes before calling parent
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.scaler = scaler
        self.fit_scaler = fit_scaler
        self.use_prefix_data = use_prefix_data
        self.min_cutoff_ratio = min_cutoff_ratio
        self.max_cutoff_ratio = max_cutoff_ratio
        self.num_prefix_sequences = num_prefix_sequences

        # Call _prepare_sequences which we override to extract all targets
        (
            self.sequences,
            self.lengths,
            self.labels,
            self.match_ids,
            self.end_timestamps,
            self.end_frame_idxs,
            self.elite_labels,
            self.buildings_labels,
        ) = self._prepare_sequences(data)

        if self.fit_scaler or self.scaler is not None:
            self._fit_scaler_if_needed()
            self._apply_scaling()

    def _prepare_sequences(self, data):
        """
        Override parent method to extract all three targets in one pass.
        """
        sequences = []
        lengths = []
        labels = []
        match_ids = []
        end_timestamps = []
        end_frame_idxs = []
        elite_labels = []
        buildings_labels = []

        required_cols = set(
            self.feature_cols
            + [
                self.target_col,
                self.elite_target_col,
                self.buildings_target_col,
                "frame_idx",
            ]
        )
        missing = required_cols.difference(set(data.columns))
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        # Ensure labels are 0/1
        data = data.dropna(
            subset=[self.target_col, self.elite_target_col, self.buildings_target_col]
        )
        data[self.target_col] = data[self.target_col].astype(int)

        for match_id in data["match_id"].unique():
            match_df = (
                data[data["match_id"] == match_id]
                .sort_values("frame_idx")
                .reset_index(drop=True)
            )
            if match_df.empty:
                continue

            win_label = int(match_df[self.target_col].iloc[0])
            num_frames = len(match_df)

            if self.use_prefix_data:
                # Mode 2: Create multiple sequences per game with random cutoffs
                np.random.seed(hash(match_id) % (2**32))
                cutoff_ratios = np.random.uniform(
                    self.min_cutoff_ratio,
                    self.max_cutoff_ratio,
                    size=self.num_prefix_sequences,
                )
                cutoff_ratios = np.sort(cutoff_ratios)

                for ratio in cutoff_ratios:
                    end_idx = int(round(num_frames * ratio))
                    # Need at least min_sequence_length+1 frames: min_sequence_length for features + 1 for prediction
                    if end_idx < self.min_sequence_length + 1:
                        continue

                    # Use features from frames [0, end_idx-1] (historical data)
                    # Predict labels at frame end_idx-1 (current state based on past information)
                    feature_df = match_df.iloc[:end_idx-1]  # Frames 0 to end_idx-2 (historical)
                    label_frame = match_df.iloc[end_idx-1:end_idx]  # Frame end_idx-1 (prediction target)
                    
                    seq_len = len(feature_df)

                    if seq_len < self.min_sequence_length:
                        continue

                    # Include all features including outcome variables from historical frames
                    # This allows the model to learn temporal patterns (autoregressive approach)
                    feature_values = feature_df[self.feature_cols].values.astype(np.float32)
                    end_timestamps.append(int(label_frame["timestamp"].iloc[-1]))
                    end_frame_idxs.append(int(label_frame["frame_idx"].iloc[-1]))
                    sequences.append(feature_values)
                    lengths.append(seq_len)
                    labels.append(win_label)
                    match_ids.append(match_id)

                    # Extract additional targets from prediction frame (end_idx-1) and convert to binary
                    # > 0 means Blue ahead = 1, <= 0 means Red ahead or tied = 0
                    elite_val = float(label_frame[self.elite_target_col].iloc[-1])
                    buildings_val = float(label_frame[self.buildings_target_col].iloc[-1])
                    elite_labels.append(1 if elite_val > 0 else 0)
                    buildings_labels.append(1 if buildings_val > 0 else 0)
            else:
                # Mode 1: Use one sequence per game with random cutoff
                np.random.seed(hash(match_id) % (2**32))
                cutoff_ratio = np.random.uniform(
                    self.min_cutoff_ratio, self.max_cutoff_ratio
                )
                end_idx = int(round(num_frames * cutoff_ratio))

                # Need at least min_sequence_length+1 frames: min_sequence_length for features + 1 for prediction
                if end_idx < self.min_sequence_length + 1:
                    continue

                # Use features from frames [0, end_idx-1] (historical data)
                # Predict labels at frame end_idx-1 (current state based on past information)
                feature_df = match_df.iloc[:end_idx-1]  # Frames 0 to end_idx-2 (historical)
                label_frame = match_df.iloc[end_idx-1:end_idx]  # Frame end_idx-1 (prediction target)
                
                seq_len = len(feature_df)

                if seq_len < self.min_sequence_length:
                    continue

                # Include all features including outcome variables from historical frames
                # This allows the model to learn temporal patterns (autoregressive approach)
                feature_values = feature_df[self.feature_cols].values.astype(np.float32)
                end_timestamps.append(int(label_frame["timestamp"].iloc[-1]))
                end_frame_idxs.append(int(label_frame["frame_idx"].iloc[-1]))
                sequences.append(feature_values)
                lengths.append(seq_len)
                labels.append(win_label)
                match_ids.append(match_id)

                # Extract additional targets from prediction frame (end_idx-1) and convert to binary
                # > 0 means Blue ahead = 1, <= 0 means Red ahead or tied = 0
                elite_val = float(label_frame[self.elite_target_col].iloc[-1])
                buildings_val = float(label_frame[self.buildings_target_col].iloc[-1])
                elite_labels.append(1 if elite_val > 0 else 0)
                buildings_labels.append(1 if buildings_val > 0 else 0)

        return (
            sequences,
            lengths,
            labels,
            match_ids,
            end_timestamps,
            end_frame_idxs,
            elite_labels,
            buildings_labels,
        )

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        length = int(self.lengths[idx])
        win_label = torch.FloatTensor([self.labels[idx]])  # BCE expects float target
        elite_label = torch.FloatTensor([self.elite_labels[idx]])  # Binary: 0 or 1
        buildings_label = torch.FloatTensor([self.buildings_labels[idx]])  # Binary: 0 or 1
        match_id = self.match_ids[idx]

        return {
            "sequence": sequence,
            "length": torch.LongTensor([length]),
            "win_label": win_label,
            "elite_label": elite_label,
            "buildings_label": buildings_label,
            "match_id": match_id,
        }


def multi_task_collate_fn(batch):
    """Collate function for multi-task dataset."""
    sequences = [item["sequence"] for item in batch]
    lengths = [item["length"] for item in batch]
    win_labels = [item["win_label"] for item in batch]
    elite_labels = [item["elite_label"] for item in batch]
    buildings_labels = [item["buildings_label"] for item in batch]
    match_ids = [item["match_id"] for item in batch]

    padded_sequences = nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0.0
    )
    lengths = torch.cat(lengths)
    win_labels = torch.cat(win_labels)
    elite_labels = torch.cat(elite_labels)
    buildings_labels = torch.cat(buildings_labels)

    return {
        "sequences": padded_sequences,
        "lengths": lengths,
        "win_labels": win_labels,
        "elite_labels": elite_labels,
        "buildings_labels": buildings_labels,
        "match_ids": match_ids,
    }


def evaluate_multi_task(model, data_loader, device, loss_weights: Dict[str, float]):
    """Evaluate multi-task model on all tasks (all binary classification)."""
    model.eval()
    bce_criterion = nn.BCEWithLogitsLoss()

    win_losses = []
    elite_losses = []
    buildings_losses = []
    all_win_probs = []
    all_win_labels = []
    all_elite_probs = []
    all_elite_labels = []
    all_buildings_probs = []
    all_buildings_labels = []

    with torch.no_grad():
        for batch in data_loader:
            sequences = batch["sequences"].to(device)
            lengths = batch["lengths"].to(device)
            # Don't squeeze - labels are already shape (batch_size,) from collate_fn
            win_labels = batch["win_labels"].to(device)
            elite_labels = batch["elite_labels"].to(device)
            buildings_labels = batch["buildings_labels"].to(device)

            outputs = model(sequences, lengths)
            win_logit = outputs["win_logit"]  # Shape: (batch_size,)
            elite_logit = outputs["elite_logit"]  # Shape: (batch_size,)
            buildings_logit = outputs["buildings_logit"]  # Shape: (batch_size,)

            # Ensure shapes match (both should be (batch_size,))
            if win_labels.dim() > 1:
                win_labels = win_labels.squeeze(-1)
            if elite_labels.dim() > 1:
                elite_labels = elite_labels.squeeze(-1)
            if buildings_labels.dim() > 1:
                buildings_labels = buildings_labels.squeeze(-1)

            # Compute losses (all BCE since all are binary classification)
            # For evaluation, we use unweighted losses
            win_loss = bce_criterion(win_logit, win_labels)
            elite_loss = bce_criterion(elite_logit, elite_labels)
            buildings_loss = bce_criterion(buildings_logit, buildings_labels)

            win_losses.append(win_loss.item())
            elite_losses.append(elite_loss.item())
            buildings_losses.append(buildings_loss.item())

            # Collect predictions and labels
            win_probs = torch.sigmoid(win_logit).cpu().numpy()
            all_win_probs.append(win_probs)
            all_win_labels.append(win_labels.cpu().numpy())

            elite_probs = torch.sigmoid(elite_logit).cpu().numpy()
            all_elite_probs.append(elite_probs)
            all_elite_labels.append(elite_labels.cpu().numpy())

            buildings_probs = torch.sigmoid(buildings_logit).cpu().numpy()
            all_buildings_probs.append(buildings_probs)
            all_buildings_labels.append(buildings_labels.cpu().numpy())

    # Aggregate metrics
    avg_win_loss = np.mean(win_losses) if win_losses else 0.0
    avg_elite_loss = np.mean(elite_losses) if elite_losses else 0.0
    avg_buildings_loss = np.mean(buildings_losses) if buildings_losses else 0.0

    # Win task metrics
    if all_win_probs:
        win_probs = np.concatenate(all_win_probs)
        win_labels = np.concatenate(all_win_labels)
        win_preds = (win_probs >= 0.5).astype(int)
        win_acc = accuracy_score(win_labels, win_preds)
        try:
            win_auc = roc_auc_score(win_labels, win_probs)
        except ValueError:
            win_auc = float("nan")
    else:
        win_acc = 0.0
        win_auc = float("nan")

    # Elite task metrics (binary classification)
    if all_elite_probs:
        elite_probs = np.concatenate(all_elite_probs)
        elite_labels = np.concatenate(all_elite_labels)
        elite_preds = (elite_probs >= 0.5).astype(int)
        elite_acc = accuracy_score(elite_labels, elite_preds)
        try:
            elite_auc = roc_auc_score(elite_labels, elite_probs)
        except ValueError:
            elite_auc = float("nan")
    else:
        elite_acc = 0.0
        elite_auc = float("nan")

    # Buildings task metrics (binary classification)
    if all_buildings_probs:
        buildings_probs = np.concatenate(all_buildings_probs)
        buildings_labels = np.concatenate(all_buildings_labels)
        buildings_preds = (buildings_probs >= 0.5).astype(int)
        buildings_acc = accuracy_score(buildings_labels, buildings_preds)
        try:
            buildings_auc = roc_auc_score(buildings_labels, buildings_probs)
        except ValueError:
            buildings_auc = float("nan")
    else:
        buildings_acc = 0.0
        buildings_auc = float("nan")

    # Combined loss (for evaluation, use fixed weights)
    total_loss = (
        loss_weights["win"] * avg_win_loss
        + loss_weights["elite"] * avg_elite_loss
        + loss_weights["buildings"] * avg_buildings_loss
    )

    return {
        "total_loss": total_loss,
        "win_loss": avg_win_loss,
        "elite_loss": avg_elite_loss,
        "buildings_loss": avg_buildings_loss,
        "win_acc": win_acc,
        "win_auc": win_auc,
        "elite_acc": elite_acc,
        "elite_auc": elite_auc,
        "buildings_acc": buildings_acc,
        "buildings_auc": buildings_auc,
    }


def compute_adaptive_loss(loss, log_var):
    """
    Compute uncertainty-based adaptive loss.
    Loss = 1/(2*σ²) * L + log(σ)
    where σ² = exp(log_var) for numerical stability.
    """
    precision = torch.exp(-log_var)  # 1/σ²
    return precision * loss + log_var


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
    model = MultiTaskTemporalTransformer(
        input_size=input_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        use_adaptive_loss=args.use_adaptive_loss,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Loss function (all tasks are binary classification)
    bce_criterion = nn.BCEWithLogitsLoss()

    # Loss weights (used for evaluation and as initial weights if not using adaptive)
    loss_weights = {
        "win": args.loss_weight_win,
        "elite": args.loss_weight_elite,
        "buildings": args.loss_weight_buildings,
    }
    
    if args.use_adaptive_loss:
        print("🔧 Using adaptive loss weighting (uncertainty-based)")
        print(f"   Initial log variances: win={model.log_var_win.item():.3f}, "
              f"elite={model.log_var_elite.item():.3f}, "
              f"buildings={model.log_var_buildings.item():.3f}")
    else:
        print(f"📊 Using fixed loss weights: win={loss_weights['win']}, "
              f"elite={loss_weights['elite']}, buildings={loss_weights['buildings']}")

    history = {
        "train_total_loss": [],
        "train_win_loss": [],
        "train_elite_loss": [],
        "train_buildings_loss": [],
        "train_win_acc": [],
        "train_win_auc": [],
        "val_total_loss": [],
        "val_win_loss": [],
        "val_elite_loss": [],
        "val_buildings_loss": [],
        "val_win_acc": [],
        "val_win_auc": [],
        "val_elite_acc": [],
        "val_elite_auc": [],
        "val_buildings_acc": [],
        "val_buildings_auc": [],
        "test_total_loss": [],
        "test_win_loss": [],
        "test_elite_loss": [],
        "test_buildings_loss": [],
        "test_win_acc": [],
        "test_win_auc": [],
        "test_elite_acc": [],
        "test_elite_auc": [],
        "test_buildings_acc": [],
        "test_buildings_auc": [],
    }
    
    # Track adaptive weights if using adaptive loss
    if args.use_adaptive_loss:
        history["adaptive_weight_win"] = []
        history["adaptive_weight_elite"] = []
        history["adaptive_weight_buildings"] = []
        history["log_var_win"] = []
        history["log_var_elite"] = []
        history["log_var_buildings"] = []

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_win_loss = 0.0
        running_elite_loss = 0.0
        running_buildings_loss = 0.0
        train_win_probs_epoch = []
        train_win_labels_epoch = []

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=False
        ):
            sequences = batch["sequences"].to(device)
            lengths = batch["lengths"].to(device)
            # Labels are already shape (batch_size,) from collate_fn
            win_labels = batch["win_labels"].to(device)
            elite_labels = batch["elite_labels"].to(device)
            buildings_labels = batch["buildings_labels"].to(device)

            optimizer.zero_grad()
            outputs = model(sequences, lengths)

            win_logit = outputs["win_logit"]  # Shape: (batch_size,)
            elite_logit = outputs["elite_logit"]  # Shape: (batch_size,)
            buildings_logit = outputs["buildings_logit"]  # Shape: (batch_size,)

            # Ensure shapes match (both should be (batch_size,))
            if win_labels.dim() > 1:
                win_labels = win_labels.squeeze(-1)
            if elite_labels.dim() > 1:
                elite_labels = elite_labels.squeeze(-1)
            if buildings_labels.dim() > 1:
                buildings_labels = buildings_labels.squeeze(-1)

            # Compute individual losses (all BCE since all are binary classification)
            win_loss = bce_criterion(win_logit, win_labels)
            elite_loss = bce_criterion(elite_logit, elite_labels)
            buildings_loss = bce_criterion(buildings_logit, buildings_labels)

            # Combined loss (adaptive or fixed weights)
            if model.use_adaptive_loss:
                # Uncertainty-based adaptive weighting
                # Loss = 1/(2*σ²) * L + log(σ) where σ² = exp(log_var)
                win_weighted = compute_adaptive_loss(win_loss, model.log_var_win)
                elite_weighted = compute_adaptive_loss(elite_loss, model.log_var_elite)
                buildings_weighted = compute_adaptive_loss(buildings_loss, model.log_var_buildings)
                total_loss = win_weighted + elite_weighted + buildings_weighted
            else:
                # Fixed weights
                total_loss = (
                    loss_weights["win"] * win_loss
                    + loss_weights["elite"] * elite_loss
                    + loss_weights["buildings"] * buildings_loss
                )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_win_loss += win_loss.item()
            running_elite_loss += elite_loss.item()
            running_buildings_loss += buildings_loss.item()

            with torch.no_grad():
                win_probs_batch = torch.sigmoid(win_logit).cpu().numpy()
                train_win_probs_epoch.append(win_probs_batch)
                train_win_labels_epoch.append(win_labels.cpu().numpy())

        # Training metrics
        avg_train_win_loss = running_win_loss / len(train_loader)
        avg_train_elite_loss = running_elite_loss / len(train_loader)
        avg_train_buildings_loss = running_buildings_loss / len(train_loader)
        
        # Compute total loss for logging (use fixed weights for consistency)
        avg_train_total_loss = (
            loss_weights["win"] * avg_train_win_loss
            + loss_weights["elite"] * avg_train_elite_loss
            + loss_weights["buildings"] * avg_train_buildings_loss
        )
        
        # Log adaptive weights if enabled
        win_weight_adaptive = None
        elite_weight_adaptive = None
        buildings_weight_adaptive = None
        if model.use_adaptive_loss:
            # Convert log variance to actual weights for display
            # weight = 1/(2*exp(log_var)) = 0.5 * exp(-log_var)
            win_weight_adaptive = 0.5 * torch.exp(-model.log_var_win).item()
            elite_weight_adaptive = 0.5 * torch.exp(-model.log_var_elite).item()
            buildings_weight_adaptive = 0.5 * torch.exp(-model.log_var_buildings).item()
            
            # Store in history for plotting
            history["adaptive_weight_win"].append(win_weight_adaptive)
            history["adaptive_weight_elite"].append(elite_weight_adaptive)
            history["adaptive_weight_buildings"].append(buildings_weight_adaptive)
            history["log_var_win"].append(model.log_var_win.item())
            history["log_var_elite"].append(model.log_var_elite.item())
            history["log_var_buildings"].append(model.log_var_buildings.item())

        history["train_total_loss"].append(avg_train_total_loss)
        history["train_win_loss"].append(avg_train_win_loss)
        history["train_elite_loss"].append(avg_train_elite_loss)
        history["train_buildings_loss"].append(avg_train_buildings_loss)

        if train_win_probs_epoch:
            train_win_probs = np.concatenate(train_win_probs_epoch)
            train_win_labels = np.concatenate(train_win_labels_epoch)
            train_win_preds = (train_win_probs >= 0.5).astype(int)
            train_win_acc = accuracy_score(train_win_labels, train_win_preds)
            try:
                train_win_auc = roc_auc_score(train_win_labels, train_win_probs)
            except ValueError:
                train_win_auc = float("nan")
        else:
            train_win_acc = 0.0
            train_win_auc = float("nan")

        history["train_win_acc"].append(train_win_acc)
        history["train_win_auc"].append(train_win_auc)

        # Validation and test evaluation
        val_metrics = evaluate_multi_task(model, val_loader, device, loss_weights)
        test_metrics = evaluate_multi_task(model, test_loader, device, loss_weights)

        history["val_total_loss"].append(val_metrics["total_loss"])
        history["val_win_loss"].append(val_metrics["win_loss"])
        history["val_elite_loss"].append(val_metrics["elite_loss"])
        history["val_buildings_loss"].append(val_metrics["buildings_loss"])
        history["val_win_acc"].append(val_metrics["win_acc"])
        history["val_win_auc"].append(val_metrics["win_auc"])
        history["val_elite_acc"].append(val_metrics["elite_acc"])
        history["val_elite_auc"].append(val_metrics["elite_auc"])
        history["val_buildings_acc"].append(val_metrics["buildings_acc"])
        history["val_buildings_auc"].append(val_metrics["buildings_auc"])

        history["test_total_loss"].append(test_metrics["total_loss"])
        history["test_win_loss"].append(test_metrics["win_loss"])
        history["test_elite_loss"].append(test_metrics["elite_loss"])
        history["test_buildings_loss"].append(test_metrics["buildings_loss"])
        history["test_win_acc"].append(test_metrics["win_acc"])
        history["test_win_auc"].append(test_metrics["win_auc"])
        history["test_elite_acc"].append(test_metrics["elite_acc"])
        history["test_elite_auc"].append(test_metrics["elite_auc"])
        history["test_buildings_acc"].append(test_metrics["buildings_acc"])
        history["test_buildings_auc"].append(test_metrics["buildings_auc"])

        if model.use_adaptive_loss:
            print(
                f"Epoch {epoch}/{args.num_epochs} | "
                f"Train Total: {avg_train_total_loss:.4f} "
                f"(Win: {avg_train_win_loss:.4f}, Elite: {avg_train_elite_loss:.4f}, Buildings: {avg_train_buildings_loss:.4f}) | "
                f"Adaptive Weights: Win={win_weight_adaptive:.3f}, Elite={elite_weight_adaptive:.3f}, Buildings={buildings_weight_adaptive:.3f} | "
                f"Train Win Acc: {train_win_acc:.4f}, AUC: {train_win_auc:.4f} | "
                f"Val Total: {val_metrics['total_loss']:.4f} "
                f"(Win: {val_metrics['win_loss']:.4f}, Elite: {val_metrics['elite_loss']:.4f}, Buildings: {val_metrics['buildings_loss']:.4f}) | "
                f"Val Win Acc: {val_metrics['win_acc']:.4f}, AUC: {val_metrics['win_auc']:.4f} | "
                f"Val Elite Acc: {val_metrics['elite_acc']:.4f}, AUC: {val_metrics['elite_auc']:.4f} | "
                f"Val Buildings Acc: {val_metrics['buildings_acc']:.4f}, AUC: {val_metrics['buildings_auc']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch}/{args.num_epochs} | "
                f"Train Total: {avg_train_total_loss:.4f} "
                f"(Win: {avg_train_win_loss:.4f}, Elite: {avg_train_elite_loss:.4f}, Buildings: {avg_train_buildings_loss:.4f}) | "
                f"Train Win Acc: {train_win_acc:.4f}, AUC: {train_win_auc:.4f} | "
                f"Val Total: {val_metrics['total_loss']:.4f} "
                f"(Win: {val_metrics['win_loss']:.4f}, Elite: {val_metrics['elite_loss']:.4f}, Buildings: {val_metrics['buildings_loss']:.4f}) | "
                f"Val Win Acc: {val_metrics['win_acc']:.4f}, AUC: {val_metrics['win_auc']:.4f} | "
                f"Val Elite Acc: {val_metrics['elite_acc']:.4f}, AUC: {val_metrics['elite_auc']:.4f} | "
                f"Val Buildings Acc: {val_metrics['buildings_acc']:.4f}, AUC: {val_metrics['buildings_auc']:.4f}"
            )

        # Early stopping based on validation total loss
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            patience_counter = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_total_loss": val_metrics["total_loss"],
                "val_win_acc": val_metrics["win_acc"],
                "val_win_auc": val_metrics["win_auc"],
                "val_elite_acc": val_metrics["elite_acc"],
                "val_elite_auc": val_metrics["elite_auc"],
                "val_buildings_acc": val_metrics["buildings_acc"],
                "val_buildings_auc": val_metrics["buildings_auc"],
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


def plot_adaptive_weights(history, save_path: Optional[str] = None):
    """Plot how adaptive weights evolve during training."""
    if "adaptive_weight_win" not in history or not history["adaptive_weight_win"]:
        print("⚠ No adaptive weight history to plot")
        return
    
    epochs = range(1, len(history["adaptive_weight_win"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Adaptive weights over time
    axes[0, 0].plot(epochs, history["adaptive_weight_win"], label="Win Weight", linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(epochs, history["adaptive_weight_elite"], label="Elite Weight", linewidth=2, marker='s', markersize=4)
    axes[0, 0].plot(epochs, history["adaptive_weight_buildings"], label="Buildings Weight", linewidth=2, marker='^', markersize=4)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Adaptive Weight (1/(2*σ²))")
    axes[0, 0].set_title("Adaptive Loss Weights Over Time\n(Higher weight = Lower uncertainty = Easier task)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(bottom=0)
    
    # Plot 2: Log variance over time (uncertainty)
    axes[0, 1].plot(epochs, history["log_var_win"], label="Win log(σ²)", linewidth=2, marker='o', markersize=4)
    axes[0, 1].plot(epochs, history["log_var_elite"], label="Elite log(σ²)", linewidth=2, marker='s', markersize=4)
    axes[0, 1].plot(epochs, history["log_var_buildings"], label="Buildings log(σ²)", linewidth=2, marker='^', markersize=4)
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label="Initial value (σ²=1)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("log(σ²)")
    axes[0, 1].set_title("Learned Uncertainty (log variance)\n(Higher = More uncertain = Harder task)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Weight ratios (relative importance)
    win_weights = np.array(history["adaptive_weight_win"])
    elite_weights = np.array(history["adaptive_weight_elite"])
    buildings_weights = np.array(history["adaptive_weight_buildings"])
    total_weights = win_weights + elite_weights + buildings_weights
    
    axes[1, 0].plot(epochs, win_weights / total_weights * 100, label="Win %", linewidth=2, marker='o', markersize=4)
    axes[1, 0].plot(epochs, elite_weights / total_weights * 100, label="Elite %", linewidth=2, marker='s', markersize=4)
    axes[1, 0].plot(epochs, buildings_weights / total_weights * 100, label="Buildings %", linewidth=2, marker='^', markersize=4)
    axes[1, 0].axhline(y=33.33, color='gray', linestyle='--', alpha=0.5, label="Equal (33.3%)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Relative Weight (%)")
    axes[1, 0].set_title("Relative Task Importance\n(Percentage of total weight)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 100)
    
    # Plot 4: Weight vs Loss correlation
    win_losses = np.array(history["train_win_loss"])
    elite_losses = np.array(history["train_elite_loss"])
    buildings_losses = np.array(history["train_buildings_loss"])
    
    # Normalize losses to [0, 1] for comparison
    all_losses = np.concatenate([win_losses, elite_losses, buildings_losses])
    min_loss, max_loss = all_losses.min(), all_losses.max()
    if max_loss > min_loss:
        win_losses_norm = (win_losses - min_loss) / (max_loss - min_loss)
        elite_losses_norm = (elite_losses - min_loss) / (max_loss - min_loss)
        buildings_losses_norm = (buildings_losses - min_loss) / (max_loss - min_loss)
    else:
        win_losses_norm = win_losses
        elite_losses_norm = elite_losses
        buildings_losses_norm = buildings_losses
    
    axes[1, 1].scatter(win_losses_norm, win_weights, label="Win", alpha=0.6, s=50)
    axes[1, 1].scatter(elite_losses_norm, elite_weights, label="Elite", alpha=0.6, s=50)
    axes[1, 1].scatter(buildings_losses_norm, buildings_weights, label="Buildings", alpha=0.6, s=50)
    axes[1, 1].set_xlabel("Normalized Task Loss (higher = harder)")
    axes[1, 1].set_ylabel("Adaptive Weight")
    axes[1, 1].set_title("Weight vs Loss Relationship\n(Should see: Higher loss → Lower weight)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Adaptive weights plot saved to {save_path}")
    plt.close()


def plot_history(history, save_path: Optional[str] = None):
    """Plot training curves for all tasks."""
    epochs = range(1, len(history["train_total_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Losses
    axes[0, 0].plot(epochs, history["train_total_loss"], label="Train Total Loss")
    axes[0, 0].plot(epochs, history["val_total_loss"], label="Val Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Combined Loss")
    axes[0, 0].set_title("Total Multi-Task Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, history["train_win_loss"], label="Train Win Loss")
    axes[0, 1].plot(epochs, history["val_win_loss"], label="Val Win Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("BCE Loss")
    axes[0, 1].set_title("Win Probability Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[0, 2].plot(epochs, history["train_elite_loss"], label="Train Elite Loss")
    axes[0, 2].plot(epochs, history["val_elite_loss"], label="Val Elite Loss")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("MSE Loss")
    axes[0, 2].set_title("Elite Monster Loss")
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Row 2: Metrics
    axes[1, 0].plot(epochs, history["train_win_acc"], label="Train Win Acc")
    axes[1, 0].plot(epochs, history["val_win_acc"], label="Val Win Acc")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Win Probability Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].plot(epochs, history["val_elite_acc"], label="Val Elite Acc")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title("Elite Monster Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim(0, 1)

    axes[1, 2].plot(epochs, history["val_buildings_acc"], label="Val Buildings Acc")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Accuracy")
    axes[1, 2].set_title("Buildings Accuracy")
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Training curves saved to {save_path}")
    plt.close()


def build_multi_task_datasets(
    feature_list,
    max_sequence_length: int,
    min_sequence_length: int,
    use_prefix_data: bool = False,
    train_data_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
    min_cutoff_ratio: float = 0.5,
    max_cutoff_ratio: float = 0.9,
    num_prefix_sequences: int = 3,
):
    """
    Build multi-task datasets for training.
    """
    from models.lstm_win_classifier import build_datasets as build_base_datasets

    # Load data (load files directly, not using load_data_splits which expects a directory)
    if train_data_path is None:
        train_data_path = "data/splits/train.parquet"
    if val_data_path is None:
        val_data_path = "data/splits/val.parquet"
    if test_data_path is None:
        test_data_path = "data/splits/test.parquet"

    train_df = pd.read_parquet(train_data_path)
    val_df = pd.read_parquet(val_data_path)
    test_df = pd.read_parquet(test_data_path)

    # Resolve feature list
    # NOTE: We include Elite_Monster_Killed_Difference and Buildings_Taken_Difference
    # as autoregressive features (historical values from frames 0 to t-1)
    if feature_list is None:
        feature_cols = [
            col
            for col in train_df.columns
            if col
            not in [
                "match_id",
                "frame_idx",
                "timestamp",
                "Y_won",
                "puuid",
                "team",
                # NOTE: Elite_Monster_Killed_Difference and Buildings_Taken_Difference
                # are INCLUDED as features (autoregressive approach)
            ]
        ]
    elif isinstance(feature_list, str):
        if feature_list.lower() == "specified":
            feature_cols = get_specified_features()
        elif feature_list.lower() == "none":
            feature_cols = [
                col
                for col in train_df.columns
                if col
                not in [
                    "match_id",
                    "frame_idx",
                    "timestamp",
                    "Y_won",
                    "puuid",
                    "team",
                    # NOTE: Elite_Monster_Killed_Difference and Buildings_Taken_Difference
                    # are INCLUDED as features (autoregressive approach)
                ]
            ]
        elif feature_list.endswith(".csv"):
            feature_cols = get_specified_features(feature_list)
        else:
            feature_cols = [f.strip() for f in feature_list.split(",") if f.strip()]
    else:
        feature_cols = feature_list

    # Filter out missing features
    # NOTE: We will include Elite_Monster_Killed_Difference and Buildings_Taken_Difference
    # as features from PREVIOUS timestamps (autoregressive approach)
    excluded_cols = {
        "match_id", "frame_idx", "timestamp", "Y_won", "puuid", "team"
        # NOTE: Elite_Monster_Killed_Difference and Buildings_Taken_Difference are NOT excluded
        # They will be included as features from historical frames (0 to t-1)
        # but excluded from the prediction frame (t) to avoid leakage
    }
    
    # Remove metadata columns but keep outcome variables (they'll be used as autoregressive features)
    feature_cols = [f for f in feature_cols if f not in excluded_cols]
    
    available_features = [f for f in feature_cols if f in train_df.columns]
    missing_features = [f for f in feature_cols if f not in train_df.columns]
    if missing_features:
        print(f"⚠ Warning: {len(missing_features)} features missing from data: {missing_features[:5]}...")
    feature_cols = available_features
    
    # Double-check no data leakage (metadata columns only)
    leakage_check = [f for f in feature_cols if f in excluded_cols]
    if leakage_check:
        raise ValueError(f"❌ Data leakage detected! Features contain metadata columns: {leakage_check}")
    
    # Check if outcome variables are in features (they should be for autoregressive approach)
    outcome_vars_in_features = [
        f for f in feature_cols 
        if f in ["Elite_Monster_Killed_Difference", "Buildings_Taken_Difference"]
    ]
    if outcome_vars_in_features:
        print(f"✅ Using autoregressive features: {outcome_vars_in_features} (historical values from frames 0 to t-1)")
    
    print(f"✅ Using {len(feature_cols)} features (excluded metadata: {excluded_cols})")

    # Ensure target columns exist
    required_targets = ["Y_won", "Elite_Monster_Killed_Difference", "Buildings_Taken_Difference"]
    for target in required_targets:
        if target not in train_df.columns:
            raise ValueError(f"Required target column '{target}' not found in dataset")

    # Build datasets
    train_dataset = MultiTaskSequenceDataset(
        train_df,
        feature_cols=feature_cols,
        target_col="Y_won",
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        scaler=None,
        fit_scaler=True,
        use_prefix_data=use_prefix_data,
        min_cutoff_ratio=min_cutoff_ratio,
        max_cutoff_ratio=max_cutoff_ratio,
        num_prefix_sequences=num_prefix_sequences,
    )

    val_dataset = MultiTaskSequenceDataset(
        val_df,
        feature_cols=feature_cols,
        target_col="Y_won",
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        scaler=train_dataset.scaler,
        fit_scaler=False,
        use_prefix_data=use_prefix_data,
        min_cutoff_ratio=min_cutoff_ratio,
        max_cutoff_ratio=max_cutoff_ratio,
        num_prefix_sequences=num_prefix_sequences,
    )

    test_dataset = MultiTaskSequenceDataset(
        test_df,
        feature_cols=feature_cols,
        target_col="Y_won",
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        scaler=train_dataset.scaler,
        fit_scaler=False,
        use_prefix_data=use_prefix_data,
        min_cutoff_ratio=min_cutoff_ratio,
        max_cutoff_ratio=max_cutoff_ratio,
        num_prefix_sequences=num_prefix_sequences,
    )

    return train_dataset, val_dataset, test_dataset, feature_cols, train_dataset.scaler


def build_multi_task_dataloaders(
    train_dataset, val_dataset, test_dataset, batch_size: int, num_workers: int = 4
):
    """Build dataloaders with multi-task collate function."""
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=multi_task_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=multi_task_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=multi_task_collate_fn,
    )
    return train_loader, val_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Multi-Task Temporal Transformer",
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
    parser.add_argument(
        "--use_prefix_data",
        action="store_true",
        help="If set, create multiple sequences per game with random cutoffs (data augmentation). Otherwise, use one sequence per game with random cutoff.",
    )
    parser.add_argument(
        "--min_cutoff_ratio",
        type=float,
        default=0.5,
        help="Minimum cutoff ratio for random cutoff (default: 0.5, i.e., 50%% of game length)",
    )
    parser.add_argument(
        "--max_cutoff_ratio",
        type=float,
        default=0.9,
        help="Maximum cutoff ratio for random cutoff (default: 0.9, i.e., 90%% of game length)",
    )
    parser.add_argument(
        "--num_prefix_sequences",
        type=int,
        default=3,
        help="Number of sequences to create per game in prefix mode (default: 3)",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=None,
        help="Path to train parquet file (if not specified, uses default data/splits/train.parquet)",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=None,
        help="Path to val parquet file (if not specified, uses default data/splits/val.parquet)",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
        help="Path to test parquet file (if not specified, uses default data/splits/test.parquet)",
    )

    # Loss weights
    parser.add_argument(
        "--loss_weight_win",
        type=float,
        default=1.0,
        help="Weight for win probability loss (default: 1.0)",
    )
    parser.add_argument(
        "--loss_weight_elite",
        type=float,
        default=0.3,
        help="Weight for Elite_Monster_Killed_Difference loss (default: 0.3)",
    )
    parser.add_argument(
        "--loss_weight_buildings",
        type=float,
        default=0.3,
        help="Weight for Buildings_Taken_Difference loss (default: 0.3)",
    )
    parser.add_argument(
        "--use_adaptive_loss",
        action="store_true",
        help="Enable adaptive loss weighting using uncertainty-based method (Kendall et al., 2018). "
             "When enabled, learns log variance parameters for each task during training. "
             "If disabled, uses fixed loss weights specified by --loss_weight_* arguments.",
    )

    parser.add_argument(
        "--model_save_dir", type=str, default="models/multi_task_transformer_classifier"
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="multi_task_transformer_win_classifier.pth",
    )
    parser.add_argument(
        "--best_checkpoint_name",
        type=str,
        default="multi_task_transformer_win_classifier_best.pth",
    )
    parser.add_argument(
        "--curve_save_path",
        type=str,
        default="results/Model_MultiTask_Transformer/win_classifier_training_curves.png",
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

    print("🚀 Training Multi-Task Temporal Transformer")
    print("=" * 60)
    print("Tasks (all binary classification):")
    print("  1. Win Probability (Blue wins = 1, Red wins = 0)")
    print("  2. Elite_Monster_Killed_Difference (Blue ahead = 1, Red ahead/tied = 0)")
    print("  3. Buildings_Taken_Difference (Blue ahead = 1, Red ahead/tied = 0)")
    print("=" * 60)

    if args.use_prefix_data:
        print(
            f"📊 Dataset mode: Prefix data ({args.num_prefix_sequences} sequences per game with random cutoffs)"
        )
        print(
            f"   Cutoff range: {args.min_cutoff_ratio:.1%} - {args.max_cutoff_ratio:.1%} of game length"
        )
    else:
        print(
            "📊 Dataset mode: Full game sequences (1 sequence per game with random cutoff)"
        )
        print(
            f"   Cutoff range: {args.min_cutoff_ratio:.1%} - {args.max_cutoff_ratio:.1%} of game length"
        )

    print(f"\nLoss weights:")
    print(f"  Win: {args.loss_weight_win}")
    print(f"  Elite: {args.loss_weight_elite}")
    print(f"  Buildings: {args.loss_weight_buildings}")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        feature_cols,
        scaler,
    ) = build_multi_task_datasets(
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

    train_loader, val_loader, test_loader = build_multi_task_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=args.batch_size
    )

    input_size = train_dataset.sequences[0].shape[-1]
    print(
        f"\nFeature count: {input_size} | "
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

    if history["train_total_loss"]:
        plot_history(history, save_path=args.curve_save_path)
        
        # Plot adaptive weights if using adaptive loss
        if args.use_adaptive_loss and "adaptive_weight_win" in history and history["adaptive_weight_win"]:
            # Save adaptive weights plot in same directory as training curves
            adaptive_weights_path = args.curve_save_path.replace("training_curves.png", "adaptive_weights.png")
            plot_adaptive_weights(history, save_path=adaptive_weights_path)

    print("\n✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

