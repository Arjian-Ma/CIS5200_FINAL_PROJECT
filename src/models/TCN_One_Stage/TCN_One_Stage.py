"""
Hierarchical TCN (Temporal Convolutional Network) for binary win prediction (Y_won).

This model adapts the hierarchical architecture from MultiStep_LSTM_DynamicWeight.py
for binary classification using TCN instead of LSTM/GRU. It uses the same hierarchical structure:
- X-minute encoder processing match state features
- Hierarchical feature decomposition (team-level combat, macro-vision, player stats)
- Player group scorers → team embeddings → A/B/C projectors
- TCN layers with dilated causal convolutions to process the sequence

Key differences from LSTM/GRU:
- Uses TCN (dilated causal convolutions) instead of recurrent cells
- Processes entire sequence in parallel (more efficient)
- Uses residual connections and exponential dilation rates
- No separate decoder needed (TCN can process full sequence)

Key differences from MultiStep_LSTM_DynamicWeight:
- Total_Gold_Difference and Total_Xp_Difference are now features (included in X1)
- Target is Y_won (binary: 0/1) instead of gold difference sequence
- Output is a single win probability (aggregated from TCN output)
- Uses BCE loss for binary classification
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
# Feature specification
# ----------------------------------------------------------------------------------------------------------------------

ENCODER_LENGTH = 15
TARGET_COLUMN = "Y_won"
GROUP_COLUMN = "match_id"
TIME_COLUMN = "frame"

# X1: Team-level combat signals (damage, kills, assists) + economy (gold/XP differences)
TEAM_X1_FEATURES: List[str] = [
    "Total_Gold_Difference",  # Now a feature!
    "Total_Xp_Difference",   # Now a feature!
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

# Remove duplicates while preserving order
_seen = set()
TEAM_X2_FEATURES = [f for f in TEAM_X2_FEATURES if not (f in _seen or _seen.add(f))]

X1_DIM = len(TEAM_X1_FEATURES)
X2_DIM = len(TEAM_X2_FEATURES)

# Player features (same as MultiStep_LSTM_DynamicWeight.py)
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

PLAYER_GROUP_SLICES: List[Tuple[int, int]] = [
    (0, 6),   # Offense: Attack_Damage, Attack_Speed, Ability_Power, Armor_Pen_Percent, Magic_Pen, Magic_Pen_Percent
    (6, 10),  # Defense: Armor, Magic_Resist, Health_Percentage, Health_Regen
    (10, 11), # Sustain: Life_Steal
    (11, 13), # Resources: Power_Percent, Power_Regen
    (13, 16), # Mobility: Movement_Speed, X_Position, Y_Position
]

PLAYER_GROUP_DIMS = [end - start for start, end in PLAYER_GROUP_SLICES]

ALL_FEATURE_COLUMNS: List[str] = TEAM_X1_FEATURES + TEAM_X2_FEATURES
for player_idx in range(1, 11):
    for feature_name in PLAYER_FEATURE_ORDER:
        ALL_FEATURE_COLUMNS.append(f"Player{player_idx}_{feature_name}")


# ----------------------------------------------------------------------------------------------------------------------
# Dataset (same as LSTM/GRU version)
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class WinPredictionBatch:
    x1: Tensor
    x2: Tensor
    players: Tensor
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

        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, str, int]] = []

        self._x1_indices = [ALL_FEATURE_COLUMNS.index(feat) for feat in TEAM_X1_FEATURES]
        self._x2_indices = [ALL_FEATURE_COLUMNS.index(feat) for feat in TEAM_X2_FEATURES]
        self._player_indices = [
            [ALL_FEATURE_COLUMNS.index(f"Player{i}_{feat}") for feat in PLAYER_FEATURE_ORDER]
            for i in range(1, 11)
        ]

        self._load()

    def _load(self) -> None:
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.parquet_path}")

        df = pd.read_parquet(self.parquet_path)

        if self.match_filter is not None:
            df = df[df[self.group_column].isin(self.match_filter)]

        missing = [col for col in ALL_FEATURE_COLUMNS + [self.target_column, self.group_column] if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if self.time_column and self.time_column not in df.columns:
            self.time_column = None

        if self.fit_scaler:
            self.scaler.fit(df[ALL_FEATURE_COLUMNS].values)

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

            features = match_df[ALL_FEATURE_COLUMNS].values.astype(np.float32)
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

            x1 = encoder_slice[:, self._x1_indices]
            x2 = encoder_slice[:, self._x2_indices]

            player_list = []
            for p_idx in self._player_indices:
                player_list.append(encoder_slice[:, p_idx])
            players = np.stack(player_list, axis=1)

            start_frame = (
                int(match_df.iloc[enc_start][self.time_column])
                if self.time_column and enc_start < len(match_df)
                else enc_start
            )

            self.samples.append(
                (
                    x1.astype(np.float32),
                    x2.astype(np.float32),
                    players.astype(np.float32),
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
        x1, x2, players, target, match_id, start_frame = self.samples[idx]
        return {
            "x1": torch.from_numpy(x1),
            "x2": torch.from_numpy(x2),
            "players": torch.from_numpy(players),
            "target": torch.tensor(target, dtype=torch.float32),
            "match_id": match_id,
            "start_frame": start_frame,
        }


def collate_fn(batch: List[Dict[str, Tensor]]) -> WinPredictionBatch:
    x1 = torch.stack([item["x1"] for item in batch], dim=0)
    x2 = torch.stack([item["x2"] for item in batch], dim=0)
    players = torch.stack([item["players"] for item in batch], dim=0)
    target = torch.stack([item["target"] for item in batch], dim=0)
    match_ids = [item["match_id"] for item in batch]
    start_frames = [item["start_frame"] for item in batch]
    return WinPredictionBatch(x1, x2, players, target, match_ids, start_frames)


# ----------------------------------------------------------------------------------------------------------------------
# Hierarchical projector (same as LSTM/GRU version)
# ----------------------------------------------------------------------------------------------------------------------

def make_mlp(input_dim: int, output_dim: int, hidden_dims: Optional[List[int]] = None, dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    hidden_dims = hidden_dims or []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class HierarchicalProjector(nn.Module):
    def __init__(
        self,
        team_embed_dim: int,
        proj_dim: int,
        hidden_dim: int,
        condition_on_hidden: bool = True,
        recurrent_dropout: float = 0.0,
    ):
        super().__init__()
        self.team_embed_dim = team_embed_dim
        self.proj_dim = proj_dim
        self.condition_on_hidden = condition_on_hidden
        self.recurrent_dropout = recurrent_dropout

        self.group_mlps = nn.ModuleList(
            [
                make_mlp(dim, 1, hidden_dims=[max(8, dim * 2)], dropout=0.1)
                for dim in PLAYER_GROUP_DIMS
            ]
        )

        self.player_combiner = make_mlp(5, 16, hidden_dims=[16], dropout=0.1)
        self.player_score_head = nn.Linear(16, 1)
        self.player_embedding = make_mlp(1, team_embed_dim, hidden_dims=[team_embed_dim], dropout=0.1)
        self.team_mlp = nn.Sequential(
            nn.Linear(team_embed_dim, team_embed_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(team_embed_dim, team_embed_dim),
        )

        self.A = make_mlp(X1_DIM, proj_dim, hidden_dims=[64, proj_dim], dropout=0.1)
        self.B = make_mlp(X2_DIM, proj_dim, hidden_dims=[32, proj_dim], dropout=0.1)
        self.C = make_mlp(team_embed_dim * 2, proj_dim, hidden_dims=[proj_dim], dropout=0.1)

        self.fA = make_mlp(X1_DIM, 1, hidden_dims=[32], dropout=0.1)
        self.fB = make_mlp(X2_DIM, 1, hidden_dims=[16], dropout=0.1)
        self.fC = make_mlp(team_embed_dim * 2, 1, hidden_dims=[32], dropout=0.1)

        if self.condition_on_hidden:
            self.hidden_mod = nn.Linear(hidden_dim, 3)

    def _player_group_scores(self, players: Tensor) -> Tensor:
        batch, num_players, _ = players.shape
        group_outputs: List[Tensor] = []
        for idx, (start, end) in enumerate(PLAYER_GROUP_SLICES):
            group_slice = players[:, :, start:end]
            group_flat = group_slice.reshape(batch * num_players, -1)
            score = self.group_mlps[idx](group_flat)
            score = score.view(batch, num_players, 1)
            group_outputs.append(score)
        group_stack = torch.cat(group_outputs, dim=-1)  # [B, num_players, 5]
        combined = self.player_combiner(group_stack.reshape(batch * num_players, -1))
        player_scores = self.player_score_head(combined).view(batch, num_players)
        return player_scores

    def _player_embeddings(self, player_scores: Tensor) -> Tensor:
        batch, num_players = player_scores.shape
        score_flat = player_scores.reshape(batch * num_players, 1)
        embed_flat = self.player_embedding(score_flat)
        return embed_flat.view(batch, num_players, self.team_embed_dim)

    def _team_pool(self, embeddings: Tensor) -> Tensor:
        batch, players, dim = embeddings.shape
        transformed = self.team_mlp(embeddings.reshape(batch * players, dim))
        transformed = transformed.reshape(batch, players, dim)
        return transformed.mean(dim=1)

    def compute(self, x1: Tensor, x2: Tensor, players: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        player_scores = self._player_group_scores(players)
        embeddings = self._player_embeddings(player_scores)

        team1 = self._team_pool(embeddings[:, :5, :])
        team2 = self._team_pool(embeddings[:, 5:, :])
        team_concat = torch.cat([team1, team2], dim=-1)

        P = self.A(x1) + self.B(x2) + self.C(team_concat)

        context = {
            "x1": x1,
            "x2": x2,
            "team_concat": team_concat,
            "team1": team1,
            "team2": team2,
        }
        return P, context

    def predict(self, context: Dict[str, Tensor], hidden: Optional[Tensor] = None) -> Tensor:
        a = self.fA(context["x1"]).squeeze(-1)
        b = self.fB(context["x2"]).squeeze(-1)
        c = self.fC(context["team_concat"]).squeeze(-1)
        if self.condition_on_hidden and hidden is not None:
            mods = torch.tanh(self.hidden_mod(hidden))
            return (1 + mods[:, 0]) * a + (1 + mods[:, 1]) * b + (1 + mods[:, 2]) * c
        return a + b + c


# ----------------------------------------------------------------------------------------------------------------------
# TCN Components
# ----------------------------------------------------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    """Temporal Convolutional Block with dilated causal convolution and residual connection."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d((kernel_size - 1) * dilation)  # Causal padding removal
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d((kernel_size - 1) * dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )

        # Downsample for residual connection if needed
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding from the right side (causal padding)."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network with multiple temporal blocks."""
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch, channels, sequence_length]
        return self.network(x)


# ----------------------------------------------------------------------------------------------------------------------
# Hierarchical TCN for Win Prediction
# ----------------------------------------------------------------------------------------------------------------------

class HierarchicalWinPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        proj_dim: int = 128,
        team_embed_dim: int = 32,
        condition_on_hidden: bool = True,
        recurrent_dropout: float = 0.0,
        tcn_channels: Optional[List[int]] = None,
        tcn_kernel_size: int = 2,
        tcn_dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.team_embed_dim = team_embed_dim

        self.projector = HierarchicalProjector(
            team_embed_dim=team_embed_dim,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            condition_on_hidden=condition_on_hidden,
            recurrent_dropout=recurrent_dropout,
        )

        # TCN configuration
        if tcn_channels is None:
            # Default: 3 layers with increasing channels
            tcn_channels = [hidden_dim, hidden_dim, hidden_dim]

        # TCN processes the sequence of projections
        # Input: [batch, proj_dim, seq_len] -> Output: [batch, hidden_dim, seq_len]
        self.tcn = TemporalConvNet(
            num_inputs=proj_dim,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
        )

        # Final hidden dimension from TCN
        final_tcn_dim = tcn_channels[-1]

        # Classification head: use the last timestep output from TCN
        self.classifier = nn.Sequential(
            nn.Linear(final_tcn_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        players: Tensor,
    ) -> Tensor:
        batch_size, seq_len, _, _ = players.size()
        device = x1.device

        # Generate hierarchical projections for all timesteps
        projections = []
        for t in range(seq_len):
            P_t, _ = self.projector.compute(x1[:, t, :], x2[:, t, :], players[:, t, :, :])
            projections.append(P_t)

        # Stack projections: [batch, seq_len, proj_dim] -> [batch, proj_dim, seq_len] for TCN
        P_sequence = torch.stack(projections, dim=1)  # [batch, seq_len, proj_dim]
        P_sequence = P_sequence.transpose(1, 2)  # [batch, proj_dim, seq_len]

        # Process through TCN
        tcn_output = self.tcn(P_sequence)  # [batch, hidden_dim, seq_len]

        # Use the last timestep output
        final_hidden = tcn_output[:, :, -1]  # [batch, hidden_dim]

        # Classification head
        logits = self.classifier(final_hidden)
        return logits.squeeze(-1)

