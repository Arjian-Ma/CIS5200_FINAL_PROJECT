"""
Seq2Seq hierarchical LSTM with non-linear A/B/C projectors and player→team attention.

This script mirrors the data loading, training, and evaluation flow of
`MultiStep_LSTM.py` while upgrading the architecture to match the specification:

    • 15-minute encoder ⇒ 10-minute decoder (T = 15, H = 10)
    • Inputs split into x₁ (damage/kill/assist/taken diffs), x₂ (vision/macro),
      and player feature groups (five subgroups per player)
    • Non-linear A/B/C projectors feeding a structured LSTM gate
    • Player group scorers → player score → neural team pooling embeddings
    • Decoder combines hidden-conditioned context updates with residual
      XP-difference deltas and path-aware losses (level, velocity, direction)
      atop the autoregressive running total

The core mathematical definitions (loss, gate equations, BPTT) follow the same
framework documented in `MultiStep_LSTM.py`, ensuring the gradients flow through
the hierarchical hierarchy:

    head → team attention → player scorers Gₖ,F,φ → structured projectors →
    LSTM recurrences (encoder & decoder)

The model operates without access to future features during decoding; it
updates the latent team/player embeddings using the decoder hidden state and
predicts residual deltas while autoregressively unfolding on past XP
difference values (teacher-forced or self-generated).
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
FORECAST_HORIZON = 10
TARGET_COLUMN = "Total_Xp_Difference"
GROUP_COLUMN = "match_id"
TIME_COLUMN = "frame"

TEAM_X1_FEATURES: List[str] = [
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

TEAM_X2_FEATURES: List[str] = [
    "Total_Jungle_Minions_Killed_Difference",
    "Total_Ward_Placed_Difference",
    "Total_Ward_Killed_Difference",
    "Time_Enemy_Spent_Controlled_Difference",
    "Elite_Monster_Killed_Difference",
    "Buildings_Taken_Difference",
    "Total_Jungle_Minions_Killed_Difference",  # duplicate removed below
]

# Remove duplicates while preserving order
_seen = set()
TEAM_X2_FEATURES = [f for f in TEAM_X2_FEATURES if not (f in _seen or _seen.add(f))]

X1_DIM = len(TEAM_X1_FEATURES)
X2_DIM = len(TEAM_X2_FEATURES)

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
    (0, 6),
    (6, 10),
    (10, 11),
    (11, 13),
    (13, 16),
]

PLAYER_GROUP_DIMS = [end - start for start, end in PLAYER_GROUP_SLICES]

ALL_FEATURE_COLUMNS: List[str] = TEAM_X1_FEATURES + TEAM_X2_FEATURES
for player_idx in range(1, 11):
    for feature_name in PLAYER_FEATURE_ORDER:
        ALL_FEATURE_COLUMNS.append(f"Player{player_idx}_{feature_name}")


# ----------------------------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class DynamicWeightBatch:
    x1: Tensor
    x2: Tensor
    players: Tensor
    decoder_input: Tensor
    target: Tensor
    target_raw: Tensor
    match_ids: List[str]
    start_frames: List[int]


class DynamicWeightDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        encoder_length: int = ENCODER_LENGTH,
        horizon: int = FORECAST_HORIZON,
        stride: int = 1,
        target_column: str = TARGET_COLUMN,
        group_column: str = GROUP_COLUMN,
        time_column: Optional[str] = TIME_COLUMN,
        scaler: Optional[StandardScaler] = None,
        target_scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
        fit_target_scaler: bool = True,
        match_filter: Optional[set] = None,
        min_timesteps: Optional[int] = None,
    ):
        self.parquet_path = Path(parquet_path)
        self.encoder_length = encoder_length
        self.horizon = horizon
        self.stride = stride
        self.target_column = target_column
        self.group_column = group_column
        self.time_column = time_column if time_column else None
        self.scaler = scaler or StandardScaler()
        self.fit_scaler = fit_scaler
        self.target_scaler = target_scaler or StandardScaler()
        self.fit_target_scaler = fit_target_scaler
        self.match_filter = set(match_filter) if match_filter is not None else None
        self.min_timesteps = min_timesteps or (encoder_length + horizon)

        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]] = []

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
        if self.fit_target_scaler:
            self.target_scaler.fit(df[[self.target_column]].values)

        for match_id, match_df in df.groupby(self.group_column):
            if self.match_filter is not None and match_id not in self.match_filter:
                continue
            match_df = match_df.copy()
            if self.time_column:
                match_df = match_df.sort_values(self.time_column)

            if len(match_df) < self.min_timesteps:
                continue

            features = match_df[ALL_FEATURE_COLUMNS].values.astype(np.float32)
            features = self.scaler.transform(features).astype(np.float32)
            targets = match_df[self.target_column].values.astype(np.float32)

            if len(match_df) < self.encoder_length + self.horizon:
                continue

            enc_start = 0
            enc_end = self.encoder_length
            dec_end = enc_end + self.horizon

            encoder_slice = features[enc_start:enc_end]
            x1 = encoder_slice[:, self._x1_indices]
            x2 = encoder_slice[:, self._x2_indices]

            player_list = []
            for p_idx in self._player_indices:
                player_list.append(encoder_slice[:, p_idx])
            players = np.stack(player_list, axis=1)

            decoder_target_raw = targets[enc_end:dec_end]
            previous_xp_raw = targets[enc_end - 1 : dec_end - 1]

            decoder_target_scaled = self.target_scaler.transform(decoder_target_raw.reshape(-1, 1)).squeeze(-1)
            previous_xp_scaled = self.target_scaler.transform(previous_xp_raw.reshape(-1, 1)).squeeze(-1)

            start_frame = (
                int(match_df.iloc[enc_start][self.time_column])
                if self.time_column
                else enc_start
            )

            self.samples.append(
                (
                    x1.astype(np.float32),
                    x2.astype(np.float32),
                    players.astype(np.float32),
                    previous_xp_scaled.astype(np.float32),
                    decoder_target_scaled.astype(np.float32),
                    decoder_target_raw.astype(np.float32),
                    str(match_id),
                    start_frame,
                )
            )

        if not self.samples:
            raise RuntimeError("No samples were generated; adjust window parameters or verify dataset coverage.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        x1, x2, players, decoder_input, target, target_raw, match_id, start_frame = self.samples[idx]
        return {
            "x1": torch.from_numpy(x1),
            "x2": torch.from_numpy(x2),
            "players": torch.from_numpy(players),
            "decoder_input": torch.from_numpy(decoder_input),
            "target": torch.from_numpy(target),
            "target_raw": torch.from_numpy(target_raw),
            "match_id": match_id,
            "start_frame": start_frame,
        }


def collate_fn(batch: List[Dict[str, Tensor]]) -> DynamicWeightBatch:
    x1 = torch.stack([item["x1"] for item in batch], dim=0)
    x2 = torch.stack([item["x2"] for item in batch], dim=0)
    players = torch.stack([item["players"] for item in batch], dim=0)
    decoder_input = torch.stack([item["decoder_input"] for item in batch], dim=0)
    target = torch.stack([item["target"] for item in batch], dim=0)
    target_raw = torch.stack([item["target_raw"] for item in batch], dim=0)
    match_ids = [item["match_id"] for item in batch]
    start_frames = [item["start_frame"] for item in batch]
    return DynamicWeightBatch(x1, x2, players, decoder_input, target, target_raw, match_ids, start_frames)


# ----------------------------------------------------------------------------------------------------------------------
# Hierarchical projector
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

        hidden_to_team_layers: List[nn.Module] = [
            nn.Linear(hidden_dim, team_embed_dim * 2),
            nn.Tanh(),
        ]
        if self.recurrent_dropout > 0:
            hidden_to_team_layers.insert(1, nn.Dropout(self.recurrent_dropout))
        self.hidden_to_team = nn.Sequential(*hidden_to_team_layers)

        hidden_to_proj_layers: List[nn.Module] = [
            nn.Linear(hidden_dim, proj_dim),
            nn.Tanh(),
        ]
        if self.recurrent_dropout > 0:
            hidden_to_proj_layers.insert(1, nn.Dropout(self.recurrent_dropout))
        self.hidden_to_proj = nn.Sequential(*hidden_to_proj_layers)

        residual_layers: List[nn.Module] = [
            nn.Linear(team_embed_dim * 2 + hidden_dim, 32),
            nn.LeakyReLU(0.2),
        ]
        if self.recurrent_dropout > 0:
            residual_layers.append(nn.Dropout(self.recurrent_dropout))
        residual_layers.append(nn.Linear(32, 1))
        self.residual_head = nn.Sequential(*residual_layers)

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

    def decode_projection(self, context: Dict[str, Tensor], hidden: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        delta_team = self.hidden_to_team(hidden)
        delta_team1, delta_team2 = torch.chunk(delta_team, 2, dim=-1)
        team1 = context["team1"] + delta_team1
        team2 = context["team2"] + delta_team2
        team_concat = torch.cat([team1, team2], dim=-1)
        base_proj = self.A(context["x1"]) + self.B(context["x2"]) + self.C(team_concat)
        proj_delta = self.hidden_to_proj(hidden)
        P = base_proj + proj_delta
        new_context = {
            **context,
            "team1": team1,
            "team2": team2,
            "team_concat": team_concat,
        }
        return P, new_context

    def predict_with_residual(self, context: Dict[str, Tensor], hidden: Tensor, prev_xp: Tensor) -> Tensor:
        delta_base = self.predict(context, hidden)
        residual = self.residual_head(torch.cat([context["team_concat"], hidden], dim=-1)).squeeze(-1)
        delta = delta_base + residual
        return prev_xp + delta


class StructuredLSTMCell(nn.Module):
    def __init__(self, proj_dim: int, hidden_dim: int, input_dim: int = 0, recurrent_dropout: float = 0.0):
        super().__init__()
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.recurrent_dropout = recurrent_dropout

        self.W_pi = nn.Linear(proj_dim, hidden_dim, bias=True)
        self.W_pf = nn.Linear(proj_dim, hidden_dim, bias=True)
        self.W_pg = nn.Linear(proj_dim, hidden_dim, bias=True)
        self.W_po = nn.Linear(proj_dim, hidden_dim, bias=True)

        self.W_hi = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_hf = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_hg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ho = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if input_dim > 0:
            self.W_xi = nn.Linear(input_dim, hidden_dim, bias=False)
            self.W_xf = nn.Linear(input_dim, hidden_dim, bias=False)
            self.W_xg = nn.Linear(input_dim, hidden_dim, bias=False)
            self.W_xo = nn.Linear(input_dim, hidden_dim, bias=False)
        else:
            self.W_xi = self.W_xf = self.W_xg = self.W_xo = None

    def forward(self, P: Tensor, hidden: Tensor, cell: Tensor, ext_input: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        gate_i = self.W_pi(P) + self.W_hi(hidden)
        gate_f = self.W_pf(P) + self.W_hf(hidden)
        gate_g = self.W_pg(P) + self.W_hg(hidden)
        gate_o = self.W_po(P) + self.W_ho(hidden)

        if self.input_dim > 0 and ext_input is not None:
            gate_i = gate_i + self.W_xi(ext_input)
            gate_f = gate_f + self.W_xf(ext_input)
            gate_g = gate_g + self.W_xg(ext_input)
            gate_o = gate_o + self.W_xo(ext_input)

        i = torch.sigmoid(gate_i)
        f = torch.sigmoid(gate_f)
        g = torch.tanh(gate_g)
        o = torch.sigmoid(gate_o)

        if self.training and self.recurrent_dropout > 0:
            dropout_mask = torch.empty_like(g).bernoulli_(1 - self.recurrent_dropout) / (1 - self.recurrent_dropout)
            g = g * dropout_mask

        c_new = f * cell + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class HierarchicalSeq2Seq(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        proj_dim: int = 128,
        team_embed_dim: int = 32,
        teacher_forcing_ratio: float = 1.0,
        condition_on_hidden: bool = True,
        horizon: int = FORECAST_HORIZON,
        recurrent_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.team_embed_dim = team_embed_dim
        self.horizon = horizon
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.recurrent_dropout = recurrent_dropout

        self.projector = HierarchicalProjector(
            team_embed_dim=team_embed_dim,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            condition_on_hidden=condition_on_hidden,
            recurrent_dropout=recurrent_dropout,
        )

        self.encoder_cell = StructuredLSTMCell(
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            input_dim=0,
            recurrent_dropout=recurrent_dropout,
        )
        self.decoder_cell = StructuredLSTMCell(
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            input_dim=1,
            recurrent_dropout=recurrent_dropout,
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        players: Tensor,
        decoder_input: Tensor,
        teacher_forcing_ratio: Optional[float] = None,
    ) -> Tensor:
        batch_size, seq_len, _, _ = players.size()
        device = x1.device

        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)

        last_context: Optional[Dict[str, Tensor]] = None

        for t in range(seq_len):
            P_t, context_t = self.projector.compute(x1[:, t, :], x2[:, t, :], players[:, t, :, :])
            h, c = self.encoder_cell(P_t, h, c)
            last_context = context_t

        if last_context is None:
            raise RuntimeError("Encoder did not process any timesteps; verify sequence length.")

        outputs: List[Tensor] = []
        tf_ratio = self.teacher_forcing_ratio if teacher_forcing_ratio is None else teacher_forcing_ratio

        current_input = decoder_input[:, 0].unsqueeze(-1)
        context_dec = last_context

        for step in range(self.horizon):
            prev_xp = current_input.squeeze(-1)
            P_dec, context_dec = self.projector.decode_projection(context_dec, h)
            h, c = self.decoder_cell(P_dec, h, c, current_input)
            pred = self.projector.predict_with_residual(context_dec, h, prev_xp)
            outputs.append(pred)

            if self.training and torch.rand(1, device=device).item() < tf_ratio and step + 1 < self.horizon:
                current_input = decoder_input[:, step + 1].unsqueeze(-1)
            else:
                current_input = pred.unsqueeze(-1)

        return torch.stack(outputs, dim=1)


def batch_velocity_metrics(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    if preds.size(1) <= 1:
        zero = torch.tensor(0.0, device=preds.device)
        return zero, zero
    pred_diff = preds[:, 1:] - preds[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    velocity_mae = torch.mean(torch.abs(pred_diff - target_diff))
    direction_pred = pred_diff >= 0
    direction_true = target_diff >= 0
    direction_acc = torch.mean((direction_pred == direction_true).float())
    return velocity_mae, direction_acc


def batch_correlation(preds: Tensor, target: Tensor) -> Tensor:
    preds_centered = preds - preds.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    numerator = torch.sum(preds_centered * target_centered, dim=1)
    denominator = torch.sqrt(
        torch.sum(preds_centered ** 2, dim=1) * torch.sum(target_centered ** 2, dim=1) + 1e-8
    )
    corr = numerator / (denominator + 1e-8)
    return corr.mean()


def soft_dtw_loss(preds: Tensor, target: Tensor, gamma: float = 0.1) -> Tensor:
    if gamma <= 0 or preds.size(1) == 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)

    B, L = preds.shape
    diff = preds.unsqueeze(2) - target.unsqueeze(1)
    D = diff * diff
    inf = 1e6
    R = torch.full((B, L + 1, L + 1), inf, device=preds.device, dtype=preds.dtype)
    R[:, 0, 0] = 0.0

    for i in range(1, L + 1):
        for j in range(1, L + 1):
            r0 = -R[:, i - 1, j - 1] / gamma
            r1 = -R[:, i - 1, j] / gamma
            r2 = -R[:, i, j - 1] / gamma
            softmin = -gamma * torch.logsumexp(torch.stack([r0, r1, r2], dim=-1), dim=-1)
            R[:, i, j] = D[:, i - 1, j - 1] + softmin

    return R[:, -1, -1].mean()


@torch.no_grad()
def evaluate(
    model: HierarchicalSeq2Seq,
    loader: DataLoader,
    device: torch.device,
    target_scaler: StandardScaler,
    teacher_forcing_ratio: float = 0.0,
    shape_loss_weight: float = 1.0,
    direction_loss_weight: float = 1.0,
    soft_dtw_weight: float = 0.0,
    gamma_softdtw: float = 0.1,
) -> Dict[str, float]:
    model.eval()
    mse_loss = nn.MSELoss()
    smooth_l1 = nn.SmoothL1Loss(reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")

    total_scaled = 0.0
    total_raw = 0.0
    total_count = 0
    sigma = float(target_scaler.scale_[0])
    mu = float(target_scaler.mean_[0])

    shape_total = 0.0
    direction_total = 0.0
    soft_dtw_total = 0.0
    vel_total = 0.0
    dir_acc_total = 0.0
    corr_total = 0.0
    sample_count = 0

    for batch in loader:
        x1 = batch.x1.to(device)
        x2 = batch.x2.to(device)
        players = batch.players.to(device)
        decoder_input = batch.decoder_input.to(device)
        target = batch.target.to(device)
        target_raw = batch.target_raw.to(device)

        preds = model(x1, x2, players, decoder_input, teacher_forcing_ratio=teacher_forcing_ratio)
        loss_scaled = mse_loss(preds, target)
        numel = target.numel()
        total_scaled += loss_scaled.item() * numel

        preds_raw = preds * sigma + mu
        loss_raw = mse_loss(preds_raw, target_raw)
        total_raw += loss_raw.item() * numel
        total_count += numel

        if preds.size(1) > 1:
            pred_diff = preds[:, 1:] - preds[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            shape_loss = smooth_l1(pred_diff, target_diff).mean()
            direction_target = (target_diff >= 0).float()
            direction_loss = bce(pred_diff, direction_target).mean()
            shape_total += shape_loss.item() * x1.size(0)
            direction_total += direction_loss.item() * x1.size(0)

            if soft_dtw_weight > 0:
                soft_dtw_val = soft_dtw_loss(
                    preds.squeeze(0), target.squeeze(0), gamma=gamma_softdtw
                )
                soft_dtw_total += soft_dtw_val.item() * x1.size(0)

            vel_mae, dir_acc = batch_velocity_metrics(preds, target)
            vel_total += vel_mae.item() * x1.size(0)
            dir_acc_total += dir_acc.item() * x1.size(0)

        corr_total += batch_correlation(preds, target).item() * x1.size(0)
        sample_count += x1.size(0)

    rmse_scaled = math.sqrt(total_scaled / total_count) if total_count else float("nan")
    rmse_raw = math.sqrt(total_raw / total_count) if total_count else float("nan")
    denom = sample_count if sample_count else float("nan")
    velocity_mae = vel_total / sample_count if sample_count else float("nan")
    direction_acc = dir_acc_total / sample_count if sample_count else float("nan")
    corr = corr_total / sample_count if sample_count else float("nan")
    shape_avg = shape_total / sample_count if sample_count else float("nan")
    direction_avg = direction_total / sample_count if sample_count else float("nan")
    soft_dtw_avg = soft_dtw_total / sample_count if sample_count and soft_dtw_weight > 0 else 0.0

    return {
        "rmse_xp_diff_scaled": rmse_scaled,
        "rmse_xp_diff_raw": rmse_raw,
        "velocity_mae": velocity_mae,
        "directional_accuracy": direction_acc,
        "pearson_correlation": corr,
        "shape_loss": shape_avg,
        "direction_loss": direction_avg,
        "soft_dtw": soft_dtw_avg,
    }


def train(
    model: HierarchicalSeq2Seq,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    target_scaler: StandardScaler,
    epochs: int = 30,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    teacher_forcing_ratio: float = 1.0,
    scheduled_sampling_min: float = 0.1,
    aux_free_run_weight: float = 0.3,
    free_run_steps: Optional[int] = None,
    tf_warmup_fraction: float = 0.2,
    horizon_weight_start: float = 1.0,
    horizon_weight_end: float = 1.5,
    early_stop_patience: int = 15,
    shape_loss_weight: float = 1.0,
    direction_loss_weight: float = 1.0,
    soft_dtw_weight: float = 0.05,
    gamma_softdtw: float = 0.1,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    mse = nn.MSELoss(reduction="none")
    smooth_l1 = nn.SmoothL1Loss(reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")

    history = {
        "rmse_xp_diff_scaled_train": [],
        "rmse_xp_diff_raw_train": [],
        "rmse_xp_diff_scaled_val": [],
        "rmse_xp_diff_raw_val": [],
        "velocity_mae_train": [],
        "velocity_mae_val": [],
        "directional_accuracy_train": [],
        "directional_accuracy_val": [],
        "pearson_correlation_train": [],
        "pearson_correlation_val": [],
        "shape_loss_train": [],
        "shape_loss_val": [],
        "direction_loss_train": [],
        "direction_loss_val": [],
        "soft_dtw_train": [],
        "soft_dtw_val": [],
        "learning_rate": [],
        "early_stopped_epoch": None,
    }
    model.to(device)
    sigma = float(target_scaler.scale_[0])

    best_val = float("inf")
    patience_counter = 0
    early_stop = False
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_total_scaled = 0.0
        train_total_count = 0
        epoch_grad_norm = 0.0
        batches = 0
        shape_running = 0.0
        direction_running = 0.0
        soft_dtw_running = 0.0
        vel_mae_running = 0.0
        dir_acc_running = 0.0
        corr_running = 0.0

        progress_fraction = (epoch - 1) / max(1, epochs - 1)
        if progress_fraction < tf_warmup_fraction:
            tf_ratio = teacher_forcing_ratio
        else:
            decay_fraction = (progress_fraction - tf_warmup_fraction) / max(1e-8, 1 - tf_warmup_fraction)
            tf_ratio = teacher_forcing_ratio - (teacher_forcing_ratio - scheduled_sampling_min) * decay_fraction
        tf_ratio = max(scheduled_sampling_min, tf_ratio)

        horizon_weights = None
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            x1 = batch.x1.to(device)
            x2 = batch.x2.to(device)
            players = batch.players.to(device)
            decoder_input = batch.decoder_input.to(device)
            target = batch.target.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(x1, x2, players, decoder_input, teacher_forcing_ratio=tf_ratio)
            if horizon_weights is None:
                horizon_weights = torch.linspace(
                    horizon_weight_start,
                    horizon_weight_end,
                    steps=target.size(1),
                    device=device,
                    dtype=preds.dtype,
                )
                horizon_weights = horizon_weights / horizon_weights.sum()
            sq_errors = mse(preds, target)
            weighted_loss = (sq_errors * horizon_weights.view(1, -1)).sum(dim=1).mean()
            loss = weighted_loss
            aux_loss_val = 0.0
            if aux_free_run_weight > 0:
                free_preds = model(x1, x2, players, decoder_input, teacher_forcing_ratio=0.0)
                if free_run_steps is not None:
                    free_preds = free_preds[:, :free_run_steps]
                    free_target = target[:, :free_run_steps]
                else:
                    free_target = target
                free_sq = mse(free_preds, free_target)
                aux_weights = horizon_weights[: free_preds.size(1)]
                aux_weights = aux_weights / aux_weights.sum()
                aux_loss = (free_sq * aux_weights.view(1, -1)).sum(dim=1).mean()
                aux_loss_val = aux_loss.item()
                loss = loss + aux_free_run_weight * aux_loss

            if target.size(1) > 1:
                pred_diff = preds[:, 1:] - preds[:, :-1]
                target_diff = target[:, 1:] - target[:, :-1]
                shape_loss = smooth_l1(pred_diff, target_diff).mean()
                direction_target = (target_diff >= 0).float()
                direction_loss = bce(pred_diff, direction_target).mean()
                loss = loss + shape_loss_weight * shape_loss + direction_loss_weight * direction_loss

                shape_running += shape_loss.item() * x1.size(0)
                direction_running += direction_loss.item() * x1.size(0)

                if soft_dtw_weight > 0:
                    soft_dtw_val = soft_dtw_loss(
                        preds.squeeze(0), target.squeeze(0), gamma=gamma_softdtw
                    )
                    loss = loss + soft_dtw_weight * soft_dtw_val
                    soft_dtw_running += soft_dtw_val.item() * x1.size(0)

                vel_mae, dir_acc = batch_velocity_metrics(preds, target)
                vel_mae_running += vel_mae.item() * x1.size(0)
                dir_acc_running += dir_acc.item() * x1.size(0)

            corr_running += batch_correlation(preds, target).item() * x1.size(0)

            loss.backward()

            grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            numel = target.numel()
            train_total_scaled += sq_errors.sum().item()
            train_total_count += numel
            epoch_grad_norm += grad_norm.item() if isinstance(grad_norm, Tensor) else float(grad_norm)
            batches += 1

            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "aux": f"{aux_loss_val:.4f}",
                    "tf": f"{tf_ratio:.2f}",
                    "grad": f"{grad_norm:.2f}",
                }
            )

        train_rmse_scaled = math.sqrt(train_total_scaled / train_total_count) if train_total_count else float("nan")
        train_rmse_raw = train_rmse_scaled * sigma if not math.isnan(train_rmse_scaled) else float("nan")
        denom = len(train_loader.dataset) if len(train_loader.dataset) else float("nan")
        mean_shape = shape_running / len(train_loader.dataset) if len(train_loader.dataset) else float("nan")
        mean_direction = direction_running / len(train_loader.dataset) if len(train_loader.dataset) else float("nan")
        mean_soft_dtw = soft_dtw_running / len(train_loader.dataset) if len(train_loader.dataset) and soft_dtw_weight > 0 else float("nan")
        mean_vel_mae = vel_mae_running / len(train_loader.dataset) if len(train_loader.dataset) else float("nan")
        mean_dir_acc = dir_acc_running / len(train_loader.dataset) if len(train_loader.dataset) else float("nan")
        mean_corr = corr_running / len(train_loader.dataset) if len(train_loader.dataset) else float("nan")

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            target_scaler,
            teacher_forcing_ratio=0.0,
            shape_loss_weight=shape_loss_weight,
            direction_loss_weight=direction_loss_weight,
            soft_dtw_weight=soft_dtw_weight,
            gamma_softdtw=gamma_softdtw,
        )
        scheduler.step(val_metrics["rmse_xp_diff_scaled"])

        train_metrics_eval = evaluate(
            model,
            train_loader,
            device,
            target_scaler,
            teacher_forcing_ratio=0.0,
            shape_loss_weight=shape_loss_weight,
            direction_loss_weight=direction_loss_weight,
            soft_dtw_weight=soft_dtw_weight,
            gamma_softdtw=gamma_softdtw,
        )

        history["rmse_xp_diff_scaled_train"].append(train_metrics_eval["rmse_xp_diff_scaled"])
        history["rmse_xp_diff_raw_train"].append(train_metrics_eval["rmse_xp_diff_raw"])
        history["rmse_xp_diff_scaled_val"].append(val_metrics["rmse_xp_diff_scaled"])
        history["rmse_xp_diff_raw_val"].append(val_metrics["rmse_xp_diff_raw"])
        history["velocity_mae_train"].append(mean_vel_mae)
        history["velocity_mae_val"].append(val_metrics["velocity_mae"])
        history["directional_accuracy_train"].append(mean_dir_acc)
        history["directional_accuracy_val"].append(val_metrics["directional_accuracy"])
        history["pearson_correlation_train"].append(mean_corr)
        history["pearson_correlation_val"].append(val_metrics["pearson_correlation"])
        history["shape_loss_train"].append(mean_shape)
        history["shape_loss_val"].append(val_metrics["shape_loss"])
        history["direction_loss_train"].append(mean_direction)
        history["direction_loss_val"].append(val_metrics["direction_loss"])
        history["soft_dtw_train"].append(mean_soft_dtw)
        history["soft_dtw_val"].append(val_metrics["soft_dtw"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        current_val = val_metrics["rmse_xp_diff_scaled"]
        if current_val + 1e-6 < best_val:
            best_val = current_val
            patience_counter = 0
            best_state = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
            }
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                early_stop = True
                history["early_stopped_epoch"] = epoch

        tqdm.write(
            f"[Epoch {epoch:02d}] "
            f"train_rmse_raw_eval={train_metrics_eval['rmse_xp_diff_raw']:.3f} | "
            f"val_rmse_raw={val_metrics['rmse_xp_diff_raw']:.3f} | "
            f"train_rmse_scaled_eval={train_metrics_eval['rmse_xp_diff_scaled']:.3f} | "
            f"val_rmse_scaled={val_metrics['rmse_xp_diff_scaled']:.3f} | "
            f"shape_val={val_metrics['shape_loss']:.4f} | "
            f"direction_val={val_metrics['direction_loss']:.4f} | "
            f"vel_mae_val={val_metrics['velocity_mae']:.2f} | "
            f"dir_acc_val={val_metrics['directional_accuracy']:.3f} | "
            f"grad_norm={epoch_grad_norm / max(1, batches):.3f}"
        )

        if early_stop:
            tqdm.write(f"Early stopping triggered at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical Seq2Seq LSTM with dynamic A/B/C projectors.")
    parser.add_argument(
        "--data",
        type=str,
        default="Data/processed/featured_data.parquet",
        help="Path to the Parquet file containing match timelines.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden size of the structured LSTM.")
    parser.add_argument("--proj-dim", type=int, default=128, help="Dimensionality of the projector space.")
    parser.add_argument("--team-embed-dim", type=int, default=32, help="Dimension E for team embeddings.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--teacher-forcing", type=float, default=1.0, help="Initial teacher forcing ratio.")
    parser.add_argument("--scheduled-min", type=float, default=0.1, help="Minimum teacher forcing ratio.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="results/seq2seq_dynamic_xp", help="Output directory.")
    parser.add_argument("--condition-on-hidden", action="store_true", default=True, help="Enable hidden conditioning.")
    parser.add_argument("--aux-free-run-weight", type=float, default=0.3, help="Weight of auxiliary free-run loss.")
    parser.add_argument("--free-run-steps", type=int, default=None, help="Optional number of steps for auxiliary free-run loss.")
    parser.add_argument("--tf-warmup-frac", type=float, default=0.2, help="Fraction of training with full teacher forcing before decaying.")
    parser.add_argument("--horizon-weight-start", type=float, default=1.0, help="Starting weight for horizon-aware loss.")
    parser.add_argument("--horizon-weight-end", type=float, default=1.5, help="Ending weight for horizon-aware loss.")
    parser.add_argument("--recurrent-dropout", type=float, default=0.2, help="Dropout applied to recurrent projections.")
    parser.add_argument("--early-stop-patience", type=int, default=50, help="Epoch patience for early stopping.")
    parser.add_argument("--shape-loss-weight", type=float, default=1.0, help="Weight applied to shape (velocity) loss.")
    parser.add_argument("--direction-loss-weight", type=float, default=2.0, help="Weight applied to direction classification loss.")
    parser.add_argument("--soft-dtw-weight", type=float, default=0.05, help="Weight applied to soft-DTW alignment loss.")
    parser.add_argument("--gamma-softdtw", type=float, default=0.1, help="Gamma parameter for soft-DTW smoothing.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    full_df = pd.read_parquet(args.data)
    match_ids = full_df[GROUP_COLUMN].unique()
    train_ids, val_ids = train_test_split(
        match_ids,
        test_size=1 - args.train_split,
        random_state=args.seed,
        shuffle=True,
    )
    train_match_set = set(train_ids)
    val_match_set = set(val_ids)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    train_dataset = DynamicWeightDataset(
        parquet_path=args.data,
        encoder_length=ENCODER_LENGTH,
        horizon=FORECAST_HORIZON,
        stride=1,
        scaler=feature_scaler,
        target_scaler=target_scaler,
        fit_scaler=True,
        fit_target_scaler=True,
        match_filter=train_match_set,
    )
    val_dataset = DynamicWeightDataset(
        parquet_path=args.data,
        encoder_length=ENCODER_LENGTH,
        horizon=FORECAST_HORIZON,
        stride=1,
        scaler=feature_scaler,
        target_scaler=target_scaler,
        fit_scaler=False,
        fit_target_scaler=False,
        match_filter=val_match_set,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(
        f"[1/8] Split summary → train_matches={len(train_match_set)}, "
        f"val_matches={len(val_match_set)}, "
        f"train_sequences={len(train_dataset)}, val_sequences={len(val_dataset)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HierarchicalSeq2Seq(
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        team_embed_dim=args.team_embed_dim,
        teacher_forcing_ratio=args.teacher_forcing,
        condition_on_hidden=args.condition_on_hidden,
        horizon=FORECAST_HORIZON,
        recurrent_dropout=args.recurrent_dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[2/8] Model summary → trainable_params={trainable_params:,} "
        f"(total={total_params:,}), recurrent_dropout={args.recurrent_dropout:.3f}"
    )

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        target_scaler=train_dataset.target_scaler,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        teacher_forcing_ratio=args.teacher_forcing,
        scheduled_sampling_min=args.scheduled_min,
        aux_free_run_weight=args.aux_free_run_weight,
        free_run_steps=args.free_run_steps,
        tf_warmup_fraction=args.tf_warmup_frac,
        horizon_weight_start=args.horizon_weight_start,
        horizon_weight_end=args.horizon_weight_end,
        early_stop_patience=args.early_stop_patience,
        shape_loss_weight=args.shape_loss_weight,
        direction_loss_weight=args.direction_loss_weight,
        soft_dtw_weight=args.soft_dtw_weight,
        gamma_softdtw=args.gamma_softdtw,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "seq2seq_dynamic_weight_xp.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": train_dataset.scaler.mean_.tolist(),
            "scaler_scale": train_dataset.scaler.scale_.tolist(),
            "target_scaler_mean": train_dataset.target_scaler.mean_.tolist(),
            "target_scaler_scale": train_dataset.target_scaler.scale_.tolist(),
            "feature_columns": ALL_FEATURE_COLUMNS,
            "history": history,
            "config": vars(args),
        },
        model_path,
    )

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    metrics = evaluate(
        model,
        val_loader,
        device,
        train_dataset.target_scaler,
        teacher_forcing_ratio=0.0,
        shape_loss_weight=args.shape_loss_weight,
        direction_loss_weight=args.direction_loss_weight,
        soft_dtw_weight=args.soft_dtw_weight,
        gamma_softdtw=args.gamma_softdtw,
    )
    print(
        f"[8/8] Validation performance → rmse_raw={metrics['rmse_xp_diff_raw']:.3f}, "
        f"rmse_scaled={metrics['rmse_xp_diff_scaled']:.3f}, "
        f"velocity_mae={metrics['velocity_mae']:.2f}, "
        f"direction_acc={metrics['directional_accuracy']:.3f}, "
        f"corr={metrics['pearson_correlation']:.3f}"
    )
    with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(f"Model saved to     : {model_path}")
    print(f"Validation metrics : {metrics}")


if __name__ == "__main__":
    main(parse_args())


