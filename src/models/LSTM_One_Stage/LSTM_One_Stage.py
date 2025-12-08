"""
Hierarchical Seq2Seq LSTM for binary win prediction (Y_won).

This model adapts the hierarchical Seq2Seq architecture from MultiStep_LSTM_DynamicWeight.py
for binary classification. It uses the same hierarchical structure:
- 15-minute encoder processing match state features
- Hierarchical feature decomposition (team-level combat, macro-vision, player stats)
- Player group scorers → team embeddings → A/B/C projectors
- Structured LSTM cells with context-aware memory

Key differences:
- Total_Gold_Difference and Total_Xp_Difference are now features (included in X1)
- Target is Y_won (binary: 0/1) instead of gold difference sequence
- Output is a single win probability (aggregated from decoder or encoder final state)
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
# Dataset
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
# Hierarchical projector (same as MultiStep_LSTM_DynamicWeight.py)
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


# ----------------------------------------------------------------------------------------------------------------------
# Hierarchical Seq2Seq for Win Prediction
# ----------------------------------------------------------------------------------------------------------------------

class HierarchicalWinPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        proj_dim: int = 128,
        team_embed_dim: int = 32,
        condition_on_hidden: bool = True,
        recurrent_dropout: float = 0.0,
        use_decoder: bool = True,
        decoder_steps: int = 5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.team_embed_dim = team_embed_dim
        self.use_decoder = use_decoder
        self.decoder_steps = decoder_steps

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

        if self.use_decoder:
            self.decoder_cell = StructuredLSTMCell(
                proj_dim=proj_dim,
                hidden_dim=hidden_dim,
                input_dim=0,  # No external input for decoder
                recurrent_dropout=recurrent_dropout,
            )

        # Classification head: aggregate encoder/decoder states to predict win probability
        if self.use_decoder:
            # Use both encoder final state and decoder aggregated state
            head_input_dim = hidden_dim * 2
        else:
            # Use only encoder final state
            head_input_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
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

        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Encoder: process all timesteps
        for t in range(seq_len):
            P_t, _ = self.projector.compute(x1[:, t, :], x2[:, t, :], players[:, t, :, :])
            h, c = self.encoder_cell(P_t, h, c)

        encoder_final_h = h

        if self.use_decoder:
            # Decoder: unfold for decoder_steps
            decoder_states = []
            for step in range(self.decoder_steps):
                # Use the last context from encoder (we'll recompute projection)
                # For simplicity, use the last timestep's context
                P_dec, context_dec = self.projector.compute(
                    x1[:, -1, :], x2[:, -1, :], players[:, -1, :, :]
                )
                h, c = self.decoder_cell(P_dec, h, c)
                decoder_states.append(h)

            # Aggregate decoder states (mean pooling)
            decoder_agg = torch.stack(decoder_states, dim=1).mean(dim=1)
            combined = torch.cat([encoder_final_h, decoder_agg], dim=-1)
        else:
            combined = encoder_final_h

        # Classification head
        logits = self.classifier(combined)
        return logits.squeeze(-1)


# ----------------------------------------------------------------------------------------------------------------------
# Training and Evaluation
# ----------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: HierarchicalWinPredictor,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_count = 0
    correct = 0
    total_samples = 0

    all_probs = []
    all_targets = []

    for batch in loader:
        x1 = batch.x1.to(device)
        x2 = batch.x2.to(device)
        players = batch.players.to(device)
        target = batch.target.to(device)

        logits = model(x1, x2, players)
        loss = bce(logits, target)
        total_loss += loss.item() * x1.size(0)
        total_count += x1.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == target).sum().item()
        total_samples += target.size(0)

        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / total_count if total_count > 0 else float("nan")
    accuracy = correct / total_samples if total_samples > 0 else float("nan")

    # Compute additional metrics
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = float("nan")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auc": auc,
    }


def train(
    model: HierarchicalWinPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    early_stop_patience: int = 15,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    bce = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_auc": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auc": [],
        "learning_rate": [],
        "early_stopped_epoch": None,
    }

    model.to(device)
    best_val = float("inf")
    patience_counter = 0
    early_stop = False
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_total_loss = 0.0
        train_total_count = 0
        train_correct = 0
        train_total_samples = 0
        epoch_grad_norm = 0.0
        batches = 0

        train_probs = []
        train_targets = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            x1 = batch.x1.to(device)
            x2 = batch.x2.to(device)
            players = batch.players.to(device)
            target = batch.target.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x1, x2, players)
            loss = bce(logits, target)

            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_total_loss += loss.item() * x1.size(0)
            train_total_count += x1.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            train_correct += (preds == target).sum().item()
            train_total_samples += target.size(0)

            train_probs.extend(probs.detach().cpu().numpy())
            train_targets.extend(target.detach().cpu().numpy())

            epoch_grad_norm += grad_norm.item() if isinstance(grad_norm, Tensor) else float(grad_norm)
            batches += 1

            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "grad": f"{grad_norm:.2f}",
                }
            )

        train_loss = train_total_loss / train_total_count if train_total_count > 0 else float("nan")
        train_acc = train_correct / train_total_samples if train_total_samples > 0 else float("nan")

        try:
            from sklearn.metrics import roc_auc_score
            train_auc = roc_auc_score(train_targets, train_probs)
        except:
            train_auc = float("nan")

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        current_val = val_metrics["loss"]
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
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | train_auc={train_auc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"grad_norm={epoch_grad_norm / max(1, batches):.3f}"
        )

        if early_stop:
            tqdm.write(f"Early stopping triggered at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical Seq2Seq LSTM for binary win prediction.")
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
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="results/lstm_one_stage", help="Output directory.")
    parser.add_argument("--condition-on-hidden", action="store_true", default=True, help="Enable hidden conditioning.")
    parser.add_argument("--recurrent-dropout", type=float, default=0.2, help="Dropout applied to recurrent projections.")
    parser.add_argument("--early-stop-patience", type=int, default=15, help="Epoch patience for early stopping.")
    parser.add_argument("--use-decoder", action="store_true", default=True, help="Use decoder steps for prediction.")
    parser.add_argument("--decoder-steps", type=int, default=5, help="Number of decoder steps.")
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

    train_dataset = WinPredictionDataset(
        parquet_path=args.data,
        encoder_length=ENCODER_LENGTH,
        scaler=feature_scaler,
        fit_scaler=True,
        match_filter=train_match_set,
    )
    val_dataset = WinPredictionDataset(
        parquet_path=args.data,
        encoder_length=ENCODER_LENGTH,
        scaler=feature_scaler,
        fit_scaler=False,
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
        f"[1/6] Split summary → train_matches={len(train_match_set)}, "
        f"val_matches={len(val_match_set)}, "
        f"train_samples={len(train_dataset)}, val_samples={len(val_dataset)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HierarchicalWinPredictor(
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        team_embed_dim=args.team_embed_dim,
        condition_on_hidden=args.condition_on_hidden,
        recurrent_dropout=args.recurrent_dropout,
        use_decoder=args.use_decoder,
        decoder_steps=args.decoder_steps,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[2/6] Model summary → trainable_params={trainable_params:,} "
        f"(total={total_params:,}), recurrent_dropout={args.recurrent_dropout:.3f}"
    )

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "lstm_one_stage.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": train_dataset.scaler.mean_.tolist(),
            "scaler_scale": train_dataset.scaler.scale_.tolist(),
            "feature_columns": ALL_FEATURE_COLUMNS,
            "history": history,
            "config": vars(args),
        },
        model_path,
    )

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    metrics = evaluate(model, val_loader, device)
    print(
        f"[6/6] Validation performance → loss={metrics['loss']:.4f}, "
        f"accuracy={metrics['accuracy']:.4f}, auc={metrics['auc']:.4f}"
    )
    with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(f"Model saved to     : {model_path}")
    print(f"Validation metrics : {metrics}")


if __name__ == "__main__":
    main(parse_args())
