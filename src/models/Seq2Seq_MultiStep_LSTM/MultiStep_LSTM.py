"""
Seq2Seq multi-step LSTM forecaster for League of Legends gold difference.

This module implements a 15-minute encoder → 15-minute decoder architecture using
teacher forcing during training and autoregressive rollout at inference time.
It also documents the exact loss and back-propagation (BPTT) equations that are
used internally by PyTorch when gradients are computed.

Notation (per the project specifications):

    • Past window length  T = 15
    • Forecast horizon    H = 15
    • Encoder inputs      x_t ∈ ℝ¹⁸⁶   (all contextual features)
    • Past gold history   y_t ∈ ℝ      (Total_Gold_Difference at time t)
    • Concatenated input  z_t = [x_t ; y_t] ∈ ℝ¹⁸⁷
    • Decoder predicts    ŷ_{T+1:T+H}
    • Decoder input       u_k = y_{T+k-1} (teacher forcing) or ŷ_{T+k-1} (free running)

Loss (mean squared error over the H-step horizon):

    L = (1 / H) * Σ_{k=1..H} ( ŷ_{T+k} – y_{T+k} )²

Gradients at the linear head:

    δ_k^{(y)}          = ∂L/∂ŷ_{T+k} = (2 / H) ( ŷ_{T+k} – y_{T+k} )
    ∂L/∂w_o            = Σ_k δ_k^{(y)} h_k^dec
    ∂L/∂b_o            = Σ_k δ_k^{(y)}
    ∂L/∂h_k^dec        = δ_k^{(y)} w_o

Decoder BPTT (backwards for k = H … 1):

    do_k = dh_k ⊙ tanh(c_k) ⊙ σ'(a^o_k)
    dc_k += dh_k ⊙ o_k ⊙ (1 - tanh²(c_k))
    df_k = dc_k ⊙ c_{k-1} ⊙ σ'(a^f_k)
    di_k = dc_k ⊙ g_k     ⊙ σ'(a^i_k)
    dg_k = dc_k ⊙ i_k     ⊙ (1 - tanh²(a^g_k))
    dc_{k-1} = dc_k ⊙ f_k
    dh_{k-1} = W_hiᵀ di_k + W_hfᵀ df_k + W_hoᵀ do_k + W_hgᵀ dg_k

    Parameter updates accumulate as:
        ∂L/∂W_xi += di_k x_kᵀ,   ∂L/∂W_hi += di_k h_{k-1}ᵀ,   ∂L/∂b_i += di_k,   etc.

When the decoder runs freely (u_k = ŷ_{T+k-1}) the gradient wrt decoder input
adds an autoregressive term:

    ∂L/∂ŷ_{T+k-1} += d x_k

The encoder receives gradients only via the decoder initial state:

    ∂L/∂h_T^enc = d h_0^dec,    ∂L/∂c_T^enc = d c_0^dec

Implementation notes:
    • Teacher forcing is enabled during training.
    • Scheduled sampling can be configured (default 0.0 → pure teacher forcing).
    • Gradients are clipped (global norm) every step.
    • Supports loss masking for sequences shorter than T + H (if needed later).

The model consumes features from `Data/processed/featured_data.parquet` and uses
`Total_Gold_Difference` as the primary target. Additional contextual features
are listed in FEATURE_COLUMNS below.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
# Configuration
# ----------------------------------------------------------------------------------------------------------------------

ENCODER_LENGTH = 15  # 15-minute history window
FORECAST_HORIZON = 15  # Predict next 15 minutes
TARGET_COLUMN = "Total_Gold_Difference"
GROUP_COLUMN = "match_id"
TIME_COLUMN = "frame"  # fallback ordering column; update if dataset uses another name

FEATURE_COLUMNS: List[str] = [
    "Total_Jungle_Minions_Killed_Difference",
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
    "Total_Ward_Placed_Difference",
    "Total_Ward_Killed_Difference",
    "Time_Enemy_Spent_Controlled_Difference",
    "Elite_Monster_Killed_Difference",
    "Buildings_Taken_Difference",
]

_PLAYER_STATS = [
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

for idx in range(1, 11):
    FEATURE_COLUMNS.extend([f"Player{idx}_{stat}" for stat in _PLAYER_STATS])

INPUT_DIM = len(FEATURE_COLUMNS) + 1  # +1 for past gold difference concatenation


# ----------------------------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class MultiStepBatch:
    encoder_input: Tensor  # [B, T, F+1]
    decoder_input: Tensor  # [B, H] (teacher forcing values)
    target: Tensor  # [B, H]
    match_ids: List[str]
    start_frames: List[int]


class MultiStepLSTMDataset(Dataset):
    """
    Creates fixed-length encoder/decoder sequences from match timelines.

    For each match we slide a window of length T + H and produce a single training example:
        • encoder_input  : [T, F+1]   where each row is [features, gold_diff]
        • decoder_input  : [H]        previous gold differences (teacher forcing)
        • target         : [H]        future gold differences to forecast
    """

    def __init__(
        self,
        parquet_path: str | Path,
        encoder_length: int = ENCODER_LENGTH,
        forecast_horizon: int = FORECAST_HORIZON,
        feature_columns: Optional[List[str]] = None,
        target_column: str = TARGET_COLUMN,
        group_column: str = GROUP_COLUMN,
        time_column: Optional[str] = TIME_COLUMN,
        stride: int = 1,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
        min_timesteps: Optional[int] = None,
    ):
        self.parquet_path = Path(parquet_path)
        if feature_columns is None:
            feature_columns = FEATURE_COLUMNS

        self.encoder_length = encoder_length
        self.horizon = forecast_horizon
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.group_column = group_column
        self.time_column = time_column
        self.stride = stride
        self.scaler = scaler or StandardScaler()
        self.fit_scaler = fit_scaler
        self.min_timesteps = min_timesteps or (encoder_length + forecast_horizon)

        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]] = []

        self._load_and_prepare()

    def _load_and_prepare(self) -> None:
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.parquet_path}")

        df = pd.read_parquet(self.parquet_path)
        missing = [col for col in self.feature_columns + [self.target_column, self.group_column] if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if self.time_column and self.time_column not in df.columns:
            # If the requested time column is absent, fallback to positional index within each match
            self.time_column = None

        # Fit scaler on the entire dataset (only feature columns)
        if self.fit_scaler:
            self.scaler.fit(df[self.feature_columns].values)

        for match_id, match_df in df.groupby(self.group_column):
            match_df = match_df.copy()
            if self.time_column:
                match_df = match_df.sort_values(self.time_column)

            if len(match_df) < self.min_timesteps:
                continue

            features = match_df[self.feature_columns].values.astype(np.float32)
            targets = match_df[self.target_column].values.astype(np.float32)

            features = self.scaler.transform(features).astype(np.float32)

            num_steps = len(match_df)
            start_indices = range(self.encoder_length, num_steps - self.horizon + 1, self.stride)
            for end_idx in start_indices:
                enc_start = end_idx - self.encoder_length
                enc_end = end_idx
                dec_end = end_idx + self.horizon

                encoder_features = features[enc_start:enc_end]  # [T, F]
                encoder_gold = targets[enc_start:enc_end][:, None]  # [T, 1]
                encoder_input = np.concatenate([encoder_features, encoder_gold], axis=-1)  # [T, F+1]

                decoder_target = targets[enc_end:dec_end]  # [H]

                # Teacher forcing inputs: previous gold diffs (y_{T+k-1})
                previous_gold = targets[enc_end - 1 : dec_end - 1]
                decoder_input = previous_gold  # [H]

                start_frame = int(match_df.iloc[enc_start][self.time_column]) if self.time_column else enc_start
                self.samples.append(
                    (
                        encoder_input.astype(np.float32),
                        decoder_input.astype(np.float32),
                        decoder_target.astype(np.float32),
                        str(match_id),
                        start_frame,
                    )
                )

        if not self.samples:
            raise RuntimeError("No training samples were generated; check window sizes or dataset coverage.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        encoder_input, decoder_input, target, match_id, start_frame = self.samples[idx]
        return {
            "encoder_input": torch.from_numpy(encoder_input),  # [T, F+1]
            "decoder_input": torch.from_numpy(decoder_input),  # [H]
            "target": torch.from_numpy(target),  # [H]
            "match_id": match_id,
            "start_frame": start_frame,
        }


def collate_fn(batch: List[Dict[str, Tensor]]) -> MultiStepBatch:
    encoder_input = torch.stack([item["encoder_input"] for item in batch], dim=0)
    decoder_input = torch.stack([item["decoder_input"] for item in batch], dim=0)
    target = torch.stack([item["target"] for item in batch], dim=0)
    match_ids = [item["match_id"] for item in batch]
    start_frames = [item["start_frame"] for item in batch]
    return MultiStepBatch(encoder_input, decoder_input, target, match_ids, start_frames)


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------

class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = FORECAST_HORIZON,
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.LSTM(
            input_size=1,  # decoder receives previous gold diff
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        encoder_input: Tensor,
        decoder_input: Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> Tensor:
        """
        Args:
            encoder_input: [B, T, F+1]
            decoder_input: [B, H] (ground truth gold diffs for teacher forcing)
            teacher_forcing_ratio: probability of using ground-truth decoder input per step

        Returns:
            predictions: [B, H]
        """
        batch_size = encoder_input.size(0)

        _, hidden = self.encoder(encoder_input)  # hidden = (h, c) each [num_layers, B, hidden_dim]

        outputs = []
        current_input = decoder_input[:, 0].unsqueeze(-1)  # [B, 1]

        for step in range(self.horizon):
            output, hidden = self.decoder(current_input.unsqueeze(1), hidden)  # output: [B, 1, hidden_dim]
            pred = self.output_head(output.squeeze(1)).squeeze(-1)  # [B]
            outputs.append(pred)

            if self.training and torch.rand(1).item() < teacher_forcing_ratio and step + 1 < self.horizon:
                current_input = decoder_input[:, step + 1].unsqueeze(-1)
            else:
                current_input = pred.unsqueeze(-1)

        predictions = torch.stack(outputs, dim=1)
        return predictions


# ----------------------------------------------------------------------------------------------------------------------
# Training / Evaluation
# ----------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: Seq2SeqLSTM, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        encoder_input = batch.encoder_input.to(device)
        decoder_input = batch.decoder_input.to(device)
        target = batch.target.to(device)

        preds = model(encoder_input, decoder_input, teacher_forcing_ratio=0.0)
        loss = mse_loss(preds, target)
        total_loss += loss.item() * len(target)
        total_count += len(target)

    rmse = math.sqrt(total_loss / total_count) if total_count else float("nan")
    return {"rmse": rmse}


def train(
    model: Seq2SeqLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    teacher_forcing_ratio: float = 1.0,
    scheduled_sampling_min: float = 0.3,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    history = {"train_rmse": [], "val_rmse": [], "learning_rate": []}
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        epoch_grad_norm = 0.0
        batches = 0

        # Linearly decay teacher forcing ratio, bounded by scheduled_sampling_min
        tf_ratio = max(scheduled_sampling_min, teacher_forcing_ratio * (1 - (epoch - 1) / max(1, epochs - 1)))

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            encoder_input = batch.encoder_input.to(device)
            decoder_input = batch.decoder_input.to(device)
            target = batch.target.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(encoder_input, decoder_input, teacher_forcing_ratio=tf_ratio)
            loss = criterion(preds, target)
            loss.backward()

            grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_losses.append(loss.item())
            epoch_grad_norm += grad_norm.item() if isinstance(grad_norm, Tensor) else float(grad_norm)
            batches += 1

            progress.set_postfix({"loss": f"{loss.item():.4f}", "tf": f"{tf_ratio:.2f}", "grad": f"{grad_norm:.2f}"})

        train_rmse = math.sqrt(np.mean(train_losses)) if train_losses else float("nan")
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["rmse"])

        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_metrics["rmse"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        tqdm.write(
            f"[Epoch {epoch:02d}] train_RMSE={train_rmse:.3f} | "
            f"val_RMSE={val_metrics['rmse']:.3f} | grad_norm={epoch_grad_norm / max(1, batches):.3f}"
        )

    return history


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seq2Seq LSTM for multi-step gold difference forecasting.")
    parser.add_argument(
        "--data",
        type=str,
        default="Data/processed/featured_data.parquet",
        help="Path to the Parquet file containing match timelines.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimensionality of encoder/decoder.")
    parser.add_argument("--layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout in LSTMs (ignored if layers=1).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--teacher-forcing", type=float, default=1.0, help="Initial teacher forcing ratio.")
    parser.add_argument("--scheduled-min", type=float, default=0.3, help="Minimum teacher forcing ratio.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Global gradient norm clip.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio (rest is validation).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="results/seq2seq_lstm", help="Directory for outputs.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = MultiStepLSTMDataset(
        parquet_path=args.data,
        encoder_length=ENCODER_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
        feature_columns=FEATURE_COLUMNS,
        stride=1,
        fit_scaler=True,
    )

    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=1 - args.train_split, random_state=args.seed, shuffle=True)

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqLSTM(
        input_dim=INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        horizon=FORECAST_HORIZON,
    ).to(device)

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        teacher_forcing_ratio=args.teacher_forcing,
        scheduled_sampling_min=args.scheduled_min,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "seq2seq_lstm.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": dataset.scaler.mean_.tolist(),
            "scaler_scale": dataset.scaler.scale_.tolist(),
            "feature_columns": FEATURE_COLUMNS,
            "history": history,
            "config": vars(args),
        },
        model_path,
    )

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    metrics = evaluate(model, val_loader, device)
    with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(f"Model saved to     : {model_path}")
    print(f"Validation metrics : {metrics}")


if __name__ == "__main__":
    main(parse_args())



