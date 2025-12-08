#!/usr/bin/env python3
"""
Analyze a trained win-probability classifier at fixed game times.

For each requested minute mark (e.g., 10/20/30/40):
  - Evaluate games that reach at least that timestamp
  - Select the slice whose final timestamp is the latest frame <= threshold
  - Report accuracy and confusion matrix
  - Plot precision-recall curve alongside the confusion matrix

Supports both LSTM and Temporal Transformer classifiers.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    auc,
)

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import load_data_splits  # noqa: E402
from models.lstm_win_classifier import (  # noqa: E402
    WinRateLSTM,
    WinRateSequenceDataset,
    collate_fn,
    get_specified_features,
)
from models.temporal_transformer_win_classifier import (  # noqa: E402
    TemporalTransformerClassifier,
)
from models.multi_task_transformer_win_classifier import (  # noqa: E402
    MultiTaskTemporalTransformer,
    MultiTaskSequenceDataset,
)


def load_checkpoint(model_path: str):
    # Use weights_only=False for local checkpoints that contain numpy objects (e.g., scalers)
    # PyTorch 2.6+ defaults to weights_only=True for security, but our checkpoints are trusted
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "feature_scaler" in checkpoint and checkpoint["feature_scaler"] is not None:
        checkpoint["feature_scaler"] = pickle.loads(checkpoint["feature_scaler"])
    else:
        checkpoint["feature_scaler"] = None
    return checkpoint


def extract_time_interval_from_path(file_path: str) -> Optional[int]:
    """
    Extract time interval (in minutes) from filename.
    Examples:
        'test_10min.parquet' -> 10
        'reduced_test_20min.parquet' -> 20
        'test.parquet' -> None
    """
    filename = os.path.basename(file_path)
    match = re.search(r'(\d+)min', filename)
    if match:
        return int(match.group(1))
    return None


def _resolve_feature_list(
    raw_feature_cols,
    dataset_columns: List[str],
    is_multi_task: bool = False,
):
    # Base exclusion list (metadata only)
    excluded_base = {"match_id", "frame_idx", "timestamp", "Y_won", "puuid", "team"}
    
    # NOTE: For multi-task models, we now INCLUDE Elite_Monster_Killed_Difference and 
    # Buildings_Taken_Difference as autoregressive features (historical values from frames 0 to t-1).
    # They are NOT excluded from features anymore.
    
    if raw_feature_cols is None:
        return [col for col in dataset_columns if col not in excluded_base]

    if isinstance(raw_feature_cols, str):
        if raw_feature_cols.lower() == "specified":
            raw_feature_cols = get_specified_features()
        elif raw_feature_cols.lower() == "none":
            return [col for col in dataset_columns if col not in excluded_base]
        elif raw_feature_cols.endswith(".csv"):
            csv_features = get_specified_features(raw_feature_cols)
            raw_feature_cols = csv_features
        else:
            raw_feature_cols = [feat.strip() for feat in raw_feature_cols.split(",") if feat.strip()]

    available = set(dataset_columns)
    filtered = [feat for feat in raw_feature_cols if feat in available and feat not in excluded_base]
    missing = [feat for feat in raw_feature_cols if feat not in available]
    if missing:
        print(f"⚠ Warning: {len(missing)} features missing from dataset; ignoring (showing up to 5): {missing[:5]}")
    if not filtered:
        raise ValueError("No valid features available after filtering; please check feature list and dataset columns.")
    return filtered


def build_test_dataset(
    feature_cols_raw,
    scaler,
    args_from_ckpt: Dict,
    test_data_path: str = None,
    truncate_to_timestamps: int = None,
    is_multi_task: bool = False,
) -> WinRateSequenceDataset:
    """
    Build test dataset, optionally from a custom path and with sequence truncation.
    
    Args:
        feature_cols_raw: Raw feature columns from checkpoint
        scaler: Fitted scaler
        args_from_ckpt: Arguments from checkpoint
        test_data_path: Optional path to test parquet file (if None, uses default)
        truncate_to_timestamps: If specified, truncate each game to exactly this many timestamps
    """
    if test_data_path is not None:
        test_data = pd.read_parquet(test_data_path)
        print(f"📂 Loaded test data from: {test_data_path} ({len(test_data):,} rows)")
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_splits_dir = os.path.join(project_root, "data", "splits")
        _, _, test_data = load_data_splits(data_splits_dir)
        print(f"📂 Using default test data from: {data_splits_dir} ({len(test_data):,} rows)")

    # If truncate_to_timestamps is specified, truncate each game using frame_idx
    # Use frame_idx to select the first N timestamps (frame_idx 0 to frame_idx N-1)
    if truncate_to_timestamps is not None:
        print(f"✂️  Truncating each game to {truncate_to_timestamps} timestamps using frame_idx (frame_idx < {truncate_to_timestamps})")
        truncated_rows = []
        games_included = 0
        games_excluded = 0
        for match_id in test_data["match_id"].unique():
            match_df = test_data[test_data["match_id"] == match_id].sort_values("frame_idx").reset_index(drop=True)
            # Filter by frame_idx: take only frames where frame_idx < truncate_to_timestamps
            # This gives us frame_idx 0, 1, 2, ..., (truncate_to_timestamps-1)
            truncated_match = match_df[match_df["frame_idx"] < truncate_to_timestamps].copy()
            if len(truncated_match) >= truncate_to_timestamps:
                # Game has at least N timestamps, use exactly the first N (frame_idx 0 to N-1)
                # Since we filtered by frame_idx < N, we already have frames 0 to N-1
                truncated_rows.append(truncated_match)
                games_included += 1
            else:
                # Game has fewer than N timestamps, exclude it
                games_excluded += 1
        if truncated_rows:
            test_data = pd.concat(truncated_rows, ignore_index=True)
            print(f"   After truncation: {len(test_data):,} rows from {games_included} games")
            if games_excluded > 0:
                print(f"   Excluded {games_excluded} games with fewer than {truncate_to_timestamps} timestamps")
            print(f"   Each included game has exactly {truncate_to_timestamps} timestamps (frame_idx 0 to {truncate_to_timestamps-1})")
        else:
            print(f"⚠ Warning: No games had at least {truncate_to_timestamps} timestamps")

    max_len = args_from_ckpt.get("max_sequence_length", 40)
    min_len = args_from_ckpt.get("min_sequence_length", 5)

    # For multi-task models with autoregressive features, use checkpoint's feature list directly
    # The checkpoint's feature_cols already includes outcome variables as autoregressive features
    if is_multi_task and feature_cols_raw is not None:
        # Use checkpoint's feature list directly, only filter out columns that don't exist in test data
        available_cols = set(test_data.columns)
        feature_cols = [f for f in feature_cols_raw if f in available_cols]
        missing = [f for f in feature_cols_raw if f not in available_cols]
        if missing:
            print(f"⚠ Warning: {len(missing)} features from checkpoint missing in test data: {missing[:5]}...")
    else:
        # For single-task or when feature_cols_raw is None, use the standard resolution
        feature_cols = _resolve_feature_list(feature_cols_raw, test_data.columns.tolist(), is_multi_task=is_multi_task)

    if is_multi_task:
        # Use MultiTaskSequenceDataset which binarizes elite and buildings targets
        dataset = MultiTaskSequenceDataset(
            test_data,
            feature_cols=feature_cols,
            target_col="Y_won",
            max_sequence_length=max_len,
            min_sequence_length=min_len,
            scaler=scaler,
            fit_scaler=False,
            use_prefix_data=False,  # Use full game sequences (already truncated if needed)
            min_cutoff_ratio=1.0,  # Don't apply cutoff, use full truncated sequence
            max_cutoff_ratio=1.0,
            num_prefix_sequences=1,
        )
    else:
        dataset = WinRateSequenceDataset(
            test_data,
            feature_cols=feature_cols,
            target_col="Y_won",
            max_sequence_length=max_len,
            min_sequence_length=min_len,
            scaler=scaler,
            fit_scaler=False,
            use_prefix_data=False,  # Use full game sequences (already truncated if needed)
            min_cutoff_ratio=1.0,  # Don't apply cutoff, use full truncated sequence
            max_cutoff_ratio=1.0,
            num_prefix_sequences=1,
        )
    return dataset


def select_indices_for_stage(
    dataset: WinRateSequenceDataset,
    threshold_ms: int,
) -> List[int]:
    per_match: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for idx, match_id in enumerate(dataset.match_ids):
        ts = dataset.end_timestamps[idx]
        per_match[match_id].append((ts, idx))

    selected = []
    for match_id, entries in per_match.items():
        entries.sort(key=lambda x: x[0])
        if entries[-1][0] < threshold_ms:
            continue  # game shorter than threshold
        # choose the slice closest to but not exceeding the threshold
        candidate = None
        for ts, idx in entries:
            if ts <= threshold_ms:
                candidate = (ts, idx)
            else:
                break
        if candidate is not None:
            selected.append(candidate[1])
    return selected


def run_inference(
    model: torch.nn.Module,
    dataset: WinRateSequenceDataset,
    indices: List[int],
    batch_size: int,
    device: torch.device,
    is_multi_task: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run inference on the dataset.
    
    Returns:
        For single-task: (probs, labels, None, None, None, None)
        For multi-task: (win_probs, win_labels, elite_probs, elite_labels, buildings_probs, buildings_labels)
    """
    if not indices:
        if is_multi_task:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        else:
            return np.array([]), np.array([]), None, None, None, None

    subset = torch.utils.data.Subset(dataset, indices)
    
    # Use appropriate collate function
    if is_multi_task:
        from models.multi_task_transformer_win_classifier import multi_task_collate_fn
        collate_fn_to_use = multi_task_collate_fn
    else:
        collate_fn_to_use = collate_fn
    
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_to_use,
    )

    model.eval()
    probs_all = []
    labels_all = []
    
    # Multi-task specific
    elite_probs_all = []
    elite_labels_all = []
    buildings_probs_all = []
    buildings_labels_all = []

    with torch.no_grad():
        for batch in loader:
            sequences = batch["sequences"].to(device)
            lengths = batch["lengths"].to(device)
            
            if is_multi_task:
                win_labels = batch["win_labels"].to(device)
                elite_labels = batch["elite_labels"].to(device)
                buildings_labels = batch["buildings_labels"].to(device)
                
                outputs = model(sequences, lengths)
                # Multi-task model returns dict with win_logit, elite_logit, buildings_logit
                win_logits = outputs["win_logit"]
                elite_logits = outputs["elite_logit"]
                buildings_logits = outputs["buildings_logit"]
                
                # Ensure labels are 1D
                if win_labels.dim() > 1:
                    win_labels = win_labels.squeeze(-1)
                if elite_labels.dim() > 1:
                    elite_labels = elite_labels.squeeze(-1)
                if buildings_labels.dim() > 1:
                    buildings_labels = buildings_labels.squeeze(-1)
                
                win_probs = torch.sigmoid(win_logits)
                elite_probs = torch.sigmoid(elite_logits)
                buildings_probs = torch.sigmoid(buildings_logits)
                
                probs_all.append(win_probs.cpu().numpy())
                labels_all.append(win_labels.cpu().numpy())
                elite_probs_all.append(elite_probs.cpu().numpy())
                elite_labels_all.append(elite_labels.cpu().numpy())
                buildings_probs_all.append(buildings_probs.cpu().numpy())
                buildings_labels_all.append(buildings_labels.cpu().numpy())
            else:
                labels = batch["labels"].to(device)
                logits = model(sequences, lengths)
                
                # Ensure labels are 1D
                if labels.dim() > 1:
                    labels = labels.squeeze(-1)
                
                probs = torch.sigmoid(logits)
                probs_all.append(probs.cpu().numpy())
                labels_all.append(labels.cpu().numpy())

    if probs_all:
        if is_multi_task:
            return (
                np.concatenate(probs_all),
                np.concatenate(labels_all),
                np.concatenate(elite_probs_all),
                np.concatenate(elite_labels_all),
                np.concatenate(buildings_probs_all),
                np.concatenate(buildings_labels_all),
            )
        else:
            return np.concatenate(probs_all), np.concatenate(labels_all), None, None, None, None
    else:
        if is_multi_task:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        else:
            return np.array([]), np.array([]), None, None, None, None


def build_model(model_type: str, checkpoint: Dict, device: torch.device) -> torch.nn.Module:
    args = checkpoint.get("args", {})
    input_size = checkpoint.get("input_size")

    if model_type == "lstm":
        model = WinRateLSTM(
            input_size=input_size,
            hidden_size=args.get("hidden_size", 128),
            num_layers=args.get("num_layers", 2),
            dropout=args.get("dropout", 0.3),
            bidirectional=args.get("bidirectional", False),
        )
    elif model_type == "transformer":
        model = TemporalTransformerClassifier(
            input_size=input_size,
            d_model=args.get("d_model", 256),
            nhead=args.get("nhead", 8),
            num_layers=args.get("num_layers", 4),
            dim_feedforward=args.get("dim_feedforward", 512),
            dropout=args.get("dropout", 0.1),
        )
    elif model_type == "multi_task_transformer":
        # Check if checkpoint has adaptive loss parameters
        state_dict_keys = checkpoint.get("model_state_dict", {}).keys()
        has_adaptive_loss = any(
            key.startswith("log_var_win") or 
            key.startswith("log_var_elite") or 
            key.startswith("log_var_buildings")
            for key in state_dict_keys
        )
        
        if has_adaptive_loss:
            print("🔧 Detected adaptive loss parameters in checkpoint - using adaptive loss model")
        
        model = MultiTaskTemporalTransformer(
            input_size=input_size,
            d_model=args.get("d_model", 256),
            nhead=args.get("nhead", 8),
            num_layers=args.get("num_layers", 4),
            dim_feedforward=args.get("dim_feedforward", 512),
            dropout=args.get("dropout", 0.1),
            use_adaptive_loss=has_adaptive_loss,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def plot_results(stage_metrics: List[Dict], output_path: str):
    if not stage_metrics:
        print("No stages to plot.")
        return

    rows = len(stage_metrics)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 4 * rows))
    if rows == 1:
        axes = np.array([axes])

    for row_idx, metrics in enumerate(stage_metrics):
        cm_ax, pr_ax, acc_ax = axes[row_idx]

        cm = metrics["confusion_matrix"]
        im = cm_ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        cm_ax.figure.colorbar(im, ax=cm_ax, fraction=0.046, pad=0.04)
        cm_ax.set_title(f"{metrics['label']} – Confusion Matrix")
        cm_ax.set_xticks([0, 1])
        cm_ax.set_yticks([0, 1])
        cm_ax.set_xticklabels(["Pred 0", "Pred 1"])
        cm_ax.set_yticklabels(["True 0", "True 1"])
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cm_ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        precision = metrics["precision"]
        recall = metrics["recall"]
        pr_auc = metrics["pr_auc"]
        pr_ax.plot(recall, precision, label=f"AUC={pr_auc:.3f}")
        pr_ax.set_xlabel("Recall")
        pr_ax.set_ylabel("Precision")
        pr_ax.set_title(f"{metrics['label']} – Precision-Recall Curve")
        pr_ax.set_xlim([0.0, 1.05])
        pr_ax.set_ylim([0.0, 1.05])
        pr_ax.grid(True)
        pr_ax.legend(loc="lower left")

        bin_centers = np.array(metrics["bin_centers"])
        bin_accuracy = np.array(metrics["bin_accuracy"])
        bin_actual = np.array(metrics["bin_actual"])
        bin_predicted = np.array(metrics["bin_predicted"])
        bin_counts = np.array(metrics["bin_counts"])
        bin_edges = np.array(metrics["bin_edges"])

        bin_width = 0.8 * (bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 0.15
        acc_ax.bar(bin_centers, np.nan_to_num(bin_accuracy, nan=0.0), width=bin_width, alpha=0.35, label="Bin Accuracy")
        acc_ax.plot(bin_centers, bin_actual, "-o", label="Actual Win Rate")
        acc_ax.plot(bin_centers, bin_predicted, "--x", label="Mean Predicted Prob")
        acc_ax.set_xlabel("Predicted Win Probability")
        acc_ax.set_ylabel("Rate")
        acc_ax.set_ylim([0.0, 1.05])
        acc_ax.set_title(
            f"{metrics['label']} – Accuracy by Probability Bin\nOverall acc: {metrics['accuracy']:.3f}"
        )
        acc_ax.grid(True, axis="y", alpha=0.3)
        for x, count in zip(bin_centers, bin_counts):
            if count > 0:
                acc_ax.text(x, 0.02, str(count), ha="center", va="bottom", fontsize=8, rotation=90)
        acc_ax.legend(loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved stage analysis plots to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze win classifier accuracy at specific game times",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to classifier checkpoint")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["lstm", "transformer", "multi_task_transformer"],
        default=None,
        help="Classifier backbone type (auto-detected if omitted)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="10,20,30,40",
        help="Comma separated minute marks to evaluate (e.g., '10,20,30,40')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size during evaluation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the confusion matrix + PR curves figure "
             "(defaults to results/Classifier_Analysis/<model_name>/stage_metrics.png)",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
        help="Path to custom test dataset (e.g., data/splits/test_10min.parquet). "
             "If provided, will extract time interval from filename and truncate sequences accordingly. "
             "Can be a single path or comma-separated list of paths for multiple intervals.",
    )

    args = parser.parse_args()

    stages_minutes = [int(s.strip()) for s in args.stages.split(",") if s.strip()]
    if not stages_minutes:
        raise ValueError("No valid stages provided.")

    checkpoint = load_checkpoint(args.model_path)
    state_dict_keys = checkpoint.get("model_state_dict", {}).keys()

    def detect_model_type(keys) -> str:
        # Check for multi-task transformer (has elite_head and buildings_head)
        if any(k.startswith("elite_head") or k.startswith("buildings_head") for k in keys):
            return "multi_task_transformer"
        # Check for regular transformer
        if any(k.startswith("input_projection") or k.startswith("transformer_encoder") for k in keys):
            return "transformer"
        # Check for LSTM
        if any(k.startswith("lstm.") for k in keys):
            return "lstm"
        return "lstm"

    detected_type = detect_model_type(state_dict_keys)
    model_type = args.model_type or detected_type
    is_multi_task = (model_type == "multi_task_transformer")
    
    if args.model_type and args.model_type != detected_type:
        print(
            f"⚠ Requested model_type '{args.model_type}' does not match checkpoint architecture '{detected_type}'. "
            f"Using '{detected_type}' instead."
        )
        model_type = detected_type
        is_multi_task = (model_type == "multi_task_transformer")
    
    if is_multi_task:
        print("🔍 Detected Multi-Task Transformer model (will analyze all three tasks: Win, Elite Monsters, Buildings)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_type, checkpoint, device)

    # Parse test_data_path - can be single path or comma-separated list
    test_data_paths = []
    if args.test_data_path:
        test_data_paths = [p.strip() for p in args.test_data_path.split(",") if p.strip()]
    
    model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
    default_dir = os.path.join("results", "Classifier_Analysis", model_basename)
    os.makedirs(default_dir, exist_ok=True)
    output_path = args.output_path or os.path.join(default_dir, "stage_metrics.png")

    stage_metrics = []
    batch_size = args.batch_size

    # If using custom test dataset(s) with truncation, evaluate each one
    if test_data_paths:
        print(f"\n📊 Evaluating on {len(test_data_paths)} test dataset(s)")
        
        for test_data_path in test_data_paths:
            truncate_to_timestamps = extract_time_interval_from_path(test_data_path)
            if truncate_to_timestamps:
                print(f"\n📊 Processing: {test_data_path}")
                print(f"   Detected time interval: {truncate_to_timestamps} minutes")
                print(f"   Will truncate each game to {truncate_to_timestamps} timestamps")
            else:
                print(f"\n⚠ Could not extract time interval from filename: {test_data_path}")
                print(f"   Will use full sequences")
                truncate_to_timestamps = None
            
            dataset = build_test_dataset(
                checkpoint.get("feature_cols"),
                checkpoint.get("feature_scaler"),
                checkpoint.get("args", {}),
                test_data_path=test_data_path,
                truncate_to_timestamps=truncate_to_timestamps,
                is_multi_task=is_multi_task,
            )
            
            # Evaluate on all sequences (they're already truncated to the specified length)
            all_indices = list(range(len(dataset)))
            result = run_inference(model, dataset, all_indices, batch_size, device, is_multi_task=is_multi_task)
            
            if is_multi_task:
                win_probs, win_labels, elite_probs, elite_labels, buildings_probs, buildings_labels = result
                probs, labels = win_probs, win_labels  # For backward compatibility in plotting
            else:
                probs, labels = result[:2]
                elite_probs, elite_labels, buildings_probs, buildings_labels = None, None, None, None
            
            if probs.size > 0:
                # Compute metrics for win task
                preds = (probs >= 0.5).astype(int)
                acc = accuracy_score(labels, preds)
                cm = confusion_matrix(labels, preds, labels=[0, 1])
                prec_scalar = precision_score(labels, preds, zero_division=0)
                recall_scalar = recall_score(labels, preds, zero_division=0)
                f1_scalar = f1_score(labels, preds, zero_division=0)

                try:
                    precision, recall, _ = precision_recall_curve(labels, probs)
                    pr_auc = auc(recall, precision)
                except ValueError:
                    precision = np.array([0, 1])
                    recall = np.array([0, 1])
                    pr_auc = float("nan")

                # Bin analysis for win task
                bin_edges = np.linspace(0.0, 1.0, 6)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                bin_actual = []
                bin_predicted = []
                bin_accuracy = []
                bin_counts = []
                for idx_bin in range(len(bin_edges) - 1):
                    lower = bin_edges[idx_bin]
                    upper = bin_edges[idx_bin + 1]
                    if idx_bin == len(bin_edges) - 2:
                        mask = (probs >= lower) & (probs <= upper)
                    else:
                        mask = (probs >= lower) & (probs < upper)
                    count = int(mask.sum())
                    bin_counts.append(count)
                    if count > 0:
                        bin_actual.append(float(labels[mask].mean()))
                        bin_predicted.append(float(probs[mask].mean()))
                        bin_accuracy.append(float((preds[mask] == labels[mask]).mean()))
                    else:
                        bin_actual.append(float("nan"))
                        bin_predicted.append(float("nan"))
                        bin_accuracy.append(float("nan"))

                # Compute metrics for elite and buildings tasks (if multi-task)
                elite_metrics = None
                buildings_metrics = None
                if is_multi_task and elite_probs is not None and elite_probs.size > 0:
                    elite_preds = (elite_probs >= 0.5).astype(int)
                    elite_acc = accuracy_score(elite_labels, elite_preds)
                    elite_prec = precision_score(elite_labels, elite_preds, zero_division=0)
                    elite_recall = recall_score(elite_labels, elite_preds, zero_division=0)
                    elite_f1 = f1_score(elite_labels, elite_preds, zero_division=0)
                    elite_metrics = {
                        "accuracy": elite_acc,
                        "precision": elite_prec,
                        "recall": elite_recall,
                        "f1": elite_f1,
                    }
                
                if is_multi_task and buildings_probs is not None and buildings_probs.size > 0:
                    buildings_preds = (buildings_probs >= 0.5).astype(int)
                    buildings_acc = accuracy_score(buildings_labels, buildings_preds)
                    buildings_prec = precision_score(buildings_labels, buildings_preds, zero_division=0)
                    buildings_recall = recall_score(buildings_labels, buildings_preds, zero_division=0)
                    buildings_f1 = f1_score(buildings_labels, buildings_preds, zero_division=0)
                    buildings_metrics = {
                        "accuracy": buildings_acc,
                        "precision": buildings_prec,
                        "recall": buildings_recall,
                        "f1": buildings_f1,
                    }

                label_minutes = truncate_to_timestamps if truncate_to_timestamps else "full"
                stage_metrics.append({
                    "label": f"{label_minutes} min (n={len(labels)})",
                    "minutes": truncate_to_timestamps if truncate_to_timestamps else 0,
                    "samples": len(labels),
                    "accuracy": acc,
                    "precision_scalar": prec_scalar,
                    "recall_scalar": recall_scalar,
                    "f1_scalar": f1_scalar,
                    "confusion_matrix": cm,
                    "precision": precision,
                    "recall": recall,
                    "pr_auc": pr_auc,
                    "bin_edges": bin_edges.tolist(),
                    "bin_centers": bin_centers.tolist(),
                    "bin_actual": bin_actual,
                    "bin_predicted": bin_predicted,
                    "bin_accuracy": bin_accuracy,
                    "bin_counts": bin_counts,
                    "elite_metrics": elite_metrics,
                    "buildings_metrics": buildings_metrics,
                })

                print(f"=== {label_minutes} minute mark ===")
                print(f"Samples: {len(labels)}")
                print(f"Win Task - Accuracy: {acc:.4f}, Precision: {prec_scalar:.4f}, Recall: {recall_scalar:.4f}, F1: {f1_scalar:.4f}")
                if elite_metrics:
                    print(f"Elite Task - Accuracy: {elite_metrics['accuracy']:.4f}, Precision: {elite_metrics['precision']:.4f}, Recall: {elite_metrics['recall']:.4f}, F1: {elite_metrics['f1']:.4f}")
                if buildings_metrics:
                    print(f"Buildings Task - Accuracy: {buildings_metrics['accuracy']:.4f}, Precision: {buildings_metrics['precision']:.4f}, Recall: {buildings_metrics['recall']:.4f}, F1: {buildings_metrics['f1']:.4f}")
                print("Confusion matrix (Win):")
                print(cm)
                print(f"PR AUC (Win): {pr_auc:.4f}")
                print("")
            else:
                print(f"⚠ No predictions produced for {test_data_path}")
    else:
        # Original stage-based evaluation (no custom test data path)
        # Build dataset using default test data
        dataset = build_test_dataset(
            checkpoint.get("feature_cols"),
            checkpoint.get("feature_scaler"),
            checkpoint.get("args", {}),
            test_data_path=None,  # Use default test data
            truncate_to_timestamps=None,  # Use full sequences
            is_multi_task=is_multi_task,
        )
        
        for minutes in stages_minutes:
            threshold_ms = minutes * 60 * 1000
            indices = select_indices_for_stage(dataset, threshold_ms)

            if not indices:
                print(f"⚠ No qualifying sequences for {minutes} minutes (games too short). Skipping.")
                continue

            result = run_inference(model, dataset, indices, batch_size, device, is_multi_task=is_multi_task)
            
            if is_multi_task:
                win_probs, win_labels, elite_probs, elite_labels, buildings_probs, buildings_labels = result
                probs, labels = win_probs, win_labels  # For backward compatibility in plotting
            else:
                probs, labels = result[:2]
                elite_probs, elite_labels, buildings_probs, buildings_labels = None, None, None, None
            
            if probs.size == 0:
                print(f"⚠ No predictions produced for {minutes} minutes. Skipping.")
                continue

            # Compute metrics for win task
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(labels, preds)
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            prec_scalar = precision_score(labels, preds, zero_division=0)
            recall_scalar = recall_score(labels, preds, zero_division=0)
            f1_scalar = f1_score(labels, preds, zero_division=0)

            try:
                precision, recall, _ = precision_recall_curve(labels, probs)
                pr_auc = auc(recall, precision)
            except ValueError:
                precision = np.array([0, 1])
                recall = np.array([0, 1])
                pr_auc = float("nan")

            bin_edges = np.linspace(0.0, 1.0, 6)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_actual = []
            bin_predicted = []
            bin_accuracy = []
            bin_counts = []
            for idx_bin in range(len(bin_edges) - 1):
                lower = bin_edges[idx_bin]
                upper = bin_edges[idx_bin + 1]
                if idx_bin == len(bin_edges) - 2:
                    mask = (probs >= lower) & (probs <= upper)
                else:
                    mask = (probs >= lower) & (probs < upper)
                count = int(mask.sum())
                bin_counts.append(count)
                if count > 0:
                    bin_actual.append(float(labels[mask].mean()))
                    bin_predicted.append(float(probs[mask].mean()))
                    bin_accuracy.append(float((preds[mask] == labels[mask]).mean()))
                else:
                    bin_actual.append(float("nan"))
                    bin_predicted.append(float("nan"))
                    bin_accuracy.append(float("nan"))

            # Compute metrics for elite and buildings tasks (if multi-task)
            elite_metrics = None
            buildings_metrics = None
            if is_multi_task and elite_probs is not None and elite_probs.size > 0:
                elite_preds = (elite_probs >= 0.5).astype(int)
                elite_acc = accuracy_score(elite_labels, elite_preds)
                elite_prec = precision_score(elite_labels, elite_preds, zero_division=0)
                elite_recall = recall_score(elite_labels, elite_preds, zero_division=0)
                elite_f1 = f1_score(elite_labels, elite_preds, zero_division=0)
                elite_metrics = {
                    "accuracy": elite_acc,
                    "precision": elite_prec,
                    "recall": elite_recall,
                    "f1": elite_f1,
                }
            
            if is_multi_task and buildings_probs is not None and buildings_probs.size > 0:
                buildings_preds = (buildings_probs >= 0.5).astype(int)
                buildings_acc = accuracy_score(buildings_labels, buildings_preds)
                buildings_prec = precision_score(buildings_labels, buildings_preds, zero_division=0)
                buildings_recall = recall_score(buildings_labels, buildings_preds, zero_division=0)
                buildings_f1 = f1_score(buildings_labels, buildings_preds, zero_division=0)
                buildings_metrics = {
                    "accuracy": buildings_acc,
                    "precision": buildings_prec,
                    "recall": buildings_recall,
                    "f1": buildings_f1,
                }

            stage_metrics.append(
                {
                    "label": f"{minutes} min (n={len(labels)})",
                    "minutes": minutes,
                    "samples": len(labels),
                    "accuracy": acc,
                    "precision_scalar": prec_scalar,
                    "recall_scalar": recall_scalar,
                    "f1_scalar": f1_scalar,
                    "confusion_matrix": cm,
                    "precision": precision,
                    "recall": recall,
                    "pr_auc": pr_auc,
                    "bin_edges": bin_edges.tolist(),
                    "bin_centers": bin_centers.tolist(),
                    "bin_actual": bin_actual,
                    "bin_predicted": bin_predicted,
                    "bin_accuracy": bin_accuracy,
                    "bin_counts": bin_counts,
                    "elite_metrics": elite_metrics,
                    "buildings_metrics": buildings_metrics,
                }
            )

            print(f"=== {minutes} minute mark ===")
            print(f"Samples: {len(labels)}")
            print(f"Win Task - Accuracy: {acc:.4f}, Precision: {prec_scalar:.4f}, Recall: {recall_scalar:.4f}, F1: {f1_scalar:.4f}")
            if elite_metrics:
                print(f"Elite Task - Accuracy: {elite_metrics['accuracy']:.4f}, Precision: {elite_metrics['precision']:.4f}, Recall: {elite_metrics['recall']:.4f}, F1: {elite_metrics['f1']:.4f}")
            if buildings_metrics:
                print(f"Buildings Task - Accuracy: {buildings_metrics['accuracy']:.4f}, Precision: {buildings_metrics['precision']:.4f}, Recall: {buildings_metrics['recall']:.4f}, F1: {buildings_metrics['f1']:.4f}")
            print("Confusion matrix (Win):")
            print(cm)
            print(f"PR AUC (Win): {pr_auc:.4f}")
            print("")

    if stage_metrics:
        # Check if any stage has multi-task metrics
        has_multi_task = any(
            metrics.get("elite_metrics") is not None or metrics.get("buildings_metrics") is not None
            for metrics in stage_metrics
        )
        
        print("=" * 60)
        if has_multi_task:
            print("STAGE SUMMARY (Multi-Task Classification Metrics)")
        else:
            print("STAGE SUMMARY (Classification Metrics)")
        print("=" * 60)
        
        # Win task summary
        header = f"{'Stage':<10}{'Samples':>10}{'Accuracy':>12}{'Precision':>12}{'Recall':>12}{'F1':>10}"
        print("\n📊 Win Task:")
        print(header)
        print("-" * len(header))
        for metrics in stage_metrics:
            stage_label = f"{metrics['minutes']}m"
            samples = metrics["samples"]
            print(
                f"{stage_label:<10}"
                f"{samples:>10}"
                f"{metrics['accuracy']:>12.4f}"
                f"{metrics['precision_scalar']:>12.4f}"
                f"{metrics['recall_scalar']:>12.4f}"
                f"{metrics['f1_scalar']:>10.4f}"
            )
        
        # Elite task summary (if multi-task)
        if has_multi_task:
            print("\n📊 Elite Monster Task:")
            print(header)
            print("-" * len(header))
            for metrics in stage_metrics:
                stage_label = f"{metrics['minutes']}m"
                samples = metrics["samples"]
                elite_metrics = metrics.get("elite_metrics")
                if elite_metrics:
                    print(
                        f"{stage_label:<10}"
                        f"{samples:>10}"
                        f"{elite_metrics['accuracy']:>12.4f}"
                        f"{elite_metrics['precision']:>12.4f}"
                        f"{elite_metrics['recall']:>12.4f}"
                        f"{elite_metrics['f1']:>10.4f}"
                    )
                else:
                    print(f"{stage_label:<10}{samples:>10}{'N/A':>12}{'N/A':>12}{'N/A':>12}{'N/A':>10}")
        
        # Buildings task summary (if multi-task)
        if has_multi_task:
            print("\n📊 Buildings Task:")
            print(header)
            print("-" * len(header))
            for metrics in stage_metrics:
                stage_label = f"{metrics['minutes']}m"
                samples = metrics["samples"]
                buildings_metrics = metrics.get("buildings_metrics")
                if buildings_metrics:
                    print(
                        f"{stage_label:<10}"
                        f"{samples:>10}"
                        f"{buildings_metrics['accuracy']:>12.4f}"
                        f"{buildings_metrics['precision']:>12.4f}"
                        f"{buildings_metrics['recall']:>12.4f}"
                        f"{buildings_metrics['f1']:>10.4f}"
                    )
                else:
                    print(f"{stage_label:<10}{samples:>10}{'N/A':>12}{'N/A':>12}{'N/A':>12}{'N/A':>10}")
        
        print("=" * 60)

    plot_results(stage_metrics, output_path)


if __name__ == "__main__":
    main()


