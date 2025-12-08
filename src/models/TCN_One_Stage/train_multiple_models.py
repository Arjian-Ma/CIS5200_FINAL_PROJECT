"""
Train multiple TCN One Stage models with different time windows (10, 15, 20, 25, 30 minutes).
Each model uses the first X minutes of match data, excluding games shorter than X minutes.
"""

import sys
from pathlib import Path
from typing import Optional, List
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add parent directory to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

# Import from the same directory
import importlib.util
model_path = Path(__file__).parent / "TCN_One_Stage.py"
spec = importlib.util.spec_from_file_location("tcn_one_stage", model_path)
tcn_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tcn_module
spec.loader.exec_module(tcn_module)

HierarchicalWinPredictor = tcn_module.HierarchicalWinPredictor
WinPredictionDataset = tcn_module.WinPredictionDataset
collate_fn = tcn_module.collate_fn
ALL_FEATURE_COLUMNS = tcn_module.ALL_FEATURE_COLUMNS
TARGET_COLUMN = tcn_module.TARGET_COLUMN
GROUP_COLUMN = tcn_module.GROUP_COLUMN
TIME_COLUMN = tcn_module.TIME_COLUMN

# Data paths
DATA_DIR = ROOT / "Data/processed"
TRAIN_PATH = DATA_DIR / "train.parquet"
VAL_PATH = DATA_DIR / "val.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
OUTPUT_DIR = ROOT / "results/tcn_one_stage_multiple"

# Model configurations
ENCODER_LENGTHS = [10, 15, 20, 25, 30]


def compute_metrics(preds, targets):
    """Compute accuracy, precision, recall, and F1 score."""
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
    }


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Evaluate model and return predictions and targets."""
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []
    
    for batch in loader:
        x1 = batch.x1.to(device)
        x2 = batch.x2.to(device)
        players = batch.players.to(device)
        target = batch.target.to(device)
        
        logits = model(x1, x2, players)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    return np.array(all_probs), np.array(all_preds), np.array(all_targets)


def train_model(
    encoder_length: int,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    early_stop_patience: int = 15,
    hidden_dim: int = 128,
    proj_dim: int = 128,
    team_embed_dim: int = 32,
    recurrent_dropout: float = 0.2,
    tcn_channels: Optional[List[int]] = None,
    tcn_kernel_size: int = 2,
    tcn_dropout: float = 0.2,
    seed: int = 42,
):
    """Train a single TCN model with specified encoder length."""
    print(f"\n{'=' * 60}")
    print(f"Training TCN Model: {encoder_length} minutes")
    print(f"{'=' * 60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data and check game lengths
    print(f"\n[1/7] Loading data and filtering games...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    # Filter games by minimum length (must have at least encoder_length frames)
    def filter_by_length(df, min_frames):
        match_lengths = df.groupby(GROUP_COLUMN).size()
        valid_matches = match_lengths[match_lengths >= min_frames].index
        return df[df[GROUP_COLUMN].isin(valid_matches)]
    
    train_df_filtered = filter_by_length(train_df, encoder_length)
    val_df_filtered = filter_by_length(val_df, encoder_length)
    test_df_filtered = filter_by_length(test_df, encoder_length)
    
    print(f"  Train: {len(train_df_filtered[GROUP_COLUMN].unique())} matches (from {len(train_df[GROUP_COLUMN].unique())} total)")
    print(f"  Val:   {len(val_df_filtered[GROUP_COLUMN].unique())} matches (from {len(val_df[GROUP_COLUMN].unique())} total)")
    print(f"  Test:  {len(test_df_filtered[GROUP_COLUMN].unique())} matches (from {len(test_df[GROUP_COLUMN].unique())} total)")
    
    # Create datasets
    print(f"\n[2/7] Creating datasets...")
    from sklearn.preprocessing import StandardScaler
    
    # Fit scaler on training data
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_df_filtered[ALL_FEATURE_COLUMNS].values)
    
    train_dataset = WinPredictionDataset(
        parquet_path=train_path,
        encoder_length=encoder_length,
        scaler=feature_scaler,
        fit_scaler=False,
        match_filter=set(train_df_filtered[GROUP_COLUMN].unique()),
        min_timesteps=encoder_length,
    )
    
    val_dataset = WinPredictionDataset(
        parquet_path=val_path,
        encoder_length=encoder_length,
        scaler=feature_scaler,
        fit_scaler=False,
        match_filter=set(val_df_filtered[GROUP_COLUMN].unique()),
        min_timesteps=encoder_length,
    )
    
    test_dataset = WinPredictionDataset(
        parquet_path=test_path,
        encoder_length=encoder_length,
        scaler=feature_scaler,
        fit_scaler=False,
        match_filter=set(test_df_filtered[GROUP_COLUMN].unique()),
        min_timesteps=encoder_length,
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    print(f"\n[3/7] Creating TCN model...")
    model = HierarchicalWinPredictor(
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        team_embed_dim=team_embed_dim,
        condition_on_hidden=True,
        recurrent_dropout=recurrent_dropout,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dropout=tcn_dropout,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,} (total: {total_params:,})")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    bce = torch.nn.BCEWithLogitsLoss()
    
    history = {
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "learning_rate": [],
        "early_stopped_epoch": None,
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    early_stop = False
    best_state = None
    
    # Training loop
    print(f"\n[4/7] Training model...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)
            
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        
        # Evaluate on validation and test sets
        val_probs, val_preds, val_targets = evaluate_model(model, val_loader, device)
        test_probs, test_preds, test_targets = evaluate_model(model, test_loader, device)
        
        val_acc = accuracy_score(val_targets, val_preds)
        test_metrics = compute_metrics(test_preds, test_targets)
        
        scheduler.step(1 - val_acc)  # Step on validation accuracy (inverse)
        
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_metrics["accuracy"])
        history["test_precision"].append(test_metrics["precision"])
        history["test_recall"].append(test_metrics["recall"])
        history["test_f1"].append(test_metrics["f1_score"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        
        # Early stopping
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
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
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f} | "
            f"Test Prec: {test_metrics['precision']:.4f} | "
            f"Test Rec: {test_metrics['recall']:.4f} | "
            f"Test F1: {test_metrics['f1_score']:.4f}"
        )
        
        if early_stop:
            tqdm.write(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    
    # Final evaluation
    print(f"\n[5/7] Final evaluation on all sets...")
    train_probs, train_preds, train_targets = evaluate_model(model, train_loader, device)
    val_probs, val_preds, val_targets = evaluate_model(model, val_loader, device)
    test_probs, test_preds, test_targets = evaluate_model(model, test_loader, device)
    
    train_metrics = compute_metrics(train_preds, train_targets)
    val_metrics = compute_metrics(val_preds, val_targets)
    test_metrics = compute_metrics(test_preds, test_targets)
    
    print(f"  Train - Acc: {train_metrics['accuracy']:.4f}")
    print(f"  Val   - Acc: {val_metrics['accuracy']:.4f}")
    print(f"  Test  - Acc: {test_metrics['accuracy']:.4f}, "
          f"Prec: {test_metrics['precision']:.4f}, "
          f"Rec: {test_metrics['recall']:.4f}, "
          f"F1: {test_metrics['f1_score']:.4f}")
    
    # Save model
    print(f"\n[6/7] Saving model...")
    model_dir = output_dir / f"model_{encoder_length}min"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": feature_scaler.mean_.tolist(),
            "scaler_scale": feature_scaler.scale_.tolist(),
            "feature_columns": ALL_FEATURE_COLUMNS,
            "history": history,
            "config": {
                "encoder_length": encoder_length,
                "hidden_dim": hidden_dim,
                "proj_dim": proj_dim,
                "team_embed_dim": team_embed_dim,
                "recurrent_dropout": recurrent_dropout,
                "tcn_channels": tcn_channels,
                "tcn_kernel_size": tcn_kernel_size,
                "tcn_dropout": tcn_dropout,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
            },
        },
        model_path,
    )
    
    # Save metrics
    print(f"\n[7/7] Saving metrics...")
    final_metrics = {
        "encoder_length": encoder_length,
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1_score": test_metrics["f1_score"],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_matches": len(train_df_filtered[GROUP_COLUMN].unique()),
        "val_matches": len(val_df_filtered[GROUP_COLUMN].unique()),
        "test_matches": len(test_df_filtered[GROUP_COLUMN].unique()),
    }
    
    metrics_path = model_dir / "final_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    
    history_path = model_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Metrics saved to: {metrics_path}")
    print(f"✓ History saved to: {history_path}")
    
    return final_metrics, history


def main():
    print("=" * 60)
    print("TRAINING MULTIPLE TCN ONE STAGE MODELS")
    print("=" * 60)
    print(f"\nModels to train: {ENCODER_LENGTHS} minutes")
    print(f"Train data: {TRAIN_PATH}")
    print(f"Val data:   {VAL_PATH}")
    print(f"Test data:  {TEST_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    
    # Check if data files exist
    for path, name in [(TRAIN_PATH, "train"), (VAL_PATH, "val"), (TEST_PATH, "test")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} data file not found: {path}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train all models
    all_metrics = {}
    for encoder_length in ENCODER_LENGTHS:
        try:
            metrics, history = train_model(
                encoder_length=encoder_length,
                train_path=TRAIN_PATH,
                val_path=VAL_PATH,
                test_path=TEST_PATH,
                output_dir=OUTPUT_DIR,
                epochs=15,
                batch_size=128,
                lr=1e-3,
                grad_clip=1.0,
                early_stop_patience=15,
                tcn_channels=[128, 128, 128],  # 3 TCN layers
                tcn_kernel_size=2,
                tcn_dropout=0.2,
            )
            all_metrics[f"{encoder_length}min"] = metrics
        except Exception as e:
            print(f"\n❌ Error training {encoder_length}min model: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")
    
    summary_path = OUTPUT_DIR / "summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    
    print("\nFinal Metrics Summary:")
    print(f"{'Model':<10} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Test Prec':<12} {'Test Rec':<12} {'Test F1':<12}")
    print("-" * 90)
    for model_name, metrics in sorted(all_metrics.items()):
        print(
            f"{model_name:<10} "
            f"{metrics['train_accuracy']:<12.4f} "
            f"{metrics['val_accuracy']:<12.4f} "
            f"{metrics['test_accuracy']:<12.4f} "
            f"{metrics['test_precision']:<12.4f} "
            f"{metrics['test_recall']:<12.4f} "
            f"{metrics['test_f1_score']:<12.4f}"
        )
    
    print(f"\n✓ Summary saved to: {summary_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

