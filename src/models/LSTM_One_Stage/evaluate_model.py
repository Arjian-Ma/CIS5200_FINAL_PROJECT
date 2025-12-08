"""
Evaluate the trained LSTM One Stage model and compute detailed metrics:
Accuracy, Precision, Recall, and F1 Score.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Add parent directory to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

# Import from the same directory
import importlib.util
model_path = Path(__file__).parent / "LSTM_One_Stage.py"
spec = importlib.util.spec_from_file_location("lstm_one_stage", model_path)
lstm_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lstm_module
spec.loader.exec_module(lstm_module)

HierarchicalWinPredictor = lstm_module.HierarchicalWinPredictor
WinPredictionDataset = lstm_module.WinPredictionDataset
collate_fn = lstm_module.collate_fn
ALL_FEATURE_COLUMNS = lstm_module.ALL_FEATURE_COLUMNS
ENCODER_LENGTH = lstm_module.ENCODER_LENGTH
TARGET_COLUMN = lstm_module.TARGET_COLUMN
GROUP_COLUMN = lstm_module.GROUP_COLUMN
TIME_COLUMN = lstm_module.TIME_COLUMN

MODEL_PATH = ROOT / "results/lstm_one_stage/lstm_one_stage.pth"
DATA_PATH = ROOT / "Data/processed/featured_data.parquet"
OUTPUT_DIR = ROOT / "results/lstm_one_stage"


def load_model_and_scalers(checkpoint_path: Path):
    """Load the trained model and scalers from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    
    # Reconstruct scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(checkpoint["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    scaler.n_samples_seen_ = 1
    
    # Create model
    model = HierarchicalWinPredictor(
        hidden_dim=config.get("hidden_dim", 128),
        proj_dim=config.get("proj_dim", 128),
        team_embed_dim=config.get("team_embed_dim", 32),
        condition_on_hidden=config.get("condition_on_hidden", True),
        recurrent_dropout=config.get("recurrent_dropout", 0.2),
        use_decoder=config.get("use_decoder", True),
        decoder_steps=config.get("decoder_steps", 5),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, scaler, device, config


def evaluate_model(model, dataset, device, batch_size=128):
    """Evaluate model and return predictions and targets."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    all_probs = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
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


def main():
    print("=" * 60)
    print("LSTM ONE STAGE MODEL EVALUATION")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading trained model...")
    model, scaler, device, config = load_model_and_scalers(MODEL_PATH)
    print(f"✓ Model loaded on device: {device}")
    print(f"  Config: hidden_dim={config.get('hidden_dim')}, proj_dim={config.get('proj_dim')}")
    
    # Load data and create datasets
    print("\n[2/4] Loading data and creating datasets...")
    df = pd.read_parquet(DATA_PATH)
    match_ids = df[GROUP_COLUMN].unique()
    
    # Use same split as training (80/20)
    from sklearn.model_selection import train_test_split
    train_ids, test_ids = train_test_split(
        match_ids,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
    
    train_match_set = set(train_ids)
    test_match_set = set(test_ids)
    
    # Create test dataset
    test_dataset = WinPredictionDataset(
        parquet_path=DATA_PATH,
        encoder_length=ENCODER_LENGTH,
        scaler=scaler,
        fit_scaler=False,
        match_filter=test_match_set,
    )
    
    print(f"✓ Test dataset created: {len(test_dataset)} samples from {len(test_match_set)} matches")
    
    # Evaluate
    print("\n[3/4] Evaluating model on test set...")
    probs, preds, targets = evaluate_model(model, test_dataset, device)
    
    # Compute metrics
    print("\n[4/4] Computing metrics...")
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    print(classification_report(targets, preds, target_names=["Loss", "Win"]))
    
    print("\n" + "-" * 60)
    print("Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(targets, preds)
    print(f"                Predicted")
    print(f"              Loss    Win")
    print(f"Actual Loss   {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Win    {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # Save results
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "test_samples": len(test_dataset),
        "test_matches": len(test_match_set),
    }
    
    import json
    output_path = OUTPUT_DIR / "detailed_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Detailed metrics saved to: {output_path}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

