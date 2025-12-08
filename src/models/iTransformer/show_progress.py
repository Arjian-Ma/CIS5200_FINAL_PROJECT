"""
Script to show training progress and results for iTransformer models.
Displays current training status, completed models, and final metrics.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

OUTPUT_DIR = ROOT / "results/itransformer_one_stage_multiple"
ENCODER_LENGTHS = [15, 20, 25, 30]


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def show_progress():
    """Display training progress and results."""
    print("=" * 70)
    print("iTRANSFORMER TRAINING PROGRESS")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not OUTPUT_DIR.exists():
        print("❌ Output directory does not exist. Training has not started yet.")
        return
    
    # Check each model
    completed = []
    in_progress = []
    not_started = []
    
    for encoder_length in ENCODER_LENGTHS:
        model_dir = OUTPUT_DIR / f"model_{encoder_length}min"
        metrics_path = model_dir / "final_metrics.json"
        history_path = model_dir / "training_history.json"
        model_path = model_dir / "model.pth"
        
        if model_path.exists() and metrics_path.exists():
            # Model is completed
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                completed.append((encoder_length, metrics))
            except Exception as e:
                print(f"⚠️  Warning: Could not read metrics for {encoder_length}min: {e}")
                in_progress.append(encoder_length)
        elif history_path.exists():
            # Training in progress (history exists but model not saved yet)
            in_progress.append(encoder_length)
        else:
            # Not started
            not_started.append(encoder_length)
    
    # Display status
    print(f"📊 Status Summary:")
    print(f"   ✅ Completed: {len(completed)}/{len(ENCODER_LENGTHS)}")
    print(f"   🔄 In Progress: {len(in_progress)}/{len(ENCODER_LENGTHS)}")
    print(f"   ⏳ Not Started: {len(not_started)}/{len(ENCODER_LENGTHS)}")
    print()
    
    # Show completed models
    if completed:
        print("=" * 70)
        print("✅ COMPLETED MODELS")
        print("=" * 70)
        print(f"{'Model':<10} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Test Prec':<12} {'Test Rec':<12} {'Test F1':<12}")
        print("-" * 90)
        for encoder_length, metrics in sorted(completed, key=lambda x: x[0]):
            print(
                f"{encoder_length}min{'':<6} "
                f"{metrics.get('train_accuracy', 0):<12.4f} "
                f"{metrics.get('val_accuracy', 0):<12.4f} "
                f"{metrics.get('test_accuracy', 0):<12.4f} "
                f"{metrics.get('test_precision', 0):<12.4f} "
                f"{metrics.get('test_recall', 0):<12.4f} "
                f"{metrics.get('test_f1_score', 0):<12.4f}"
            )
        print()
    
    # Show in-progress models
    if in_progress:
        print("=" * 70)
        print("🔄 MODELS IN PROGRESS")
        print("=" * 70)
        for encoder_length in in_progress:
            model_dir = OUTPUT_DIR / f"model_{encoder_length}min"
            history_path = model_dir / "training_history.json"
            
            if history_path.exists():
                try:
                    with open(history_path, "r") as f:
                        history = json.load(f)
                    
                    epochs_trained = len(history.get("train_loss", []))
                    if epochs_trained > 0:
                        latest_train_acc = history["train_acc"][-1] if history.get("train_acc") else 0
                        latest_val_acc = history["val_acc"][-1] if history.get("val_acc") else 0
                        latest_test_acc = history["test_acc"][-1] if history.get("test_acc") else 0
                        
                        print(f"  {encoder_length}min model:")
                        print(f"    Epochs trained: {epochs_trained}/15")
                        print(f"    Latest Train Acc: {latest_train_acc:.4f}")
                        print(f"    Latest Val Acc: {latest_val_acc:.4f}")
                        print(f"    Latest Test Acc: {latest_test_acc:.4f}")
                        if history.get("early_stopped_epoch"):
                            print(f"    ⚠️  Early stopped at epoch {history['early_stopped_epoch']}")
                        print()
                except Exception as e:
                    print(f"  {encoder_length}min: Error reading history - {e}")
        print()
    
    # Show not started models
    if not_started:
        print("=" * 70)
        print("⏳ MODELS NOT STARTED")
        print("=" * 70)
        for encoder_length in not_started:
            print(f"  {encoder_length}min model: Waiting to start")
        print()
    
    # Show summary if available
    summary_path = OUTPUT_DIR / "summary_metrics.json"
    if summary_path.exists():
        print("=" * 70)
        print("📋 SUMMARY METRICS")
        print("=" * 70)
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            
            print(f"{'Model':<10} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Test Prec':<12} {'Test Rec':<12} {'Test F1':<12}")
            print("-" * 90)
            for model_name in sorted(summary.keys()):
                metrics = summary[model_name]
                print(
                    f"{model_name:<10} "
                    f"{metrics.get('train_accuracy', 0):<12.4f} "
                    f"{metrics.get('val_accuracy', 0):<12.4f} "
                    f"{metrics.get('test_accuracy', 0):<12.4f} "
                    f"{metrics.get('test_precision', 0):<12.4f} "
                    f"{metrics.get('test_recall', 0):<12.4f} "
                    f"{metrics.get('test_f1_score', 0):<12.4f}"
                )
        except Exception as e:
            print(f"Error reading summary: {e}")
        print()
    
    print("=" * 70)
    print("💡 Tip: Run this script periodically to check training progress")
    print("=" * 70)


if __name__ == "__main__":
    show_progress()

