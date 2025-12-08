#!/usr/bin/env python3
"""
Diagnostic script to check what features are being used in the classifier
and verify there's no data leakage (e.g., Y_won being used as a feature).
"""

import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_win_classifier import get_specified_features


def check_features(data_path: str, feature_list_source: str = "specified"):
    """Check what features are being used and verify Y_won is not included."""
    print("=" * 60)
    print(f"Diagnosing features for: {data_path}")
    print("=" * 60)
    
    # Load data
    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        return
    
    df = pd.read_parquet(data_path)
    print(f"\n📊 Data Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Unique matches: {df['match_id'].nunique()}")
    
    # Get feature list
    if feature_list_source == "specified":
        feature_list = get_specified_features()
    else:
        feature_list = get_specified_features(feature_list_source)
    
    print(f"\n📋 Feature List ({len(feature_list)} features):")
    for i, feat in enumerate(feature_list, 1):
        print(f"  {i:2d}. {feat}")
    
    # Check for Y_won in feature list
    if "Y_won" in feature_list:
        print(f"\n❌ CRITICAL: Y_won is in the feature list! This is data leakage!")
        return
    else:
        print(f"\n✅ Y_won is NOT in the feature list")
    
    # Check which features are available in data
    available = set(df.columns)
    feature_cols = [feat for feat in feature_list if feat in available and feat != "Y_won"]
    missing = [feat for feat in feature_list if feat not in available]
    
    print(f"\n🔍 Feature Availability:")
    print(f"  Available in data: {len(feature_cols)}/{len(feature_list)}")
    if missing:
        print(f"  Missing from data: {len(missing)}")
        print(f"    {missing[:10]}{'...' if len(missing) > 10 else ''}")
    
    # Check if Y_won exists in data
    if "Y_won" in df.columns:
        print(f"\n✅ Y_won exists in data (as target, not feature)")
        print(f"  Y_won distribution:")
        print(f"    {df['Y_won'].value_counts().to_dict()}")
    else:
        print(f"\n⚠ Y_won not found in data columns")
    
    # Check for any suspicious columns that might leak information
    suspicious_cols = [col for col in df.columns if 'won' in col.lower() or 'win' in col.lower()]
    if suspicious_cols:
        print(f"\n⚠ Suspicious columns found (might contain win information):")
        for col in suspicious_cols:
            print(f"    - {col}")
    
    # Check correlation of features with Y_won (if it exists)
    if "Y_won" in df.columns and len(feature_cols) > 0:
        print(f"\n📈 Feature-Target Correlations (top 10):")
        correlations = df[feature_cols + ["Y_won"]].corr()["Y_won"].abs().sort_values(ascending=False)
        correlations = correlations[correlations.index != "Y_won"]
        for feat, corr in correlations.head(10).items():
            print(f"  {feat}: {corr:.4f}")
    
    # Final check: verify feature_cols doesn't contain Y_won
    if "Y_won" in feature_cols:
        print(f"\n❌ CRITICAL ERROR: Y_won is in feature_cols after filtering!")
        print(f"  This should never happen - there's a bug in the code!")
    else:
        print(f"\n✅ Final verification: Y_won is NOT in feature_cols")
        print(f"  Will use {len(feature_cols)} features for training")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose classifier features for data leakage")
    parser.add_argument("--data_path", type=str, default="data/splits/reduced_train.parquet",
                        help="Path to data file to check")
    parser.add_argument("--feature_list", type=str, default="specified",
                        help="Feature list source: 'specified', CSV path, or 'None'")
    
    args = parser.parse_args()
    
    check_features(args.data_path, args.feature_list)


if __name__ == "__main__":
    main()

