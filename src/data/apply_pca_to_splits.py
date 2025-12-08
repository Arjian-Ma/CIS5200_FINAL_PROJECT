#!/usr/bin/env python3
"""
Apply saved PCA model to train/val/test splits.
Transforms the existing splits using a pre-trained PCA model.

python src/data/apply_pca_to_splits.py --pca_model data/processed/pca/pca_model_161components.pkl
"""

import pandas as pd
import numpy as np
import argparse
import os
import pickle
from pathlib import Path
from typing import List, Tuple


def load_pca_model(model_path: str) -> Tuple:
    """Load the saved PCA model and scaler."""
    print(f"📂 Loading PCA model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PCA model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    pca = model_data['pca']
    scaler = model_data['scaler']
    n_components = model_data.get('n_components', pca.n_components_)
    
    print(f"✓ Loaded PCA model with {n_components} components")
    print(f"   Explained variance: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    return pca, scaler, n_components


def prepare_features_for_pca(
    df: pd.DataFrame,
    exclude_outcome_vars: bool = True
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Prepare features for PCA by excluding metadata and outcome variables.
    Must match the preprocessing used when training the PCA model.
    
    Args:
        df: Input dataframe
        exclude_outcome_vars: Whether to exclude outcome variables from PCA (but preserve in output)
    
    Returns:
        Tuple of (feature_df, metadata_column_names, outcome_column_names)
    """
    # Columns to always exclude (metadata)
    metadata_cols = {"match_id", "frame_idx", "timestamp", "puuid", "team"}
    
    # Outcome variables (exclude from PCA but preserve in output)
    outcome_cols = set()
    if exclude_outcome_vars:
        outcome_cols = {
            "Y_won",
            # "Elite_Monster_Killed_Difference",
            # "Buildings_Taken_Difference",
            # "Total_Gold_Difference",
            # "Total_Xp_Difference",
        }
    
    # Columns to exclude from PCA features
    exclude_from_pca = metadata_cols | outcome_cols
    
    # Get feature columns (for PCA)
    feature_cols = [col for col in df.columns if col not in exclude_from_pca]
    
    # Get metadata columns that exist (for preserving in output)
    metadata_present = [col for col in metadata_cols if col in df.columns]
    
    # Get outcome columns that exist (for preserving in output)
    outcome_present = [col for col in outcome_cols if col in df.columns]
    
    # Check for non-numeric columns in features
    non_numeric = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    
    if non_numeric:
        print(f"⚠ Warning: {len(non_numeric)} non-numeric feature columns found (will be excluded)")
        feature_cols = [col for col in feature_cols if col not in non_numeric]
    
    # Extract feature data
    feature_df = df[feature_cols].copy()
    
    # Check for missing values
    missing_counts = feature_df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"⚠ Warning: Found missing values in {missing_counts[missing_counts.sum() > 0].shape[0]} columns")
        print(f"   Filling missing values with column means...")
        feature_df = feature_df.fillna(feature_df.mean())
    
    return feature_df, metadata_present, outcome_present


def apply_pca_to_dataframe(
    df: pd.DataFrame,
    pca,
    scaler,
    exclude_outcome_vars: bool = True
) -> pd.DataFrame:
    """
    Apply PCA transformation to a dataframe.
    
    Args:
        df: Input dataframe
        pca: Fitted PCA model
        scaler: Fitted StandardScaler
        exclude_outcome_vars: Whether to exclude outcome variables from PCA (but preserve in output)
    
    Returns:
        Transformed dataframe with PCA components + metadata + outcome variables
    """
    # Prepare features
    feature_df, metadata_cols, outcome_cols = prepare_features_for_pca(df, exclude_outcome_vars=exclude_outcome_vars)
    
    # Standardize features (using the same scaler from training)
    X_scaled = scaler.transform(feature_df)
    
    # Apply PCA transformation
    X_pca = pca.transform(X_scaled)
    
    # Create dataframe with component names
    n_components = pca.n_components_
    component_names = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=component_names, index=df.index)
    
    # Combine: metadata + outcome variables + PCA components
    result_parts = []
    
    # Add metadata columns
    if metadata_cols:
        result_parts.append(df[metadata_cols].copy())
    
    # Add outcome variables (preserve them!)
    if outcome_cols:
        result_parts.append(df[outcome_cols].copy())
    
    # Add PCA components
    result_parts.append(pca_df)
    
    # Concatenate all parts
    if result_parts:
        result_df = pd.concat(result_parts, axis=1)
    else:
        result_df = pca_df
    
    return result_df


def process_split(
    input_path: str,
    output_path: str,
    pca,
    scaler,
    split_name: str,
    exclude_outcome_vars: bool = True
):
    """Process a single split (train/val/test)."""
    print(f"\n{'='*70}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*70}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    # Load data
    if not os.path.exists(input_path):
        print(f"⚠ Warning: {input_path} not found, skipping...")
        return False
    
    print(f"📂 Loading {split_name} data...")
    df = pd.read_parquet(input_path)
    print(f"✓ Loaded: {len(df):,} rows × {df.shape[1]} columns")
    
    # Apply PCA transformation
    print(f"🔧 Applying PCA transformation...")
    transformed_df = apply_pca_to_dataframe(df, pca, scaler, exclude_outcome_vars=exclude_outcome_vars)
    
    # Save transformed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transformed_df.to_parquet(output_path, index=False)
    print(f"✓ Saved transformed {split_name} data: {output_path}")
    print(f"   Shape: {transformed_df.shape[0]:,} rows × {transformed_df.shape[1]} columns")
    print(f"   Components: {pca.n_components_}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Apply saved PCA model to train/val/test splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply PCA to all splits (default paths)
  python src/data/apply_pca_to_splits.py --pca_model data/processed/pca/pca_model_58components.pkl

  # Custom input/output directories
  python src/data/apply_pca_to_splits.py \\
    --pca_model data/processed/pca/pca_model_58components.pkl \\
    --input_dir data/splits \\
    --output_dir data/splits/pca

  # Process only specific splits
  python src/data/apply_pca_to_splits.py \\
    --pca_model data/processed/pca/pca_model_58components.pkl \\
    --splits train test
        """
    )
    
    parser.add_argument(
        '--pca_model',
        type=str,
        required=True,
        help='Path to saved PCA model pickle file (e.g., data/processed/pca/pca_model_58components.pkl)'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/splits',
        help='Directory containing train.parquet, val.parquet, test.parquet (default: data/splits)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/splits/pca',
        help='Output directory for PCA-transformed splits (default: data/splits/pca)'
    )
    
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val', 'test'],
        choices=['train', 'val', 'test'],
        help='Which splits to process (default: train val test)'
    )
    
    parser.add_argument(
        '--include_outcomes',
        action='store_true',
        help='Include outcome variables in PCA (should match PCA model training)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("PCA Transformation for Data Splits")
    print("="*70)
    print(f"PCA Model:      {args.pca_model}")
    print(f"Input Dir:      {args.input_dir}")
    print(f"Output Dir:     {args.output_dir}")
    print(f"Splits:         {', '.join(args.splits)}")
    print(f"Exclude Outcomes: {not args.include_outcomes}")
    print("="*70)
    
    # Load PCA model
    pca, scaler, n_components = load_pca_model(args.pca_model)
    
    # Process each split
    results = {}
    for split_name in args.splits:
        input_path = os.path.join(args.input_dir, f"{split_name}.parquet")
        output_path = os.path.join(args.output_dir, f"{split_name}_pca_{n_components}components.parquet")
        
        success = process_split(
            input_path,
            output_path,
            pca,
            scaler,
            split_name,
            exclude_outcome_vars=not args.include_outcomes
        )
        results[split_name] = success
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    for split_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {split_name.upper()}: {'Completed' if success else 'Skipped'}")
    
    successful = sum(results.values())
    total = len(results)
    print(f"\nProcessed {successful}/{total} splits successfully")
    
    if successful == total:
        print("\n✅ All splits processed successfully!")
        print(f"\nOutput files in: {args.output_dir}")
        for split_name in args.splits:
            if results[split_name]:
                output_path = os.path.join(args.output_dir, f"{split_name}_pca_{n_components}components.parquet")
                print(f"  - {output_path}")
    else:
        print("\n⚠ Warning: Some splits were not processed. Check the warnings above.")
    
    print()


if __name__ == '__main__':
    main()

