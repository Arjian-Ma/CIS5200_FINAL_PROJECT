#!/usr/bin/env python3
"""
Check for data leakage between train/val/test splits.

Data leakage occurs when the same match_id appears in multiple splits.
This script checks for overlapping match_ids between splits.
"""

import argparse
import os
import pandas as pd
from typing import Set, Tuple


def get_unique_match_ids(file_path: str) -> Set[str]:
    """Load parquet file and return unique match_ids."""
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {file_path}")
        return set()
    
    try:
        df = pd.read_parquet(file_path)
        if 'match_id' not in df.columns:
            print(f"⚠ No 'match_id' column in: {file_path}")
            return set()
        match_ids = set(df['match_id'].unique())
        print(f"  {file_path}: {len(match_ids)} unique matches, {len(df)} rows")
        return match_ids
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return set()


def check_leakage(
    train_path: str,
    val_path: str,
    test_path: str,
    split_name: str = "split"
) -> Tuple[bool, dict]:
    """
    Check for data leakage between train/val/test splits.
    
    Returns:
        (has_leakage, leakage_info)
    """
    print(f"\n{'='*60}")
    print(f"Checking {split_name} for data leakage...")
    print(f"{'='*60}")
    
    train_ids = get_unique_match_ids(train_path)
    val_ids = get_unique_match_ids(val_path)
    test_ids = get_unique_match_ids(test_path)
    
    leakage_info = {
        'train_val_overlap': train_ids & val_ids,
        'train_test_overlap': train_ids & test_ids,
        'val_test_overlap': val_ids & test_ids,
        'train_count': len(train_ids),
        'val_count': len(val_ids),
        'test_count': len(test_ids),
    }
    
    has_leakage = (
        len(leakage_info['train_val_overlap']) > 0 or
        len(leakage_info['train_test_overlap']) > 0 or
        len(leakage_info['val_test_overlap']) > 0
    )
    
    print(f"\n📊 Summary for {split_name}:")
    print(f"  Train matches: {len(train_ids):,}")
    print(f"  Val matches: {len(val_ids):,}")
    print(f"  Test matches: {len(test_ids):,}")
    print(f"  Total unique matches: {len(train_ids | val_ids | test_ids):,}")
    
    if has_leakage:
        print(f"\n❌ DATA LEAKAGE DETECTED in {split_name}!")
        if leakage_info['train_val_overlap']:
            print(f"  ⚠ Train-Val overlap: {len(leakage_info['train_val_overlap'])} matches")
            print(f"     Example matches: {list(leakage_info['train_val_overlap'])[:5]}")
        if leakage_info['train_test_overlap']:
            print(f"  ⚠ Train-Test overlap: {len(leakage_info['train_test_overlap'])} matches")
            print(f"     Example matches: {list(leakage_info['train_test_overlap'])[:5]}")
        if leakage_info['val_test_overlap']:
            print(f"  ⚠ Val-Test overlap: {len(leakage_info['val_test_overlap'])} matches")
            print(f"     Example matches: {list(leakage_info['val_test_overlap'])[:5]}")
    else:
        print(f"\n✅ No data leakage detected in {split_name}!")
        print(f"  All splits have unique match_ids ✓")
    
    return has_leakage, leakage_info


def main():
    parser = argparse.ArgumentParser(
        description="Check for data leakage between train/val/test splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="data/splits",
        help="Directory containing split files"
    )
    
    args = parser.parse_args()
    
    splits_dir = args.splits_dir
    
    print("="*60)
    print("Data Leakage Checker")
    print("="*60)
    print(f"Checking splits in: {splits_dir}")
    
    # Check standard splits
    standard_train = os.path.join(splits_dir, "train.parquet")
    standard_val = os.path.join(splits_dir, "val.parquet")
    standard_test = os.path.join(splits_dir, "test.parquet")
    
    # Check reduced splits
    reduced_train = os.path.join(splits_dir, "reduced_train.parquet")
    reduced_val = os.path.join(splits_dir, "reduced_val.parquet")
    reduced_test = os.path.join(splits_dir, "reduced_test.parquet")
    
    results = {}
    
    # Check standard splits
    if all(os.path.exists(p) for p in [standard_train, standard_val, standard_test]):
        has_leakage, info = check_leakage(
            standard_train, standard_val, standard_test, "Standard splits"
        )
        results['standard'] = {'has_leakage': has_leakage, 'info': info}
    else:
        print(f"\n⚠ Standard splits not all found, skipping...")
        results['standard'] = None
    
    # Check reduced splits
    if all(os.path.exists(p) for p in [reduced_train, reduced_val, reduced_test]):
        has_leakage, info = check_leakage(
            reduced_train, reduced_val, reduced_test, "Reduced splits"
        )
        results['reduced'] = {'has_leakage': has_leakage, 'info': info}
    else:
        print(f"\n⚠ Reduced splits not all found, skipping...")
        results['reduced'] = None
    
    # Final summary
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")
    
    if results['standard']:
        status = "❌ LEAKAGE" if results['standard']['has_leakage'] else "✅ CLEAN"
        print(f"Standard splits: {status}")
    
    if results['reduced']:
        status = "❌ LEAKAGE" if results['reduced']['has_leakage'] else "✅ CLEAN"
        print(f"Reduced splits: {status}")
    
    # Overall status
    all_clean = all(
        r is None or not r['has_leakage']
        for r in results.values()
    )
    
    if all_clean:
        print(f"\n✅ All checked splits are clean - no data leakage!")
    else:
        print(f"\n❌ Data leakage detected in at least one split set!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

