#!/usr/bin/env python3
"""
Create time-interval datasets (test/train/val) based on number of timestamps.

For each time interval (10, 15, 20, 25, 30 minutes), creates a dataset
containing only games that have at least that many timestamps/frames.

The dataset type (test/train/val) is auto-detected from the input filename
or can be specified manually.

Usage:
    python create_time_interval_test_sets.py --input data/splits/test.parquet --output_dir data/splits
    python create_time_interval_test_sets.py --input data/splits/train.parquet --output_dir data/splits
    python create_time_interval_test_sets.py --input data/splits/val.parquet --dataset_type val --output_dir data/splits
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm


def count_timestamps_per_game(df: pd.DataFrame) -> pd.Series:
    """
    Count the number of unique timestamps/frames per game.
    
    Args:
        df: DataFrame with 'match_id' and 'frame_idx' or 'timestamp' columns
        
    Returns:
        Series with match_id as index and timestamp count as values
    """
    # Use frame_idx if available, otherwise use timestamp
    if 'frame_idx' in df.columns:
        count_col = 'frame_idx'
    elif 'timestamp' in df.columns:
        count_col = 'timestamp'
    else:
        raise ValueError("DataFrame must have either 'frame_idx' or 'timestamp' column")
    
    # Count unique frames/timestamps per match
    timestamp_counts = df.groupby('match_id')[count_col].nunique()
    return timestamp_counts


def detect_dataset_type(input_path: str) -> str:
    """
    Detect dataset type (test/train/val) from input filename.
    
    Args:
        input_path: Path to input file
        
    Returns:
        Dataset type string: 'test', 'train', 'val', or 'unknown'
    """
    filename = os.path.basename(input_path).lower()
    
    if 'test' in filename:
        return 'test'
    elif 'train' in filename:
        return 'train'
    elif 'val' in filename or 'validation' in filename:
        return 'val'
    else:
        return 'unknown'


def create_time_interval_dataset(
    df: pd.DataFrame,
    min_timestamps: int,
    output_path: str,
    dataset_type: str = "unknown"
) -> pd.DataFrame:
    """
    Create a dataset containing only games with at least min_timestamps timestamps.
    
    Args:
        df: Full DataFrame
        min_timestamps: Minimum number of timestamps required
        output_path: Path to save the filtered dataset
        dataset_type: Type of dataset (test/train/val) for logging
        
    Returns:
        Filtered DataFrame
    """
    print(f"\n=== Creating {dataset_type} set for {min_timestamps} timestamps ===")
    
    # Count timestamps per game
    timestamp_counts = count_timestamps_per_game(df)
    
    # Filter games with at least min_timestamps
    qualifying_matches = timestamp_counts[timestamp_counts >= min_timestamps].index
    
    print(f"  Total games in {dataset_type} set: {df['match_id'].nunique()}")
    print(f"  Games with >= {min_timestamps} timestamps: {len(qualifying_matches)}")
    print(f"  Games filtered out: {df['match_id'].nunique() - len(qualifying_matches)}")
    
    # Filter DataFrame
    filtered_df = df[df['match_id'].isin(qualifying_matches)].copy()
    
    print(f"  Total rows in filtered set: {len(filtered_df):,}")
    print(f"  Rows filtered out: {len(df) - len(filtered_df):,}")
    
    # Save to parquet
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    filtered_df.to_parquet(output_path, index=False)
    print(f"  ✅ Saved to: {output_path}")
    
    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description="Create time-interval datasets (test/train/val) at different time intervals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/splits/test.parquet",
        help="Path to input parquet file (test/train/val)"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        choices=["test", "train", "val"],
        help="Dataset type (test/train/val). Auto-detected from filename if not specified."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits",
        help="Directory to save output datasets"
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default="10,15,20,25,30",
        help="Comma-separated list of time intervals (number of timestamps)"
    )
    
    args = parser.parse_args()
    
    # Detect dataset type
    if args.dataset_type:
        dataset_type = args.dataset_type
    else:
        dataset_type = detect_dataset_type(args.input)
        if dataset_type == "unknown":
            print(f"⚠ Warning: Could not detect dataset type from filename: {args.input}")
            print("   Using 'unknown' as dataset type. Consider using --dataset_type to specify.")
    
    # Parse intervals
    intervals = [int(x.strip()) for x in args.intervals.split(",") if x.strip()]
    intervals = sorted(intervals)  # Sort from smallest to largest
    
    print("=" * 60)
    print("Creating Time Interval Datasets")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Dataset type: {dataset_type}")
    print(f"Output directory: {args.output_dir}")
    print(f"Time intervals: {intervals}")
    print()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return
    
    # Load data
    print(f"Loading {dataset_type} data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"✅ Loaded {len(df):,} rows from {df['match_id'].nunique()} games")
    
    # Show timestamp distribution
    timestamp_counts = count_timestamps_per_game(df)
    print(f"\nTimestamp distribution:")
    print(f"  Min: {timestamp_counts.min()}")
    print(f"  Max: {timestamp_counts.max()}")
    print(f"  Mean: {timestamp_counts.mean():.1f}")
    print(f"  Median: {timestamp_counts.median():.1f}")
    print(f"  Games with >= 10 timestamps: {(timestamp_counts >= 10).sum()}")
    print(f"  Games with >= 15 timestamps: {(timestamp_counts >= 15).sum()}")
    print(f"  Games with >= 20 timestamps: {(timestamp_counts >= 20).sum()}")
    print(f"  Games with >= 25 timestamps: {(timestamp_counts >= 25).sum()}")
    print(f"  Games with >= 30 timestamps: {(timestamp_counts >= 30).sum()}")
    
    # Create datasets for each interval
    results = {}
    for min_ts in intervals:
        output_path = os.path.join(args.output_dir, f"{dataset_type}_{min_ts}min.parquet")
        filtered_df = create_time_interval_dataset(df, min_ts, output_path, dataset_type)
        results[min_ts] = {
            'path': output_path,
            'num_games': filtered_df['match_id'].nunique(),
            'num_rows': len(filtered_df)
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Interval':<12} {'Games':<10} {'Rows':<15} {'Output File'}")
    print("-" * 60)
    for min_ts in intervals:
        info = results[min_ts]
        print(f"{min_ts} min{'':<6} {info['num_games']:<10} {info['num_rows']:<15,} {info['path']}")
    
    print(f"\n✅ All {dataset_type} datasets created successfully!")


if __name__ == "__main__":
    main()

