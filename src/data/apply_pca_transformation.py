#!/usr/bin/env python3
"""
Apply PCA transformation to featured_data.parquet
Creates a PCA-transformed dataset with configurable number of components
and generates visualizations of the PCA results.
"""

import pandas as pd
import numpy as np
import argparse
import os
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(data_path: str) -> pd.DataFrame:
    """Load the featured data parquet file."""
    print(f"📂 Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def prepare_features(df: pd.DataFrame, exclude_outcome_vars: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for PCA by excluding metadata and outcome variables.
    
    Args:
        df: Input dataframe
        exclude_outcome_vars: Whether to exclude outcome variables (Y_won, etc.)
    
    Returns:
        Tuple of (feature_df, feature_column_names)
    """
    # Columns to always exclude (metadata)
    metadata_cols = {"match_id", "frame_idx", "timestamp", "puuid", "team"}
    
    # Outcome variables to exclude (if requested)
    outcome_cols = set()
    if exclude_outcome_vars:
        outcome_cols = {
            "Y_won",
            # "Elite_Monster_Killed_Difference",
            # "Buildings_Taken_Difference",
            # "Total_Gold_Difference",
            # "Total_Xp_Difference",
        }
    
    # All columns to exclude
    exclude_cols = metadata_cols | outcome_cols
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Get metadata columns that exist (for preserving in output)
    metadata_present = [col for col in metadata_cols if col in df.columns]
    
    print(f"\n📊 Feature Preparation:")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Excluded columns: {len(exclude_cols)} ({', '.join(sorted(exclude_cols))})")
    print(f"   Feature columns: {len(feature_cols)}")
    print(f"   Metadata columns to preserve: {len(metadata_present)}")
    
    # Check for non-numeric columns in features
    non_numeric = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    
    if non_numeric:
        print(f"⚠ Warning: {len(non_numeric)} non-numeric feature columns found (will be excluded):")
        print(f"   {', '.join(non_numeric[:10])}{'...' if len(non_numeric) > 10 else ''}")
        feature_cols = [col for col in feature_cols if col not in non_numeric]
    
    # Extract feature data
    feature_df = df[feature_cols].copy()
    
    # Check for missing values
    missing_counts = feature_df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"⚠ Warning: Found missing values in {missing_counts[missing_counts > 0].shape[0]} columns")
        print(f"   Total missing values: {missing_counts.sum():,}")
        print(f"   Filling missing values with column means...")
        feature_df = feature_df.fillna(feature_df.mean())
    
    return feature_df, feature_cols, metadata_present


def find_optimal_components(
    feature_df: pd.DataFrame,
    variance_threshold: float = 0.95,
    random_state: int = 42
) -> dict:
    """
    Find optimal number of PCA components using multiple criteria.
    
    Args:
        feature_df: Feature dataframe
        variance_threshold: Target cumulative variance (default: 0.95)
        random_state: Random seed
    
    Returns:
        Dictionary with recommendations and analysis
    """
    print(f"\n🔍 Finding optimal number of components...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)
    
    # Fit PCA with all components to analyze
    n_features = feature_df.shape[1]
    pca_full = PCA(n_components=min(n_features, 1000), random_state=random_state)  # Limit to 1000 for efficiency
    pca_full.fit(X_scaled)
    
    explained_var = pca_full.explained_variance_ratio_
    eigenvalues = pca_full.explained_variance_
    cumulative_var = np.cumsum(explained_var)
    
    # Criterion 1: Variance threshold (e.g., 95% variance)
    n_variance = np.argmax(cumulative_var >= variance_threshold) + 1
    if not (cumulative_var >= variance_threshold).any():
        n_variance = len(cumulative_var)
    
    # Criterion 2: Kaiser criterion (eigenvalue > 1)
    n_kaiser = np.sum(eigenvalues > 1.0)
    
    # Criterion 3: Elbow method (find point of diminishing returns)
    # Calculate second derivative to find elbow
    if len(explained_var) > 2:
        # Use percentage change in explained variance
        pct_change = np.diff(explained_var) / explained_var[:-1]
        # Find where the change drops significantly (below 0.1% of previous)
        # Look for first component where change is less than 0.001 (0.1%)
        threshold = 0.001
        elbow_candidates = np.where(pct_change < threshold)[0]
        n_elbow = elbow_candidates[0] + 2 if len(elbow_candidates) > 0 else min(50, len(explained_var))
    else:
        n_elbow = len(explained_var)
    
    # Criterion 4: 80% variance (more conservative)
    n_80 = np.argmax(cumulative_var >= 0.80) + 1 if (cumulative_var >= 0.80).any() else len(cumulative_var)
    
    # Criterion 5: 90% variance
    n_90 = np.argmax(cumulative_var >= 0.90) + 1 if (cumulative_var >= 0.90).any() else len(cumulative_var)
    
    # Recommendation: Use median of common criteria, or variance threshold
    recommendations = {
        'variance_threshold': n_variance,
        'kaiser_criterion': n_kaiser,
        'elbow_method': n_elbow,
        'variance_80': n_80,
        'variance_90': n_90,
        'variance_95': n_variance,
    }
    
    # Recommended value: use variance threshold as primary, but consider others
    recommended = max(n_variance, n_kaiser)  # Take the more conservative estimate
    
    results = {
        'recommended': recommended,
        'recommendations': recommendations,
        'explained_variance': explained_var,
        'eigenvalues': eigenvalues,
        'cumulative_variance': cumulative_var,
        'variance_at_recommended': cumulative_var[recommended - 1] if recommended <= len(cumulative_var) else cumulative_var[-1],
    }
    
    print(f"\n📊 Component Selection Analysis:")
    print(f"   Kaiser Criterion (λ > 1):     {n_kaiser} components")
    print(f"   80% Variance:                  {n_80} components")
    print(f"   90% Variance:                  {n_90} components")
    print(f"   {variance_threshold*100:.0f}% Variance:              {n_variance} components")
    print(f"   Elbow Method:                  {n_elbow} components")
    print(f"\n💡 Recommended: {recommended} components")
    print(f"   (Explains {results['variance_at_recommended']*100:.2f}% of variance)")
    
    return results


def apply_pca(
    feature_df: pd.DataFrame,
    n_components: int,
    random_state: int = 42
) -> Tuple[pd.DataFrame, PCA, StandardScaler]:
    """
    Apply PCA transformation to features.
    
    Args:
        feature_df: Feature dataframe
        n_components: Number of PCA components
        random_state: Random seed
    
    Returns:
        Tuple of (transformed_df, pca_model, scaler)
    """
    print(f"\n🔧 Applying PCA with {n_components} components...")
    
    # Standardize features first (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)
    print(f"✓ Features standardized (mean={X_scaled.mean():.2e}, std={X_scaled.std():.2f})")
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    print(f"✓ PCA transformation complete")
    
    # Create dataframe with component names
    component_names = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=component_names, index=feature_df.index)
    
    # Print explained variance information
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print(f"\n📈 Explained Variance:")
    print(f"   First 5 components: {explained_var[:5]}")
    print(f"   Cumulative variance (first 5): {cumulative_var[:5]}")
    print(f"   Total explained variance: {cumulative_var[-1]:.4f} ({cumulative_var[-1]*100:.2f}%)")
    
    return pca_df, pca, scaler


def create_visualizations(
    pca: PCA,
    output_dir: str,
    n_components: int,
    optimal_analysis: dict = None
):
    """
    Create visualizations of PCA results.
    
    Args:
        pca: Fitted PCA model
        output_dir: Directory to save plots
        n_components: Number of components
    """
    print(f"\n📊 Creating visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Explained Variance Ratio (Bar Plot)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Plot 1: Individual Explained Variance (first 20 components)
    n_show = min(20, n_components)
    axes[0, 0].bar(range(1, n_show + 1), explained_var[:n_show], alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('Principal Component', fontsize=12)
    axes[0, 0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0, 0].set_title(f'Explained Variance by Component (First {n_show})', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(1, n_show + 1, max(1, n_show // 10)))
    
    # Plot 2: Cumulative Explained Variance
    axes[0, 1].plot(range(1, n_components + 1), cumulative_var, marker='o', linewidth=2, markersize=4, color='coral')
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance', alpha=0.7)
    axes[0, 1].axhline(y=0.90, color='orange', linestyle='--', label='90% Variance', alpha=0.7)
    axes[0, 1].axhline(y=0.80, color='yellow', linestyle='--', label='80% Variance', alpha=0.7)
    axes[0, 1].set_xlabel('Number of Components', fontsize=12)
    axes[0, 1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[0, 1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # Find components needed for 80%, 90%, 95% variance
    n_80 = np.argmax(cumulative_var >= 0.80) + 1 if (cumulative_var >= 0.80).any() else n_components
    n_90 = np.argmax(cumulative_var >= 0.90) + 1 if (cumulative_var >= 0.90).any() else n_components
    n_95 = np.argmax(cumulative_var >= 0.95) + 1 if (cumulative_var >= 0.95).any() else n_components
    
    # Add text annotations
    axes[0, 1].text(n_80, 0.80, f'  {n_80} comps', fontsize=10, verticalalignment='bottom')
    axes[0, 1].text(n_90, 0.90, f'  {n_90} comps', fontsize=10, verticalalignment='bottom')
    if n_95 <= n_components:
        axes[0, 1].text(n_95, 0.95, f'  {n_95} comps', fontsize=10, verticalalignment='bottom')
    
    # Plot 3: Scree Plot (Eigenvalues)
    eigenvalues = pca.explained_variance_
    n_show_eigen = min(30, n_components)
    axes[1, 0].plot(range(1, n_show_eigen + 1), eigenvalues[:n_show_eigen], 
                    marker='o', linewidth=2, markersize=4, color='green')
    axes[1, 0].axhline(y=1.0, color='r', linestyle='--', label='Kaiser Criterion (λ=1)', alpha=0.7)
    axes[1, 0].set_xlabel('Component Number', fontsize=12)
    axes[1, 0].set_ylabel('Eigenvalue', fontsize=12)
    axes[1, 0].set_title(f'Scree Plot (Eigenvalues, First {n_show_eigen})', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Count components with eigenvalue > 1 (Kaiser criterion)
    n_kaiser = np.sum(eigenvalues > 1.0)
    axes[1, 0].text(0.02, 0.98, f'Components with λ>1: {n_kaiser}', 
                    transform=axes[1, 0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: Explained Variance Summary Table
    axes[1, 1].axis('off')
    
    # Create summary table
    summary_data = []
    milestones = [0.50, 0.75, 0.80, 0.90, 0.95, 0.99]
    for milestone in milestones:
        n_comp = np.argmax(cumulative_var >= milestone) + 1 if (cumulative_var >= milestone).any() else n_components
        var_at_milestone = cumulative_var[n_comp - 1] if n_comp <= n_components else cumulative_var[-1]
        summary_data.append([
            f"{milestone*100:.0f}%",
            f"{n_comp}",
            f"{var_at_milestone*100:.2f}%"
        ])
    
    table = axes[1, 1].table(
        cellText=summary_data,
        colLabels=['Target Variance', 'Components Needed', 'Actual Variance'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    axes[1, 1].set_title('Variance Explained Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'pca_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PCA analysis plot: {plot_path}")
    plt.close()
    
    # 2. Detailed Explained Variance Bar Chart
    if n_components >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot first 10 components' explained variance as a more detailed bar chart
        n_show_detailed = min(10, n_components)
        bars = ax.bar(range(1, n_show_detailed + 1), explained_var[:n_show_detailed], 
                     alpha=0.8, color=plt.cm.viridis(np.linspace(0, 1, n_show_detailed)))
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax.set_title(f'Detailed Explained Variance (First {n_show_detailed} Components)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, var) in enumerate(zip(bars, explained_var[:n_show_detailed])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{var:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        detailed_plot_path = os.path.join(output_dir, 'pca_explained_variance_detailed.png')
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved detailed variance plot: {detailed_plot_path}")
        plt.close()


def save_results(
    pca_df: pd.DataFrame,
    original_df: pd.DataFrame,
    metadata_cols: List[str],
    pca: PCA,
    scaler: StandardScaler,
    output_path: str,
    model_output_path: str
):
    """
    Save PCA-transformed dataset and models.
    
    Args:
        pca_df: PCA-transformed feature dataframe
        original_df: Original dataframe (for metadata)
        metadata_cols: Metadata columns to preserve
        pca: Fitted PCA model
        scaler: Fitted scaler
        output_path: Path to save transformed dataset
        model_output_path: Path to save PCA model and scaler
    """
    print(f"\n💾 Saving results...")
    
    # Combine PCA components with metadata
    result_df = original_df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=original_df.index)
    result_df = pd.concat([result_df, pca_df], axis=1)
    
    # Save transformed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    print(f"✓ Saved PCA-transformed dataset: {output_path}")
    print(f"   Shape: {result_df.shape[0]:,} rows × {result_df.shape[1]} columns")
    print(f"   Components: {len(pca_df.columns)}")
    print(f"   Metadata columns: {len(metadata_cols)}")
    
    # Save PCA model and scaler
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump({
            'pca': pca,
            'scaler': scaler,
            'n_components': pca.n_components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        }, f)
    print(f"✓ Saved PCA model and scaler: {model_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Apply PCA transformation to featured_data.parquet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-determine optimal number of components (RECOMMENDED)
  python src/data/apply_pca_transformation.py --auto_components

  # Auto-determine with custom variance threshold (default: 0.95)
  python src/data/apply_pca_transformation.py --auto_components --variance_threshold 0.90

  # Manually specify number of components
  python src/data/apply_pca_transformation.py --n_components 50

  # Custom output directory
  python src/data/apply_pca_transformation.py --auto_components --output_dir data/processed/pca_auto

  # Include outcome variables in PCA (not recommended for prediction tasks)
  python src/data/apply_pca_transformation.py --auto_components --include_outcomes
        """
    )
    
    parser.add_argument(
        '--input_path',
        type=str,
        default='data/processed/featured_data.parquet',
        help='Path to input parquet file (default: data/processed/featured_data.parquet)'
    )
    
    parser.add_argument(
        '--n_components',
        type=int,
        default=None,
        help='Number of PCA components. If not specified, will auto-determine optimal number.'
    )
    
    parser.add_argument(
        '--auto_components',
        action='store_true',
        help='Automatically determine optimal number of components using multiple criteria'
    )
    
    parser.add_argument(
        '--variance_threshold',
        type=float,
        default=0.95,
        help='Target cumulative variance for auto-selection (default: 0.95)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed/pca',
        help='Output directory for transformed data and visualizations (default: data/processed/pca)'
    )
    
    parser.add_argument(
        '--include_outcomes',
        action='store_true',
        help='Include outcome variables in PCA (default: False, excludes Y_won, etc.)'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("PCA Transformation Script")
    print("="*70)
    print(f"Input:          {args.input_path}")
    print(f"Output Dir:     {args.output_dir}")
    print(f"Exclude Outcomes: {not args.include_outcomes}")
    print(f"Random State:    {args.random_state}")
    print("="*70)
    
    # Load data
    df = load_data(args.input_path)
    
    # Prepare features
    feature_df, feature_cols, metadata_cols = prepare_features(df, exclude_outcome_vars=not args.include_outcomes)
    
    # Determine number of components
    optimal_analysis = None
    if args.auto_components or args.n_components is None:
        print("\n" + "="*70)
        print("AUTO-DETERMINING OPTIMAL NUMBER OF COMPONENTS")
        print("="*70)
        optimal_analysis = find_optimal_components(
            feature_df, 
            variance_threshold=args.variance_threshold,
            random_state=args.random_state
        )
        n_components = optimal_analysis['recommended']
        print(f"\n✓ Using {n_components} components (auto-determined)")
    else:
        n_components = args.n_components
        print(f"\nUsing {n_components} components (user-specified)")
    
    print("="*70)
    
    # Apply PCA
    pca_df, pca, scaler = apply_pca(feature_df, n_components, random_state=args.random_state)
    
    # Create visualizations
    create_visualizations(pca, args.output_dir, n_components, optimal_analysis=optimal_analysis)
    
    # Save results
    output_path = os.path.join(args.output_dir, f'featured_data_pca_{n_components}components.parquet')
    model_output_path = os.path.join(args.output_dir, f'pca_model_{n_components}components.pkl')
    
    save_results(
        pca_df, df, metadata_cols, pca, scaler,
        output_path, model_output_path
    )
    
    print("\n" + "="*70)
    print("✅ PCA transformation completed successfully!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  📊 Transformed dataset: {output_path}")
    print(f"  🤖 PCA model: {model_output_path}")
    print(f"  📈 Visualizations: {args.output_dir}/pca_*.png")
    print()


if __name__ == '__main__':
    main()

