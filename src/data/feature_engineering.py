#!/usr/bin/env python3
"""
Feature engineering utilities for League of Legends match data
Creates advanced features and handles feature selection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between different game aspects
    
    Args:
        data: DataFrame with match data
        
    Returns:
        DataFrame with interaction features added
    """
    
    logger.info("Creating interaction features...")
    
    # Gold-XP interactions
    data['Gold_XP_Ratio'] = data['Total_Gold_Difference'] / (data['Total_Xp_Difference'] + 1e-8)
    data['Gold_XP_Product'] = data['Total_Gold_Difference'] * data['Total_Xp_Difference']
    
    # Damage interactions
    if 'Total_Damage_Done_Diff' in data.columns:
        data['Damage_Efficiency'] = data['Total_Damage_Done_Diff'] / (data['Total_Damage_Taken_Diff'] + 1e-8)
        data['Damage_Ratio'] = data['Total_Damage_Done_Diff'] / (data['Total_Damage_Done_To_Champions_Diff'] + 1e-8)
    
    # Vision interactions
    if 'Total_Ward_Placed_Difference' in data.columns and 'Total_Ward_Killed_Difference' in data.columns:
        data['Ward_Net_Difference'] = data['Total_Ward_Placed_Difference'] - data['Total_Ward_Killed_Difference']
        data['Ward_Efficiency'] = data['Total_Ward_Killed_Difference'] / (data['Total_Ward_Placed_Difference'] + 1e-8)
    
    # Team composition interactions
    if 'Blue_Team_Overall_Score' in data.columns and 'Red_Team_Overall_Score' in data.columns:
        data['Team_Score_Ratio'] = data['Blue_Team_Overall_Score'] / (data['Red_Team_Overall_Score'] + 1e-8)
        data['Team_Score_Product'] = data['Blue_Team_Overall_Score'] * data['Red_Team_Overall_Score']
    
    # Player score interactions
    for i in range(1, 6):  # Blue team players
        for j in range(i+1, 6):
            if f'Player{i}_Overall_Score' in data.columns and f'Player{j}_Overall_Score' in data.columns:
                data[f'Player{i}_{j}_Score_Product'] = data[f'Player{i}_Overall_Score'] * data[f'Player{j}_Overall_Score']
    
    for i in range(6, 11):  # Red team players
        for j in range(i+1, 11):
            if f'Player{i}_Overall_Score' in data.columns and f'Player{j}_Overall_Score' in data.columns:
                data[f'Player{i}_{j}_Score_Product'] = data[f'Player{i}_Overall_Score'] * data[f'Player{j}_Overall_Score']
    
    logger.info(f"Created {len([col for col in data.columns if any(x in col for x in ['_', 'Ratio', 'Efficiency', 'Product'])])} interaction features")
    
    return data

def create_polynomial_features(data: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial features for important variables
    
    Args:
        data: DataFrame with match data
        degree: Degree of polynomial features
        
    Returns:
        DataFrame with polynomial features added
    """
    
    logger.info(f"Creating polynomial features of degree {degree}")
    
    # Important features for polynomial expansion
    important_features = [
        'Total_Gold_Difference',
        'Total_Xp_Difference',
        'Blue_Team_Overall_Score',
        'Red_Team_Overall_Score'
    ]
    
    # Filter to existing features
    important_features = [f for f in important_features if f in data.columns]
    
    for feature in important_features:
        for d in range(2, degree + 1):
            data[f'{feature}_Power_{d}'] = data[feature] ** d
    
    logger.info(f"Created polynomial features for {len(important_features)} base features")
    
    return data

def create_rolling_features(data: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """
    Create rolling window features
    
    Args:
        data: DataFrame with match data
        windows: List of window sizes for rolling features
        
    Returns:
        DataFrame with rolling features added
    """
    
    logger.info(f"Creating rolling features with windows: {windows}")
    
    # Features to create rolling statistics for
    rolling_features = [
        'Total_Gold_Difference',
        'Total_Xp_Difference',
        'Blue_Team_Overall_Score',
        'Red_Team_Overall_Score'
    ]
    
    # Filter to existing features
    rolling_features = [f for f in rolling_features if f in data.columns]
    
    grouped = data.groupby('match_id')
    
    for feature in rolling_features:
        for window in windows:
            # Rolling mean
            data[f'{feature}_MA_{window}'] = grouped[feature].rolling(window).mean()
            
            # Rolling standard deviation
            data[f'{feature}_Std_{window}'] = grouped[feature].rolling(window).std()
            
            # Rolling min/max
            data[f'{feature}_Min_{window}'] = grouped[feature].rolling(window).min()
            data[f'{feature}_Max_{window}'] = grouped[feature].rolling(window).max()
            
            # Rolling trend (slope)
            data[f'{feature}_Trend_{window}'] = grouped[feature].rolling(window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
            )
    
    logger.info(f"Created rolling features for {len(rolling_features)} base features")
    
    return data

def create_team_composition_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create team composition and synergy features
    
    Args:
        data: DataFrame with match data
        
    Returns:
        DataFrame with team composition features added
    """
    
    logger.info("Creating team composition features...")
    
    # Blue team composition
    blue_players = [f'Player{i}_Overall_Score' for i in range(1, 6)]
    blue_players = [p for p in blue_players if p in data.columns]
    
    if len(blue_players) >= 2:
        data['Blue_Team_Score_Mean'] = data[blue_players].mean(axis=1)
        data['Blue_Team_Score_Std'] = data[blue_players].std(axis=1)
        data['Blue_Team_Score_Min'] = data[blue_players].min(axis=1)
        data['Blue_Team_Score_Max'] = data[blue_players].max(axis=1)
        data['Blue_Team_Score_Range'] = data['Blue_Team_Score_Max'] - data['Blue_Team_Score_Min']
    
    # Red team composition
    red_players = [f'Player{i}_Overall_Score' for i in range(6, 11)]
    red_players = [p for p in red_players if p in data.columns]
    
    if len(red_players) >= 2:
        data['Red_Team_Score_Mean'] = data[red_players].mean(axis=1)
        data['Red_Team_Score_Std'] = data[red_players].std(axis=1)
        data['Red_Team_Score_Min'] = data[red_players].min(axis=1)
        data['Red_Team_Score_Max'] = data[red_players].max(axis=1)
        data['Red_Team_Score_Range'] = data['Red_Team_Score_Max'] - data['Red_Team_Score_Min']
    
    # Team balance features
    if 'Blue_Team_Score_Std' in data.columns and 'Red_Team_Score_Std' in data.columns:
        data['Team_Balance_Diff'] = data['Blue_Team_Score_Std'] - data['Red_Team_Score_Std']
        data['Team_Balance_Ratio'] = data['Blue_Team_Score_Std'] / (data['Red_Team_Score_Std'] + 1e-8)
    
    # Player role features (if available)
    role_features = ['Offensive_Score', 'Defensive_Score', 'Sustain_Score', 'Resources_Score', 'Mobility_Score']
    
    for role in role_features:
        blue_role_cols = [f'Player{i}_{role}' for i in range(1, 6)]
        red_role_cols = [f'Player{i}_{role}' for i in range(6, 11)]
        
        blue_role_cols = [c for c in blue_role_cols if c in data.columns]
        red_role_cols = [c for c in red_role_cols if c in data.columns]
        
        if blue_role_cols:
            data[f'Blue_Team_{role}_Mean'] = data[blue_role_cols].mean(axis=1)
        if red_role_cols:
            data[f'Red_Team_{role}_Mean'] = data[red_role_cols].mean(axis=1)
    
    logger.info("Created team composition features")
    
    return data

def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'mutual_info',
    k: int = 50,
    random_state: int = 42
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select the most important features using various methods
    
    Args:
        X: Feature DataFrame
        y: Target Series
        method: Feature selection method ('mutual_info', 'f_regression', 'random_forest')
        k: Number of features to select
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (selected_features, selected_feature_names)
    """
    
    logger.info(f"Selecting {k} features using {method} method")
    
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
    elif method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    elif method == 'random_forest':
        # Use Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
        rf.fit(X, y)
        feature_importance = rf.feature_importances_
        feature_names = X.columns
        # Select top k features
        top_k_indices = np.argsort(feature_importance)[-k:]
        selected_features = X.iloc[:, top_k_indices]
        selected_feature_names = feature_names[top_k_indices].tolist()
        return selected_features, selected_feature_names
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit selector
    X_selected = selector.fit_transform(X, y)
    selected_feature_names = X.columns[selector.get_support()].tolist()
    
    # Convert back to DataFrame
    selected_features = pd.DataFrame(X_selected, columns=selected_feature_names, index=X.index)
    
    logger.info(f"Selected {len(selected_feature_names)} features")
    
    return selected_features, selected_feature_names

def reduce_dimensionality(
    X: pd.DataFrame,
    method: str = 'pca',
    n_components: int = 50,
    random_state: int = 42
) -> Tuple[pd.DataFrame, object]:
    """
    Reduce dimensionality using PCA or other methods
    
    Args:
        X: Feature DataFrame
        method: Dimensionality reduction method ('pca', 'truncated_svd')
        n_components: Number of components to keep
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (reduced_features, transformer)
    """
    
    logger.info(f"Reducing dimensionality using {method} to {n_components} components")
    
    if method == 'pca':
        transformer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit and transform
    X_reduced = transformer.fit_transform(X)
    
    # Create column names
    component_names = [f'{method.upper()}_{i+1}' for i in range(n_components)]
    
    # Convert back to DataFrame
    X_reduced_df = pd.DataFrame(X_reduced, columns=component_names, index=X.index)
    
    # Calculate explained variance
    if method == 'pca':
        explained_variance = transformer.explained_variance_ratio_.sum()
        logger.info(f"Explained variance: {explained_variance:.3f}")
    
    logger.info(f"Reduced from {X.shape[1]} to {X_reduced_df.shape[1]} features")
    
    return X_reduced_df, transformer

def create_feature_engineering_pipeline(
    data: pd.DataFrame,
    target_cols: List[str] = None,
    create_interactions: bool = True,
    create_polynomials: bool = False,
    create_rolling: bool = False,
    create_team_composition: bool = True,
    select_features: bool = False,
    reduce_dimensions: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete feature engineering pipeline
    
    Args:
        data: Raw DataFrame
        target_cols: List of target columns
        create_interactions: Whether to create interaction features
        create_polynomials: Whether to create polynomial features
        create_rolling: Whether to create rolling features
        create_team_composition: Whether to create team composition features
        select_features: Whether to perform feature selection
        reduce_dimensions: Whether to reduce dimensionality
        **kwargs: Additional arguments for feature engineering functions
        
    Returns:
        Tuple of (engineered_data, feature_info)
    """
    
    logger.info("Starting feature engineering pipeline")
    
    feature_info = {
        'original_features': len(data.columns),
        'steps_applied': []
    }
    
    # Create interaction features
    if create_interactions:
        data = create_interaction_features(data)
        feature_info['steps_applied'].append('interactions')
    
    # Create polynomial features
    if create_polynomials:
        data = create_polynomial_features(data)
        feature_info['steps_applied'].append('polynomials')
    
    # Create rolling features
    if create_rolling:
        data = create_rolling_features(data)
        feature_info['steps_applied'].append('rolling')
    
    # Create team composition features
    if create_team_composition:
        data = create_team_composition_features(data)
        feature_info['steps_applied'].append('team_composition')
    
    feature_info['features_after_engineering'] = len(data.columns)
    
    # Feature selection
    if select_features and target_cols:
        # Separate features and targets
        feature_cols = [col for col in data.columns if col not in target_cols + ['match_id', 'frame_idx', 'timestamp']]
        X = data[feature_cols]
        y = data[target_cols[0]]  # Use first target column
        
        # Select features
        X_selected, selected_names = select_features(X, y, **kwargs)
        
        # Combine with metadata
        metadata_cols = ['match_id', 'frame_idx', 'timestamp'] + target_cols
        metadata_cols = [col for col in metadata_cols if col in data.columns]
        
        data = pd.concat([data[metadata_cols], X_selected], axis=1)
        feature_info['selected_features'] = selected_names
        feature_info['steps_applied'].append('feature_selection')
    
    # Dimensionality reduction
    if reduce_dimensions and not select_features:
        # Separate features and targets
        feature_cols = [col for col in data.columns if col not in target_cols + ['match_id', 'frame_idx', 'timestamp']]
        X = data[feature_cols]
        
        # Reduce dimensions
        X_reduced, transformer = reduce_dimensionality(X, **kwargs)
        
        # Combine with metadata
        metadata_cols = ['match_id', 'frame_idx', 'timestamp'] + target_cols
        metadata_cols = [col for col in metadata_cols if col in data.columns]
        
        data = pd.concat([data[metadata_cols], X_reduced], axis=1)
        feature_info['transformer'] = transformer
        feature_info['steps_applied'].append('dimensionality_reduction')
    
    feature_info['final_features'] = len(data.columns)
    
    logger.info("Feature engineering completed successfully")
    logger.info(f"Features: {feature_info['original_features']} → {feature_info['final_features']}")
    logger.info(f"Steps applied: {feature_info['steps_applied']}")
    
    return data, feature_info

# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("data/processed/featured_data_with_scores.csv")
    
    # Apply feature engineering
    engineered_data, feature_info = create_feature_engineering_pipeline(
        data,
        target_cols=['Total_Gold_Difference'],
        create_interactions=True,
        create_polynomials=False,
        create_rolling=False,
        create_team_composition=True,
        select_features=False,
        reduce_dimensions=False
    )
    
    print(f"Engineered data shape: {engineered_data.shape}")
    print(f"Feature info: {feature_info}")
