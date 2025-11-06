#!/usr/bin/env python3
"""
Data preprocessing utilities for League of Legends match data
Handles data cleaning, feature engineering, and preparation for ML models
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_data_leakage_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features that directly compute target variables (data leakage)
    
    Args:
        data: DataFrame with potential leakage features
        
    Returns:
        DataFrame with leakage features removed
    """
    
    leakage_features = [
        'Total_Minions_Killed_Difference',
        'Total_Jungle_Minions_Killed_Difference',
        'Total_Kill_Difference',
        'Total_Assist_Difference',
        'Elite_Monster_Killed_Difference',
        'Buildings_Taken_Difference',
        'Total_Gold_Difference_Last_Time_Frame',
        'Total_Xp_Difference_Last_Time_Frame'
    ]
    
    # Remove leakage features
    removed_features = []
    for feature in leakage_features:
        if feature in data.columns:
            data = data.drop(columns=[feature])
            removed_features.append(feature)
    
    if removed_features:
        logger.info(f"Removed {len(removed_features)} leakage features: {removed_features}")
    else:
        logger.info("No leakage features found")
    
    return data

def create_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features for tree-based models
    (LSTM models don't need these as they learn temporal patterns automatically)
    
    Args:
        data: DataFrame with match data
        
    Returns:
        DataFrame with temporal features added
    """
    
    logger.info("Creating temporal features...")
    
    # Group by match_id for temporal operations
    grouped = data.groupby('match_id')
    
    # Lag features (previous timesteps)
    for lag in range(1, 6):  # 5 previous timesteps
        data[f'Gold_Diff_Lag{lag}'] = grouped['Total_Gold_Difference'].shift(lag)
        data[f'XP_Diff_Lag{lag}'] = grouped['Total_Xp_Difference'].shift(lag)
    
    # Momentum features (rate of change)
    data['Gold_Momentum'] = grouped['Total_Gold_Difference'].diff()
    data['XP_Momentum'] = grouped['Total_Xp_Difference'].diff()
    data['Gold_Acceleration'] = grouped['Gold_Momentum'].diff()
    data['XP_Acceleration'] = grouped['XP_Momentum'].diff()
    
    # Rolling statistics
    for window in [3, 5, 10]:
        data[f'Gold_MA_{window}'] = grouped['Total_Gold_Difference'].rolling(window).mean()
        data[f'XP_MA_{window}'] = grouped['Total_Xp_Difference'].rolling(window).mean()
        data[f'Gold_Std_{window}'] = grouped['Total_Gold_Difference'].rolling(window).std()
        data[f'XP_Std_{window}'] = grouped['Total_Xp_Difference'].rolling(window).std()
    
    # Trend features
    data['Gold_Trend'] = grouped['Total_Gold_Difference'].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    data['XP_Trend'] = grouped['Total_Xp_Difference'].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    
    logger.info(f"Created {len([col for col in data.columns if any(x in col for x in ['Lag', 'Momentum', 'MA_', 'Std_', 'Trend'])])} temporal features")
    
    return data

def handle_missing_values(data: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    Args:
        data: DataFrame with potential missing values
        strategy: Strategy for imputation ('forward_fill', 'backward_fill', 'mean', 'median')
        
    Returns:
        DataFrame with missing values handled
    """
    
    logger.info(f"Handling missing values with strategy: {strategy}")
    
    # Check for missing values
    missing_counts = data.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) == 0:
        logger.info("No missing values found")
        return data
    
    logger.info(f"Found missing values in {len(missing_cols)} columns")
    
    if strategy == 'forward_fill':
        # Forward fill within each match
        data = data.groupby('match_id').fillna(method='ffill')
    elif strategy == 'backward_fill':
        # Backward fill within each match
        data = data.groupby('match_id').fillna(method='bfill')
    elif strategy in ['mean', 'median']:
        # Use sklearn imputer
        imputer = SimpleImputer(strategy=strategy)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Check remaining missing values
    remaining_missing = data.isnull().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"Still {remaining_missing} missing values after imputation")
    else:
        logger.info("All missing values handled successfully")
    
    return data

def normalize_features(
    data: pd.DataFrame,
    feature_cols: List[str] = None,
    method: str = 'standard',
    fit_on_train: bool = True
) -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Normalize feature columns
    
    Args:
        data: DataFrame with features to normalize
        feature_cols: List of columns to normalize (if None, auto-detect)
        method: Normalization method ('standard', 'minmax')
        fit_on_train: Whether to fit scaler (True for training data)
        
    Returns:
        Tuple of (normalized_data, scaler)
    """
    
    if feature_cols is None:
        # Auto-detect feature columns
        exclude_cols = ['match_id', 'frame_idx', 'timestamp'] + [
            col for col in data.columns if col.startswith('Unnamed')
        ]
        feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    logger.info(f"Normalizing {len(feature_cols)} features using {method} scaling")
    
    # Create scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit and transform
    if fit_on_train:
        data[feature_cols] = scaler.fit_transform(data[feature_cols])
        logger.info("Fitted scaler on training data")
    else:
        # For validation/test data, scaler should be pre-fitted
        logger.warning("Using pre-fitted scaler - make sure it was fitted on training data")
        return data, None
    
    return data, scaler

def create_target_variables(
    data: pd.DataFrame,
    target_type: str = 'current',
    prediction_horizon: int = 5
) -> pd.DataFrame:
    """
    Create target variables for different prediction tasks
    
    Args:
        data: DataFrame with match data
        target_type: Type of target ('current', 'future', 'momentum', 'outcome')
        prediction_horizon: Number of timesteps ahead to predict (for 'future' type)
        
    Returns:
        DataFrame with target variables added
    """
    
    logger.info(f"Creating {target_type} target variables")
    
    if target_type == 'current':
        # Use current gold/XP differences (existing columns)
        logger.info("Using current gold/XP differences as targets")
        
    elif target_type == 'future':
        # Predict future values
        data['Future_Gold_Diff'] = data.groupby('match_id')['Total_Gold_Difference'].shift(-prediction_horizon)
        data['Future_XP_Diff'] = data.groupby('match_id')['Total_Xp_Difference'].shift(-prediction_horizon)
        logger.info(f"Created future targets {prediction_horizon} steps ahead")
        
    elif target_type == 'momentum':
        # Predict rate of change
        data['Gold_Momentum_Target'] = data.groupby('match_id')['Total_Gold_Difference'].diff()
        data['XP_Momentum_Target'] = data.groupby('match_id')['Total_Xp_Difference'].diff()
        logger.info("Created momentum targets")
        
    elif target_type == 'outcome':
        # Predict match outcome (binary)
        # Use final gold difference to determine winner
        final_gold_diff = data.groupby('match_id')['Total_Gold_Difference'].transform('last')
        data['Match_Outcome'] = (final_gold_diff > 0).astype(int)
        logger.info("Created binary match outcome targets")
        
    else:
        raise ValueError(f"Unknown target type: {target_type}")
    
    return data

def detect_outliers(data: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect and handle outliers in the dataset
    
    Args:
        data: DataFrame with potential outliers
        method: Method for outlier detection ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers handled
    """
    
    logger.info(f"Detecting outliers using {method} method")
    
    # Select numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    outlier_counts = {}
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outliers = z_scores > threshold
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outlier_count = outliers.sum()
        if outlier_count > 0:
            outlier_counts[col] = outlier_count
            # Cap outliers instead of removing them
            data[col] = np.where(data[col] < lower_bound if method == 'iqr' else data[col] < data[col].mean() - threshold * data[col].std(),
                                data[col].quantile(0.05), data[col])
            data[col] = np.where(data[col] > upper_bound if method == 'iqr' else data[col] > data[col].mean() + threshold * data[col].std(),
                                data[col].quantile(0.95), data[col])
    
    if outlier_counts:
        logger.info(f"Handled outliers in {len(outlier_counts)} columns: {outlier_counts}")
    else:
        logger.info("No outliers detected")
    
    return data

def preprocess_data(
    data: pd.DataFrame,
    remove_leakage: bool = True,
    create_temporal: bool = False,
    handle_missing: bool = True,
    normalize: bool = True,
    target_type: str = 'current',
    **kwargs
) -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Complete preprocessing pipeline
    
    Args:
        data: Raw DataFrame
        remove_leakage: Whether to remove data leakage features
        create_temporal: Whether to create temporal features (for tree models)
        handle_missing: Whether to handle missing values
        normalize: Whether to normalize features
        target_type: Type of target variables to create
        **kwargs: Additional arguments for preprocessing functions
        
    Returns:
        Tuple of (processed_data, scaler)
    """
    
    logger.info("Starting data preprocessing pipeline")
    
    # Remove data leakage
    if remove_leakage:
        data = remove_data_leakage_features(data)
    
    # Create temporal features (for tree-based models)
    if create_temporal:
        data = create_temporal_features(data)
    
    # Handle missing values
    if handle_missing:
        data = handle_missing_values(data)
    
    # Create target variables
    data = create_target_variables(data, target_type=target_type)
    
    # Detect and handle outliers
    data = detect_outliers(data)
    
    # Normalize features
    scaler = None
    if normalize:
        data, scaler = normalize_features(data)
    
    logger.info("Data preprocessing completed successfully")
    logger.info(f"Final shape: {data.shape}")
    
    return data, scaler

# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("data/processed/featured_data_with_scores.csv")
    
    # Preprocess data
    processed_data, scaler = preprocess_data(
        data,
        remove_leakage=True,
        create_temporal=False,  # For LSTM models
        handle_missing=True,
        normalize=True,
        target_type='current'
    )
    
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Scaler fitted: {scaler is not None}")
