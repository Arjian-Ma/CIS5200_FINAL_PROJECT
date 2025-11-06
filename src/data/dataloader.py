#!/usr/bin/env python3
"""
PyTorch DataLoader for League of Legends sequential data
Supports RNN, LSTM, Transformer, and Diffusion models
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoLSequenceDataset(Dataset):
    """
    PyTorch Dataset for League of Legends sequential data
    Supports RNN, LSTM, Transformer, and Diffusion models
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 15,
        target_cols: List[str] = None,
        feature_cols: List[str] = None,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
        remove_leakage: bool = True
    ):
        """
        Initialize the LoL Sequence Dataset
        
        Args:
            data: DataFrame with match data
            sequence_length: Length of input sequences
            target_cols: List of target column names
            feature_cols: List of feature column names (if None, auto-detect)
            scaler: Pre-fitted scaler (if None, will fit new one)
            fit_scaler: Whether to fit scaler on this data
            remove_leakage: Whether to remove data leakage features
        """
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.target_cols = target_cols or ['Total_Gold_Difference']
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.fit_scaler = fit_scaler
        self.remove_leakage = remove_leakage
        
        # Remove data leakage features if requested
        if self.remove_leakage:
            self._remove_leakage_features()
        
        # Auto-detect feature columns if not provided
        if self.feature_cols is None:
            self._detect_feature_columns()
        
        # Create sequences
        self.sequences, self.targets, self.sequence_lengths, self.match_ids = self._create_sequences()
        
        # Fit scaler if needed
        if self.fit_scaler and self.scaler is None:
            self._fit_scaler()
        
        # Apply scaling
        if self.scaler is not None:
            self._apply_scaling()
        
        logger.info(f"Created dataset with {len(self.sequences)} sequences")
        logger.info(f"Sequence shape: {self.sequences.shape}")
        logger.info(f"Target shape: {self.targets.shape}")
    
    def _remove_leakage_features(self):
        """Remove features that directly compute targets (data leakage)"""
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
        
        # Remove leakage features from data
        for feature in leakage_features:
            if feature in self.data.columns:
                self.data = self.data.drop(columns=[feature])
                logger.info(f"Removed leakage feature: {feature}")
    
    def _detect_feature_columns(self):
        """Auto-detect feature columns (exclude targets and metadata)"""
        exclude_cols = (
            self.target_cols + 
            ['match_id', 'frame_idx', 'timestamp'] +
            [col for col in self.data.columns if col.startswith('Unnamed')]
        )
        
        self.feature_cols = [
            col for col in self.data.columns 
            if col not in exclude_cols
        ]
        
        logger.info(f"Auto-detected {len(self.feature_cols)} feature columns")
    
    def _create_sequences(self):
        """Create sequences for each match"""
        sequences = []
        targets = []
        sequence_lengths = []
        match_ids = []
        
        # Group by match_id to create sequences within each match
        for match_id in self.data['match_id'].unique():
            match_data = self.data[self.data['match_id'] == match_id].sort_values('frame_idx')
            
            if len(match_data) < self.sequence_length:
                logger.warning(f"Match {match_id} has only {len(match_data)} frames, skipping")
                continue
            
            # Create overlapping sequences within the match
            for i in range(len(match_data) - self.sequence_length + 1):
                seq = match_data.iloc[i:i + self.sequence_length]
                
                # Features
                X = seq[self.feature_cols].values
                sequences.append(X)
                
                # Targets (use last timestep of sequence)
                y = seq[self.target_cols].iloc[-1].values
                targets.append(y)
                
                # Store metadata
                sequence_lengths.append(len(seq))
                match_ids.append(match_id)
        
        return np.array(sequences), np.array(targets), np.array(sequence_lengths), np.array(match_ids)
    
    def _fit_scaler(self):
        """Fit scaler on training data"""
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        # Reshape for scaling: (n_samples * seq_len, n_features)
        reshaped = self.sequences.reshape(-1, self.sequences.shape[-1])
        self.scaler.fit(reshaped)
        logger.info("Fitted StandardScaler on training data")
    
    def _apply_scaling(self):
        """Apply scaling to sequences"""
        if self.scaler is not None:
            # Reshape for scaling
            original_shape = self.sequences.shape
            reshaped = self.sequences.reshape(-1, self.sequences.shape[-1])
            
            # Scale and reshape back
            scaled = self.scaler.transform(reshaped)
            self.sequences = scaled.reshape(original_shape)
            logger.info("Applied scaling to sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': torch.FloatTensor(self.sequences[idx]),
            'target': torch.FloatTensor(self.targets[idx]),
            'length': torch.LongTensor([self.sequence_lengths[idx]]),
            'match_id': self.match_ids[idx]
        }

class LoLDiffusionDataset(Dataset):
    """
    Specialized dataset for diffusion models
    Handles noise scheduling and conditioning
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 20,
        noise_schedule: str = 'linear',
        **kwargs
    ):
        """
        Initialize the LoL Diffusion Dataset
        
        Args:
            data: DataFrame with match data
            sequence_length: Length of input sequences
            noise_schedule: Noise scheduling strategy ('linear', 'cosine', 'quadratic')
            **kwargs: Additional arguments for base dataset
        """
        self.data = data
        self.sequence_length = sequence_length
        self.noise_schedule = noise_schedule
        
        # Create base sequences
        self.base_dataset = LoLSequenceDataset(data, sequence_length, **kwargs)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base sequence
        x, y = self.base_dataset[idx]
        
        # Add noise for diffusion training
        noise = torch.randn_like(x)
        timestep = torch.randint(0, 1000, (1,))  # Random timestep
        
        # Apply noise based on timestep
        noisy_x = self._apply_noise(x, noise, timestep)
        
        return {
            'clean': x,
            'noisy': noisy_x,
            'noise': noise,
            'timestep': timestep,
            'target': y
        }
    
    def _apply_noise(self, x, noise, timestep):
        """Apply noise based on diffusion schedule"""
        if self.noise_schedule == 'linear':
            alpha = 1.0 - (timestep.float() / 1000.0)
        elif self.noise_schedule == 'cosine':
            alpha = torch.cos(timestep.float() * np.pi / 2000.0) ** 2
        elif self.noise_schedule == 'quadratic':
            alpha = 1.0 - (timestep.float() / 1000.0) ** 2
        else:
            alpha = 1.0
        
        return alpha * x + (1 - alpha) * noise

def collate_fn(batch):
    """
    Custom collate function for variable length sequences
    Handles padding and packing for LSTM efficiency
    """
    sequences = [item['sequence'] for item in batch]
    targets = [item['target'] for item in batch]
    lengths = [item['length'] for item in batch]
    match_ids = [item['match_id'] for item in batch]
    
    # Pad sequences to same length
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Stack targets
    targets = torch.stack(targets)
    
    # Stack lengths
    lengths = torch.cat(lengths)
    
    return {
        'sequences': padded_sequences,
        'targets': targets,
        'lengths': lengths,
        'match_ids': match_ids
    }

def create_match_based_batches(
    data: pd.DataFrame,
    batch_size: int = 32,
    sequence_length: int = 20
) -> List[Dict]:
    """
    Create batches where each batch contains sequences from the same match
    This ensures temporal consistency within batches
    
    Args:
        data: DataFrame with match data
        batch_size: Number of sequences per batch
        sequence_length: Length of each sequence
        
    Returns:
        List of batch dictionaries
    """
    batches = []
    
    # Group by match_id
    for match_id in data['match_id'].unique():
        match_data = data[data['match_id'] == match_id].sort_values('frame_idx')
        
        if len(match_data) < sequence_length:
            continue
            
        # Create sequences for this match
        match_sequences = []
        for i in range(len(match_data) - sequence_length + 1):
            seq = match_data.iloc[i:i + sequence_length]
            match_sequences.append(seq)
        
        # Split into batches of size batch_size
        for i in range(0, len(match_sequences), batch_size):
            batch_sequences = match_sequences[i:i + batch_size]
            if len(batch_sequences) == batch_size:  # Only full batches
                batches.append({
                    'match_id': match_id,
                    'sequences': batch_sequences,
                    'batch_size': len(batch_sequences)
                })
    
    logger.info(f"Created {len(batches)} match-based batches")
    return batches

def create_dataloaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_type: str = 'lstm',
    batch_size: int = 32,
    sequence_length: int = 20,
    target_cols: List[str] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for different model types
    
    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        test_data: Test DataFrame
        model_type: Type of model ('rnn', 'lstm', 'transformer', 'diffusion')
        batch_size: Batch size for training
        sequence_length: Length of input sequences
        target_cols: Target columns to predict
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for GPU training
        **kwargs: Additional arguments for dataset creation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    logger.info(f"Creating DataLoaders for {model_type} model")
    logger.info(f"Batch size: {batch_size}, Sequence length: {sequence_length}")
    
    # Create datasets
    if model_type == 'diffusion':
        train_dataset = LoLDiffusionDataset(
            train_data,
            sequence_length=sequence_length,
            target_cols=target_cols,
            fit_scaler=True,
            **kwargs
        )
        
        # Use same scaler for val/test
        val_dataset = LoLDiffusionDataset(
            val_data,
            sequence_length=sequence_length,
            target_cols=target_cols,
            scaler=train_dataset.base_dataset.scaler,
            fit_scaler=False,
            **kwargs
        )
        
        test_dataset = LoLDiffusionDataset(
            test_data,
            sequence_length=sequence_length,
            target_cols=target_cols,
            scaler=train_dataset.base_dataset.scaler,
            fit_scaler=False,
            **kwargs
        )
    else:
        # Standard sequence datasets
        train_dataset = LoLSequenceDataset(
            train_data,
            sequence_length=sequence_length,
            target_cols=target_cols,
            fit_scaler=True,
            **kwargs
        )
        
        # Use same scaler for val/test
        val_dataset = LoLSequenceDataset(
            val_data,
            sequence_length=sequence_length,
            target_cols=target_cols,
            scaler=train_dataset.scaler,
            fit_scaler=False,
            **kwargs
        )
        
        test_dataset = LoLSequenceDataset(
            test_data,
            sequence_length=sequence_length,
            target_cols=target_cols,
            scaler=train_dataset.scaler,
            fit_scaler=False,
            **kwargs
        )
    
    # Create DataLoaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    logger.info(f"Created DataLoaders:")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

def create_data_splits(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test splits by match_id (not by row)
    This prevents data leakage in temporal data
    
    Args:
        data: Full dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    
    # Get unique match IDs
    unique_matches = data['match_id'].unique()
    n_matches = len(unique_matches)
    
    # Calculate split sizes
    n_train = int(n_matches * train_ratio)
    n_val = int(n_matches * val_ratio)
    n_test = n_matches - n_train - n_val
    
    # Shuffle match IDs
    np.random.seed(random_state)
    shuffled_matches = np.random.permutation(unique_matches)
    
    # Split match IDs
    train_matches = shuffled_matches[:n_train]
    val_matches = shuffled_matches[n_train:n_train + n_val]
    test_matches = shuffled_matches[n_train + n_val:]
    
    # Create data splits
    train_data = data[data['match_id'].isin(train_matches)].copy()
    val_data = data[data['match_id'].isin(val_matches)].copy()
    test_data = data[data['match_id'].isin(test_matches)].copy()
    
    logger.info(f"Data splits created:")
    logger.info(f"  Train: {len(train_matches)} matches, {len(train_data)} rows")
    logger.info(f"  Val: {len(val_matches)} matches, {len(val_data)} rows")
    logger.info(f"  Test: {len(test_matches)} matches, {len(test_data)} rows")
    
    return train_data, val_data, test_data

def save_data_splits(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_dir: str = "data/splits"
):
    """Save data splits to Parquet files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_data.to_parquet(f"{output_dir}/train.parquet", index=False)
    val_data.to_parquet(f"{output_dir}/val.parquet", index=False)
    test_data.to_parquet(f"{output_dir}/test.parquet", index=False)
    
    logger.info(f"Data splits saved to {output_dir}")
    logger.info(f"Train: {len(train_data)} rows")
    logger.info(f"Val: {len(val_data)} rows")
    logger.info(f"Test: {len(test_data)} rows")

def load_data_splits(
    data_dir: str = "data/splits"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data splits from Parquet files"""
    
    train_data = pd.read_parquet(f"{data_dir}/train.parquet")
    val_data = pd.read_parquet(f"{data_dir}/val.parquet")
    test_data = pd.read_parquet(f"{data_dir}/test.parquet")
    
    logger.info(f"Data splits loaded from {data_dir}")
    logger.info(f"Train: {len(train_data)} rows")
    logger.info(f"Val: {len(val_data)} rows")
    logger.info(f"Test: {len(test_data)} rows")
    return train_data, val_data, test_data

# Example usage for creating data splits
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create data splits for LoL dataset")
    parser.add_argument("--input_file", type=str, default="data/processed/featured_data_with_scores.parquet", 
                       help="Input Parquet file")
    parser.add_argument("--output_dir", type=str, default="data/splits", 
                       help="Output directory for splits")
    parser.add_argument("--train_ratio", type=float, default=0.7, 
                       help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, 
                       help="Validation data ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, 
                       help="Test data ratio")
    parser.add_argument("--random_state", type=int, default=42, 
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("=== Creating Data Splits for LoL Dataset ===")
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")
    print()
    
    try:
        # Load data
        print("Loading featured data...")
        data = pd.read_parquet(args.input_file)
        print(f"Loaded {len(data)} rows from {args.input_file}")
        print(f"Data shape: {data.shape}")
        print(f"Unique matches: {data['match_id'].nunique()}")
        
        # Create splits
        print("\nCreating data splits...")
        train_data, val_data, test_data = create_data_splits(
            data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_state
        )
        
        # Save splits
        print("\nSaving data splits...")
        save_data_splits(train_data, val_data, test_data, args.output_dir)
        
    except FileNotFoundError as e:
        print(f"❌ Error: Input file not found: {args.input_file}")
        print("Please make sure you have run the data_featuring.py script first")
    except Exception as e:
        print(f"❌ Error during data splitting: {e}")
