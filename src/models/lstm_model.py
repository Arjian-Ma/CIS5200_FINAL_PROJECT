#!/usr/bin/env python3
"""
PyTorch LSTM Model for League of Legends Gold Difference Prediction
Integrates with the dataloader system for sequential data processing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTM(nn.Module):
    """
    LSTM model for sequential data prediction
    Handles variable length sequences with packed sequences for efficiency
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        bidirectional: bool = False
    ):
        """
        Initialize the LSTM model
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output targets
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized LSTM model:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Hidden size: {hidden_size}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Output size: {output_size}")
        logger.info(f"  Bidirectional: {bidirectional}")
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.kaiming_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x, lengths):
        """
        Forward pass through the LSTM
        
        Args:
            x: Input sequences (batch_size, seq_len, input_size)
            lengths: Sequence lengths (batch_size,)
            
        Returns:
            Output predictions (batch_size, output_size)
        """
        # Pack sequences for efficient LSTM processing
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(packed_x)
        
        # Unpack the output
        unpacked_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Use the last valid output for each sequence
        # Get the last non-padded output for each sequence
        batch_size = x.size(0)
        last_outputs = []
        
        for i in range(batch_size):
            # Get the last valid output (at position lengths[i] - 1)
            last_idx = lengths[i] - 1
            last_output = unpacked_out[i, last_idx, :]
            last_outputs.append(last_output)
        
        # Stack the last outputs
        last_outputs = torch.stack(last_outputs)
        
        # Pass through fully connected layers
        output = self.fc_layers(last_outputs)
        
        return output
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'bidirectional': self.bidirectional
        }

class Trainer:
    """
    Trainer class for LSTM model
    Handles training, validation, and evaluation
    """
    
    def __init__(
        self,
        model: LSTM,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize the trainer
        
        Args:
            model: LSTM model to train
            device: Device to use ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        logger.info(f"Initialized trainer on device: {device}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Weight decay: {weight_decay}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move data to device
            sequences = batch['sequences'].to(self.device)
            targets = batch['targets'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences, lengths)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                sequences = batch['sequences'].to(self.device)
                targets = batch['targets'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                
                # Forward pass
                outputs = self.model(sequences, lengths)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Store predictions and targets
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        patience: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_path: Path to save the best model
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Early stopping patience: {patience}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}:")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss: {val_loss:.6f}")
            logger.info(f"  Val MSE: {val_metrics['mse']:.6f}")
            logger.info(f"  Val MAE: {val_metrics['mae']:.6f}")
            logger.info(f"  Val R²: {val_metrics['r2']:.6f}")
            logger.info(f"  Val RMSE: {val_metrics['rmse']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Saved best model to {save_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    def evaluate(self, test_loader):
        """Evaluate the model on test data"""
        logger.info("Evaluating model on test data...")
        
        val_loss, metrics = self.validate_epoch(test_loader)
        
        logger.info("Test Results:")
        logger.info(f"  MSE: {metrics['mse']:.6f}")
        logger.info(f"  MAE: {metrics['mae']:.6f}")
        logger.info(f"  R²: {metrics['r2']:.6f}")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        
        return metrics
    
    def save_model(self, path: str):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }, path)
    
    def load_model(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_metrics = checkpoint['val_metrics']
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot metrics
        if self.val_metrics:
            epochs = range(1, len(self.val_metrics) + 1)
            r2_scores = [m['r2'] for m in self.val_metrics]
            rmse_scores = [m['rmse'] for m in self.val_metrics]
            
            ax2.plot(epochs, r2_scores, label='R² Score')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('R² Score')
            ax2.set_title('Validation R² Score')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()


# Simple training function
def train_lstm(
    train_loader, val_loader, test_loader, input_size,
    num_epochs=50, feature_list=None, patience=5,
    batch_size=32, sequence_length=15, learning_rate=0.001,
    hidden_size=128, num_layers=2, dropout=0.4,
    model_save_dir=None, checkpoint_name="lstm_model.pth",
    best_checkpoint_name="lstm_model_best.pth", curve_save_path=None,
    feature_scaler=None  # Scaler for saving in checkpoint
):
    """
    Simple training function for LSTM model with early stopping and best model saving
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader  
        test_loader: Test data loader
        input_size: Number of input features
        num_epochs: Number of training epochs
        feature_list: List of feature names (for saving to checkpoint)
        patience: Early stopping patience (number of epochs with no improvement)
        batch_size: Batch size (for reference, already set in dataloader)
        sequence_length: Sequence length (for reference, already set in dataloader)
        learning_rate: Learning rate for optimizer
        hidden_size: LSTM hidden state size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        model_save_dir: Directory to save models (default: models/)
        checkpoint_name: Name for final model checkpoint
        best_checkpoint_name: Name for best model checkpoint
        curve_save_path: Path to save training curves plot
    """
    # Create model with customizable architecture
    model = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Loss and optimizer with customizable learning rate
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    val_losses = []
    test_losses = []
    
    # Early stopping and best model tracking
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    # Get paths for saving
    if model_save_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(project_root, "models")
    else:
        models_dir = model_save_dir
    
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, best_checkpoint_name)
    
    print(f"Training LSTM model on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Early stopping patience: {patience} epochs")
    
    # Training loop with progress bar
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False, unit="batch"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['lengths'].to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, targets)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"❌ NaN loss detected in training! Skipping batch.")
                continue
                
            loss.backward()
            
            # Gradient clipping to prevent NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False, unit="batch"):
                sequences = batch['sequences'].to(device)
                targets = batch['targets'].to(device)
                lengths = batch['lengths'].to(device)
                
                outputs = model(sequences, lengths)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Test
        test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1} Test", leave=False, unit="batch"):
                sequences = batch['sequences'].to(device)
                targets = batch['targets'].to(device)
                lengths = batch['lengths'].to(device)
                
                outputs = model(sequences, lengths)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        
        # Average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_test_loss = test_loss / len(test_loader)
        
        # Store losses
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        test_losses.append(avg_test_loss)
        
        # Check for NaN values
        if np.isnan(avg_train_loss) or np.isnan(avg_val_loss) or np.isnan(avg_test_loss):
            print(f"❌ NaN detected at epoch {epoch+1}! Stopping training.")
            print(f"  Train Loss: {avg_train_loss}")
            print(f"  Val Loss: {avg_val_loss}")
            print(f"  Test Loss: {avg_test_loss}")
            break
        
        # Calculate RMSE for this epoch
        train_rmse = np.sqrt(avg_train_loss)
        val_rmse = np.sqrt(avg_val_loss)
        test_rmse = np.sqrt(avg_test_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f} (RMSE: {train_rmse:.6f})")
        print(f"  Val Loss: {avg_val_loss:.6f} (RMSE: {val_rmse:.6f})")
        print(f"  Test Loss: {avg_test_loss:.6f} (RMSE: {test_rmse:.6f})")
        
        # Check for improvement and save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            # Save best model state
            best_model_state = {
                'model_state_dict': model.state_dict().copy(),
                'optimizer_state_dict': optimizer.state_dict().copy(),
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss
            }
            
            # Save best model checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses.copy(),
                'val_losses': val_losses.copy(),
                'test_losses': test_losses.copy(),
                'input_size': input_size,
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
            }
            if feature_list is not None:
                checkpoint['feature_list'] = feature_list
            if feature_scaler is not None:
                # Save scaler for feature normalization during testing
                import pickle
                import io
                scaler_buffer = io.BytesIO()
                pickle.dump(feature_scaler, scaler_buffer)
                checkpoint['feature_scaler'] = scaler_buffer.getvalue()
            
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ Best model saved! (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{patience} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹ Early stopping triggered at epoch {epoch+1}")
            print(f"   Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
            # Load best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state['model_state_dict'])
                optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
                print(f"   ✓ Restored best model from epoch {best_epoch}")
            break
    
    # Note: If early stopping occurred, we already loaded the best model above
    # If training completed without early stopping, the current model is the final model
    
    # Final results with RMSE (use best validation loss)
    best_val_rmse = np.sqrt(best_val_loss)
    final_train_rmse = np.sqrt(train_losses[-1])
    final_val_rmse = np.sqrt(val_losses[-1])
    final_test_rmse = np.sqrt(test_losses[-1])
    
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"  Best Val Loss: {best_val_loss:.6f} (RMSE: {best_val_rmse:.6f}) at epoch {best_epoch}")
    print(f"  Final Train Loss: {train_losses[-1]:.6f} (RMSE: {final_train_rmse:.6f})")
    print(f"  Final Val Loss: {val_losses[-1]:.6f} (RMSE: {final_val_rmse:.6f})")
    print(f"  Final Test Loss: {test_losses[-1]:.6f} (RMSE: {final_test_rmse:.6f})")
    print(f"{'='*60}")
    
    # Save final model (best model)
    model_path = os.path.join(models_dir, checkpoint_name)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'input_size': input_size,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
    }
    
    # Save feature list if provided (for consistent testing)
    if feature_list is not None:
        checkpoint['feature_list'] = feature_list
    
    # Save scaler for feature normalization during testing
    if feature_scaler is not None:
        import pickle
        import io
        scaler_buffer = io.BytesIO()
        pickle.dump(feature_scaler, scaler_buffer)
        checkpoint['feature_scaler'] = scaler_buffer.getvalue()
    
    torch.save(checkpoint, model_path)
    print(f"✓ Final model saved to: {model_path}")
    print(f"✓ Best model checkpoint saved to: {best_model_path}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if curve_save_path is None:
        plot_path = os.path.join(models_dir, "training_curves.png")
    else:
        plot_path = curve_save_path
        os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else '.', exist_ok=True)
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to: {plot_path}")
    plt.show()
    
    return model, train_losses, val_losses, test_losses

def load_lstm_model(model_path="models/lstm_model.pth"):
    """
    Load a trained LSTM model
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model and training history
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model with saved input size
    model = LSTM(input_size=checkpoint['input_size'], hidden_size=128, num_layers=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint

def prepare_data_loaders(feature_list=None, batch_size=32, sequence_length=15):
    """
    Combined function: Select features and create data loaders in one step
    
    Args:
        feature_list: Can be:
            - None: Auto-detect all features from train.parquet
            - List of feature names: Use these features (verified against data)
            - CSV path string: Load features from CSV file
            - 'specified': Use hardcoded specified features
        batch_size: Batch size for data loaders
        sequence_length: Sequence length for LSTM
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, input_size, feature_cols)
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.dataloader import load_data_splits, create_dataloaders
    
    print("Loading data splits...")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_splits_path = os.path.join(project_root, "data", "splits")
    train_data, val_data, test_data = load_data_splits(data_splits_path)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Available columns: {len(train_data.columns)}")
    
    # Handle CSV path string
    if isinstance(feature_list, str) and feature_list.endswith('.csv'):
        feature_list = get_specified_features(csv_path=feature_list)
    elif feature_list == 'specified':
        feature_list = get_specified_features()
    
    # Select/verify features
    y_targets = ['Total_Gold_Difference']
    metadata_cols = ['match_id', 'frame_idx', 'timestamp']
    exclude_cols = y_targets + metadata_cols
    
    if feature_list is not None:
        # Verify features exist in train.parquet and maintain order
        print(f"Verifying {len(feature_list)} specified features exist in train.parquet...")
        
        available_features = set(train_data.columns)
        verified_features = []
        missing_features = []
        
        for feat in feature_list:
            if feat in available_features:
                verified_features.append(feat)
            else:
                missing_features.append(feat)
        
        if missing_features:
            print(f"⚠ Warning: {len(missing_features)} features not found in train.parquet:")
            print(f"  Missing: {missing_features[:5]}..." if len(missing_features) > 5 else f"  Missing: {missing_features}")
        
        feature_cols = verified_features
        print(f"✓ Using {len(feature_cols)} verified features (maintaining order)")
    else:
        # Auto-detect: use all features except Y targets and metadata
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        print(f"✓ Auto-detected {len(feature_cols)} features from train.parquet")
    
    print(f"Y targets: {y_targets}")
    print(f"First 10 features: {feature_cols[:10]}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        model_type='lstm',
        batch_size=batch_size,
        sequence_length=sequence_length,
        target_cols=y_targets,
        feature_cols=feature_cols,
        remove_leakage=False
    )
    
    # Get input size from first batch
    sample_batch = next(iter(train_loader))
    input_size = sample_batch['sequences'].shape[-1]
    
    print(f"Input size: {input_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, input_size, feature_cols

def get_available_features(data_path="data/processed/featured_data_with_scores.parquet"):
    """
    Get available features from the dataset
    
    Args:
        data_path: Path to the featured data with scores
        
    Returns:
        Dictionary with different feature categories
    """
    try:
        import pandas as pd
        df = pd.read_parquet(data_path)
        
        # Categorize features
        features = {
            'all_features': [col for col in df.columns if col not in ['Total_Gold_Difference', 'Total_Xp_Difference', 'match_id', 'frame_idx', 'timestamp']],
            'team_aggregates': [col for col in df.columns if 'Difference' in col and 'Total_' in col],
            'player_scores': [col for col in df.columns if '_Score' in col],
            'spatial_features': [col for col in df.columns if col in ['CentroidDist', 'MinInterTeamDist', 'EngagedDiff', 'FrontlineOverlap', 'RadialVelocityDiff']],
            'team_scores': [col for col in df.columns if 'Team_' in col and '_Score' in col]
        }
        
        print(f"Dataset shape: {df.shape}")
        print(f"Total features: {len(features['all_features'])}")
        print(f"Team aggregates: {len(features['team_aggregates'])}")
        print(f"Player scores: {len(features['player_scores'])}")
        print(f"Spatial features: {len(features['spatial_features'])}")
        print(f"Team scores: {len(features['team_scores'])}")
        
        return features
        
    except FileNotFoundError:
        print(f"❌ File not found: {data_path}")
        print("Please run data_featuring_score.py first")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def get_specified_features(csv_path=None):
    """
    Get the specific features you want to use for training
    
    Args:
        csv_path: Optional path to CSV file with feature list. If None, returns hardcoded features.
                  CSV should have a 'feature' column with feature names.
    
    Returns:
        List of feature names in the order they appear in CSV or hardcoded list
    """
    if csv_path is not None:
        # Load features from CSV file
        import pandas as pd
        import os
        
        if not os.path.exists(csv_path):
            print(f"❌ Warning: CSV file not found: {csv_path}")
            print("Falling back to hardcoded features")
            csv_path = None
        else:
            try:
                df = pd.read_csv(csv_path)
                
                # Try different possible column names
                if 'feature' in df.columns:
                    feature_col = 'feature'
                elif 'Feature' in df.columns:
                    feature_col = 'Feature'
                else:
                    # Use first column
                    feature_col = df.columns[0]
                
                features = df[feature_col].dropna().tolist()
                print(f"✓ Loaded {len(features)} features from CSV: {csv_path}")
                print(f"  Features in order: {features[:5]}..." if len(features) > 5 else f"  Features: {features}")
                return features
            except Exception as e:
                print(f"❌ Error loading CSV: {e}")
                print("Falling back to hardcoded features")
                csv_path = None
    
    # Default hardcoded features (excluding metadata and targets)
    specified_features = [
        # Damage features
        'Total_Gold_Difference_Last_Time_Frame',
        'Total_Xp_Difference_Last_Time_Frame',
        'Total_Minions_Killed_Difference',
        'Total_Jungle_Minions_Killed_Difference',
        'Total_Kill_Difference',
        'Total_Assist_Difference',
        'Elite_Monster_Killed_Difference',
        'Buildings_Taken_Difference',
        'Magic_Damage_Done_Diff',
        'Magic_Damage_Done_To_Champions_Diff', 
        'Magic_Damage_Taken_Diff',
        'Physical_Damage_Done_Diff',
        'Physical_Damage_Done_To_Champions_Diff',
        'Physical_Damage_Taken_Diff',
        'Total_Damage_Done_Diff',
        'Total_Damage_Done_To_Champions_Diff',
        'Total_Damage_Taken_Diff',
        'True_Damage_Done_Diff',
        'True_Damage_Done_To_Champions_Diff',
        'True_Damage_Taken_Diff',
        
        # Game stats
        'Total_Kill_Difference',
        'Total_Assist_Difference',
        'Total_Ward_Placed_Difference',
        'Total_Ward_Killed_Difference',
        'Time_Enemy_Spent_Controlled_Difference',
        
        # Spatial features
        'CentroidDist',
        'MinInterTeamDist', 
        'EngagedDiff',
        'FrontlineOverlap',
        'RadialVelocityDiff',
        
        # Team scores
        'Blue_Team_Offensive_Score',
        'Blue_Team_Defensive_Score',
        'Blue_Team_Overall_Score',
        'Red_Team_Offensive_Score',
        'Red_Team_Defensive_Score', 
        'Red_Team_Overall_Score',
        'Team_Offensive_Score_Diff',
        'Team_Defensive_Score_Diff',
        'Team_Overall_Score_Diff'
    ]
    
    return specified_features


def main(
    feature_list=None,
    num_epochs=50,
    patience=5,
    batch_size=32,
    sequence_length=15,
    learning_rate=0.001,
    hidden_size=128,
    num_layers=2,
    dropout=0.4,
    model_save_dir=None,
    checkpoint_name="lstm_model.pth",
    best_checkpoint_name="lstm_model_best.pth",
    curve_save_path=None
):
    """
    Main function to train LSTM model
    
    Args:
        feature_list: Can be:
            - None: Auto-detect all features from train.parquet
            - List of feature names: Use these features
            - CSV path string (e.g., "data/Feature_Selection_RFE/best_features_rfe.csv"): Load from CSV
            - 'specified': Use hardcoded specified features
        num_epochs: Number of training epochs
        patience: Early stopping patience (number of epochs with no improvement)
        batch_size: Batch size for training
        sequence_length: Length of input sequences
        learning_rate: Learning rate for optimizer
        hidden_size: LSTM hidden state size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        model_save_dir: Directory to save models (default: models/)
        checkpoint_name: Name for final model checkpoint (default: lstm_model.pth)
        best_checkpoint_name: Name for best model checkpoint (default: lstm_model_best.pth)
        curve_save_path: Path to save training curves plot (default: models/training_curves.png)
    """
    print("🚀 LSTM Training for League of Legends Gold Difference Prediction")
    print("=" * 60)
    
    try:
        # Select features and create data loaders (combined function)
        train_loader, val_loader, test_loader, input_size, feature_cols = prepare_data_loaders(
            feature_list, batch_size=batch_size, sequence_length=sequence_length
        )
        
        if feature_cols is None or len(feature_cols) == 0:
            print("❌ Failed to select features!")
            return
        
        # Get scaler from train_loader dataset for saving in checkpoint
        train_dataset = train_loader.dataset
        feature_scaler = train_dataset.scaler if hasattr(train_dataset, 'scaler') else None
        
        # Train model
        print("\nStarting training...")
        model, train_losses, val_losses, test_losses = train_lstm(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            input_size=input_size,
            num_epochs=num_epochs,
            feature_list=feature_cols,
            patience=patience,
            batch_size=batch_size,
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            model_save_dir=model_save_dir,
            checkpoint_name=checkpoint_name,
            best_checkpoint_name=best_checkpoint_name,
            curve_save_path=curve_save_path,
            feature_scaler=feature_scaler  # Pass scaler to save in checkpoint
        )
        
        # Calculate final RMSE
        final_train_rmse = np.sqrt(train_losses[-1])
        final_val_rmse = np.sqrt(val_losses[-1])
        final_test_rmse = np.sqrt(test_losses[-1])
        
        print("\n✅ Training completed successfully!")
        print(f"Final Train Loss: {train_losses[-1]:.6f} (RMSE: {final_train_rmse:.6f})")
        print(f"Final Val Loss: {val_losses[-1]:.6f} (RMSE: {final_val_rmse:.6f})")
        print(f"Final Test Loss: {test_losses[-1]:.6f} (RMSE: {final_test_rmse:.6f})")
        print(f"Model saved to: models/lstm_model.pth")
        print(f"Training curves saved to: models/training_curves.png")
        
        # Show training summary
        print(f"\n📊 Training Summary:")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Input features: {input_size}")
        print(f"  - Target: Total_Gold_Difference")
        print(f"  - Sequence length: {sequence_length}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Num layers: {num_layers}")
        print(f"  - Dropout: {dropout}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Patience: {patience}")
        print(f"  - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"  - Final RMSE: {final_test_rmse:.6f} (in gold units)")
        if model_save_dir:
            print(f"  - Model saved to: {model_save_dir}/")
        else:
            print(f"  - Model saved to: models/")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please make sure you have:")
        print("1. Run data_featuring_score.py to create featured_data_with_scores.parquet")
        print("2. Run dataloader.py to create data splits")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train LSTM model for League of Legends Gold Difference Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with CSV feature list
  python lstm_model.py --feature_list data/Feature_Selection_RFE/best_features_rfe.csv
  
  # Train with all features, custom epochs and patience
  python lstm_model.py --feature_list None --num_epochs 100 --patience 10
  
  # Train with custom model architecture and save location
  python lstm_model.py --feature_list specified --hidden_size 256 --num_layers 3 \\
                       --model_save_dir models/custom --checkpoint_name my_model.pth
        """
    )
    
    # Feature selection
    parser.add_argument(
        '--feature_list', type=str, default=None,
        help='Feature list: CSV path, "specified", "None" for auto-detect, or comma-separated list'
    )
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience - epochs without improvement (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--sequence_length', type=int, default=15,
                       help='Sequence length for LSTM (default: 15)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    # Model architecture
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='LSTM hidden size (default: 128)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate (default: 0.4)')
    
    # Save paths
    parser.add_argument('--model_save_dir', type=str, default=None,
                       help='Directory to save models (default: models/)')
    parser.add_argument('--checkpoint_name', type=str, default='lstm_model.pth',
                       help='Final model checkpoint name (default: lstm_model.pth)')
    parser.add_argument('--best_checkpoint_name', type=str, default='lstm_model_best.pth',
                       help='Best model checkpoint name (default: lstm_model_best.pth)')
    parser.add_argument('--curve_save_path', type=str, default=None,
                       help='Path to save training curves plot (default: models/training_curves.png)')
    
    args = parser.parse_args()
    
    # Parse feature_list
    feature_list = args.feature_list
    if feature_list is not None:
        if feature_list.lower() == 'none':
            feature_list = None
        elif feature_list.lower() == 'specified':
            feature_list = 'specified'
        elif ',' in feature_list and not feature_list.endswith('.csv'):
            # Comma-separated feature list
            feature_list = [f.strip() for f in feature_list.split(',')]
        # Otherwise, treat as CSV path or let main handle it
    
    # Run training
    main(
        feature_list=feature_list,
        num_epochs=args.num_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_save_dir=args.model_save_dir,
        checkpoint_name=args.checkpoint_name,
        best_checkpoint_name=args.best_checkpoint_name,
        curve_save_path=args.curve_save_path
    )
