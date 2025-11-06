import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os
warnings.filterwarnings('ignore')

# Load the data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'Data', 'featured_data_with_scores copy.csv')
df = pd.read_csv(data_path)

# Find the column index for Total_Gold_Difference
label_col = 'Total_Gold_Difference'
all_columns = df.columns.tolist()
label_idx = all_columns.index(label_col)

# Features are all columns after the label EXCEPT Total_Gold_Difference_Last_Time_Frame
feature_columns = all_columns[label_idx + 1:]

# Remove Total_Gold_Difference_Last_Time_Frame and Total_Xp_Difference_Last_Time_Frame
# since RNN will handle temporal dependencies
columns_to_exclude = ['Total_Gold_Difference_Last_Time_Frame', 'Total_Xp_Difference_Last_Time_Frame']
feature_columns = [col for col in feature_columns if col not in columns_to_exclude]

print(f"Number of features (excluding temporal lag features): {len(feature_columns)}")
print(f"Excluded columns: {columns_to_exclude}")

# Identify the last match_id for test set
unique_matches = df['match_id'].unique()
last_match_id = unique_matches[-1]

print(f"\nTotal matches: {len(unique_matches)}")
print(f"Last match_id (test set): {last_match_id}")
print(f"Training matches: {len(unique_matches) - 1}")

# Split data by match_id
train_matches = unique_matches[:-1]
test_match = [last_match_id]

print(f"\nTraining matches: {len(train_matches)}")
print(f"Test matches: {len(test_match)}")

# Prepare sequences for each game
def prepare_sequences(match_ids, df, feature_columns, label_col):
    """
    Prepare sequences where each game is a separate sequence.
    Returns X (3D array: samples x timesteps x features) and y (2D array: samples x timesteps)
    """
    sequences_X = []
    sequences_y = []
    
    for match_id in match_ids:
        match_data = df[df['match_id'] == match_id].sort_values('frame_idx')
        
        # Extract features and labels for this match
        X_match = match_data[feature_columns].values
        y_match = match_data[label_col].values
        
        # Handle NaN values
        X_match = np.nan_to_num(X_match, nan=0.0)
        
        sequences_X.append(X_match)
        sequences_y.append(y_match)
    
    return sequences_X, sequences_y

# Prepare training and test sequences
train_sequences_X, train_sequences_y = prepare_sequences(train_matches, df, feature_columns, label_col)
test_sequences_X, test_sequences_y = prepare_sequences(test_match, df, feature_columns, label_col)

# Get sequence lengths
train_lengths = [len(seq) for seq in train_sequences_X]
test_lengths = [len(seq) for seq in test_sequences_X]

print(f"\nTraining sequence lengths - Min: {min(train_lengths)}, Max: {max(train_lengths)}, Mean: {np.mean(train_lengths):.1f}")
print(f"Test sequence lengths - Min: {min(test_lengths)}, Max: {max(test_lengths)}, Mean: {np.mean(test_lengths):.1f}")

# Pad sequences to the same length
max_length = max(max(train_lengths), max(test_lengths))
print(f"\nPadding all sequences to length: {max_length}")

def pad_sequences_custom(sequences, max_length, padding_value=0):
    """Pad sequences to max_length"""
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            padding = np.full((max_length - len(seq), seq.shape[1]), padding_value)
            padded_seq = np.vstack([seq, padding])
        else:
            padded_seq = seq[:max_length]
        padded.append(padded_seq)
    return np.array(padded)

def pad_labels(labels, max_length, padding_value=0):
    """Pad label sequences to max_length"""
    padded = []
    for label in labels:
        if len(label) < max_length:
            padding = np.full(max_length - len(label), padding_value)
            padded_label = np.concatenate([label, padding])
        else:
            padded_label = label[:max_length]
        padded.append(padded_label)
    return np.array(padded)

# Pad sequences
X_train = pad_sequences_custom(train_sequences_X, max_length)
y_train = pad_labels(train_sequences_y, max_length)

X_test = pad_sequences_custom(test_sequences_X, max_length)
y_test = pad_labels(test_sequences_y, max_length)

print(f"\nX_train shape: {X_train.shape}")  # (num_games, max_length, num_features)
print(f"y_train shape: {y_train.shape}")    # (num_games, max_length)
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Normalize features (fit on training data only)
# Reshape for scaling
n_games_train, n_timesteps, n_features = X_train.shape
X_train_reshaped = X_train.reshape(-1, n_features)
X_test_reshaped = X_test.reshape(-1, n_features)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

# Reshape back to 3D
X_train_scaled = X_train_scaled.reshape(n_games_train, n_timesteps, n_features)
X_test_scaled = X_test_scaled.reshape(X_test.shape[0], n_timesteps, n_features)

# Build LSTM model
print("\nBuilding LSTM model...")
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(max_length, n_features)),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=True),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(model.summary())

# Train the model
print("\nTraining LSTM model...")
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=8,
    verbose=1,
    callbacks=[early_stopping]
)

# Make predictions
y_train_pred = model.predict(X_train_scaled, verbose=0)
y_test_pred = model.predict(X_test_scaled, verbose=0)

# Flatten predictions and labels for metrics (only non-padded values)
# We'll use actual sequence lengths to evaluate
def calculate_metrics_for_sequences(y_true_sequences, y_pred_sequences, sequence_lengths):
    """Calculate metrics only on actual sequence data (not padding)"""
    all_true = []
    all_pred = []
    
    for i, length in enumerate(sequence_lengths):
        all_true.extend(y_true_sequences[i][:length])
        all_pred.extend(y_pred_sequences[i][:length, 0])
    
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    
    mse = mean_squared_error(all_true, all_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_true, all_pred)
    r2 = r2_score(all_true, all_pred)
    
    return mse, rmse, mae, r2, all_true, all_pred

# Calculate metrics for training set
train_mse, train_rmse, train_mae, train_r2, y_train_flat, y_train_pred_flat = \
    calculate_metrics_for_sequences(y_train, y_train_pred, train_lengths)

print("\n" + "="*60)
print("TRAINING SET METRICS")
print("="*60)
print(f"Mean Squared Error (MSE): {train_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {train_rmse:.2f}")
print(f"Mean Absolute Error (MAE): {train_mae:.2f}")
print(f"R² Score: {train_r2:.4f}")

# Calculate metrics for test set
test_mse, test_rmse, test_mae, test_r2, y_test_flat, y_test_pred_flat = \
    calculate_metrics_for_sequences(y_test, y_test_pred, test_lengths)

print("\n" + "="*60)
print("TEST SET METRICS")
print("="*60)
print(f"Mean Squared Error (MSE): {test_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {test_rmse:.2f}")
print(f"Mean Absolute Error (MAE): {test_mae:.2f}")
print(f"R² Score: {test_r2:.4f}")

# Create visualizations
fig = plt.figure(figsize=(14, 10))

# Training Set Plots
plt.subplot(2, 3, 1)
plt.scatter(y_train_flat, y_train_pred_flat, alpha=0.5, edgecolors='k', linewidth=0.5, s=30)
plt.plot([y_train_flat.min(), y_train_flat.max()], 
         [y_train_flat.min(), y_train_flat.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('True Total Gold Difference', fontsize=12)
plt.ylabel('Predicted Total Gold Difference', fontsize=12)
plt.title('Training Set: Predicted vs True Values', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Sample training predictions over time
plt.subplot(2, 3, 2)
sample_size = min(100, len(y_train_flat))
sample_indices = np.linspace(0, len(y_train_flat)-1, sample_size, dtype=int)
plt.plot(sample_indices, y_train_flat[sample_indices], 'o-', label='True Values', alpha=0.7, markersize=4)
plt.plot(sample_indices, y_train_pred_flat[sample_indices], 's-', label='Predicted Values', alpha=0.7, markersize=4)
plt.xlabel('Training Sample Index', fontsize=12)
plt.ylabel('Total Gold Difference', fontsize=12)
plt.title(f'Training: True vs Predicted (Sampled {sample_size} points)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Test Set Plots
plt.subplot(2, 3, 4)
plt.scatter(y_test_flat, y_test_pred_flat, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.plot([y_test_flat.min(), y_test_flat.max()], 
         [y_test_flat.min(), y_test_flat.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('True Total Gold Difference', fontsize=12)
plt.ylabel('Predicted Total Gold Difference', fontsize=12)
plt.title('Test Set: Predicted vs True Values', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Test predictions over time
plt.subplot(2, 3, 5)
test_indices = np.arange(len(y_test_flat))
plt.plot(test_indices, y_test_flat, 'o-', label='True Values', alpha=0.7, markersize=4)
plt.plot(test_indices, y_test_pred_flat, 's-', label='Predicted Values', alpha=0.7, markersize=4)
plt.xlabel('Test Sample Index', fontsize=12)
plt.ylabel('Total Gold Difference', fontsize=12)
plt.title('Test Set: True vs Predicted Over Time', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Training loss curve
plt.subplot(2, 3, 3)
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Residuals plot for test set
plt.subplot(2, 3, 6)
residuals = y_test_flat - y_test_pred_flat
plt.scatter(y_test_pred_flat, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Total Gold Difference', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Test Set: Residual Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_lstm_results.png', dpi=300, bbox_inches='tight')
print("\n" + "="*60)
print("Visualization saved as 'rnn_lstm_results.png'")
print("="*60)

plt.show()

