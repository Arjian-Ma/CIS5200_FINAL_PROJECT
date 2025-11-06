import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

print("\n" + "="*60)
print("LOADING DATA")
print("="*60)
print("Loading data from parquet file...")
df = pd.read_parquet("Data/processed/featured_data.parquet")

print(f"✓ Full data loaded: {df.shape}")

# Remove data leakage features
leakage_features = [
    'Total_Xp_Difference',
    'Total_Gold_Difference_Last_Time_Frame',
    'Total_Minions_Killed_Difference',
    'Total_Jungle_Minions_Killed_Difference',
    'Total_Kill_Difference',
    'Total_Assist_Difference',
    'Elite_Monster_Killed_Difference',
    'Buildings_Taken_Difference',
    'Total_Xp_Difference_Last_Time_Frame',
]

print(f"\nRemoving {len(leakage_features)} data leakage features...")
df_clean = df.drop(columns=[f for f in leakage_features if f in df.columns], errors='ignore')
print(f"Cleaned data shape: {df_clean.shape}")

# Split data by match_id (70% train, 15% val, 15% test)
unique_matches = df_clean['match_id'].unique()
n_matches = len(unique_matches)
print(f"\nTotal unique matches: {n_matches}")

np.random.seed(42)
shuffled_matches = np.random.permutation(unique_matches)

n_train = int(n_matches * 0.7)
n_val = int(n_matches * 0.15)

train_matches = shuffled_matches[:n_train]
val_matches = shuffled_matches[n_train:n_train + n_val]
test_matches = shuffled_matches[n_train + n_val:]

train_df = df_clean[df_clean['match_id'].isin(train_matches)].copy()
val_df = df_clean[df_clean['match_id'].isin(val_matches)].copy()
test_df = df_clean[df_clean['match_id'].isin(test_matches)].copy()

print(f"\nData split:")
print(f"Train: {train_df.shape} ({len(train_matches)} matches)")
print(f"Val: {val_df.shape} ({len(val_matches)} matches)")
print(f"Test: {test_df.shape} ({len(test_matches)} matches)")

# Prepare features and labels
label_col = 'Total_Gold_Difference'
exclude_cols = ['match_id', 'frame_idx', 'timestamp', label_col]
feature_columns = [col for col in df_clean.columns if col not in exclude_cols]

print(f"\nNumber of features: {len(feature_columns)}")

# Prepare X and y
X_train = train_df[feature_columns].values
y_train = train_df[label_col].values

X_val = val_df[feature_columns].values
y_val = val_df[label_col].values

X_test = test_df[feature_columns].values
y_test = test_df[label_col].values

# Handle any NaN values
X_train = np.nan_to_num(X_train, nan=0.0)
X_val = np.nan_to_num(X_val, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

print(f"\nTrain samples: {len(X_train)}")
print(f"Val samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Train Gradient Boosting model with loss tracking
print("\n" + "="*60)
print("TRAINING GRADIENT BOOSTING REGRESSOR")
print("="*60)

n_estimators = 200
gb_model = GradientBoostingRegressor(
    n_estimators=n_estimators,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,
    random_state=42,
    verbose=0,  # Turn off verbose to use tqdm
    warm_start=True  # Allow incremental training for loss tracking
)

# Track losses during training
train_losses = []
val_losses = []

print(f"Training with {n_estimators} estimators...")

# Train incrementally to track loss
for i in tqdm(range(1, n_estimators + 1), desc="Training", unit="estimator"):
    gb_model.set_params(n_estimators=i)
    gb_model.fit(X_train, y_train)
    
    # Calculate losses every 10 estimators for efficiency
    if i % 10 == 0 or i == n_estimators:
        train_pred = gb_model.predict(X_train)
        val_pred = gb_model.predict(X_val)
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        train_losses.append((i, np.sqrt(train_mse)))
        val_losses.append((i, np.sqrt(val_mse)))

print("\n✓ Training complete!")

# Make final predictions
y_train_pred = gb_model.predict(X_train)
y_val_pred = gb_model.predict(X_val)
y_test_pred = gb_model.predict(X_test)

# Calculate metrics for all sets
def print_metrics(y_true, y_pred, set_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"{set_name} METRICS")
    print(f"{'='*60}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return rmse

train_rmse = print_metrics(y_train, y_train_pred, "TRAINING SET")
val_rmse = print_metrics(y_val, y_val_pred, "VALIDATION SET")
test_rmse = print_metrics(y_test, y_test_pred, "TEST SET")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n{'='*60}")
print("TOP 15 MOST IMPORTANT FEATURES")
print(f"{'='*60}")
print(feature_importance.head(15).to_string(index=False))

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Training and Validation Loss over iterations
plt.subplot(2, 3, 1)
train_iters, train_loss_vals = zip(*train_losses)
val_iters, val_loss_vals = zip(*val_losses)
plt.plot(train_iters, train_loss_vals, 'b-', label='Training Loss', linewidth=2)
plt.plot(val_iters, val_loss_vals, 'r-', label='Validation Loss', linewidth=2)
plt.axhline(y=test_rmse, color='g', linestyle='--', linewidth=2, label=f'Test RMSE: {test_rmse:.2f}')
plt.xlabel('Number of Estimators', fontsize=12)
plt.ylabel('RMSE (Gold Difference)', fontsize=12)
plt.title('Gradient Boosting - Training and Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Add text annotation for final values
plt.text(0.02, 0.98, f'Final Train RMSE: {train_rmse:.2f}\nFinal Val RMSE: {val_rmse:.2f}\nTest RMSE: {test_rmse:.2f}',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Training Set - Predicted vs True
plt.subplot(2, 3, 2)
sample_size = min(2000, len(y_train))
sample_indices = np.random.choice(len(y_train), sample_size, replace=False)
plt.scatter(y_train[sample_indices], y_train_pred[sample_indices], alpha=0.3, edgecolors='k', linewidth=0.3, s=20)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('True Total Gold Difference', fontsize=12)
plt.ylabel('Predicted Total Gold Difference', fontsize=12)
plt.title(f'Training Set: Predicted vs True ({sample_size} samples)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Validation Set - Predicted vs True
plt.subplot(2, 3, 3)
plt.scatter(y_val, y_val_pred, alpha=0.5, edgecolors='k', linewidth=0.5, s=30, color='orange')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('True Total Gold Difference', fontsize=12)
plt.ylabel('Predicted Total Gold Difference', fontsize=12)
plt.title('Validation Set: Predicted vs True Values', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Test Set - Predicted vs True
plt.subplot(2, 3, 4)
plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', linewidth=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('True Total Gold Difference', fontsize=12)
plt.ylabel('Predicted Total Gold Difference', fontsize=12)
plt.title('Test Set: Predicted vs True Values', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Residuals for test set
plt.subplot(2, 3, 5)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Total Gold Difference', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Test Set: Residual Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 6: Top 15 Feature Importances
plt.subplot(2, 3, 6)
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue', edgecolor='k')
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=9)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/gradient_boosting_results.png', dpi=300, bbox_inches='tight')
print(f"\n{'='*60}")
print("✓ Visualization saved as 'results/gradient_boosting_results.png'")
print(f"{'='*60}")

# Create detailed feature importance plot
plt.figure(figsize=(12, 10))
top_features = feature_importance.head(25)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue', edgecolor='k', linewidth=0.5)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 25 Feature Importances in Gradient Boosting', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('results/gradient_boosting_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance plot saved as 'results/gradient_boosting_feature_importance.png'")
print(f"{'='*60}")

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
