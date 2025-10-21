import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# Load the data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'Data', 'featured_data_with_scores copy.csv')
df = pd.read_csv(data_path)

# Identify the last match_id for test set
unique_matches = df['match_id'].unique()
last_match_id = unique_matches[-1]

print(f"Total matches: {len(unique_matches)}")
print(f"Last match_id (test set): {last_match_id}")

# Split data into train and test
train_df = df[df['match_id'] != last_match_id]
test_df = df[df['match_id'] == last_match_id]

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Find the column index for Total_Gold_Difference
label_col = 'Total_Gold_Difference'
all_columns = df.columns.tolist()
label_idx = all_columns.index(label_col)

# Features are all columns after the label
feature_columns = all_columns[label_idx + 1:]
print(f"\nNumber of features: {len(feature_columns)}")

# Prepare X and y for training and testing
X_train = train_df[feature_columns].values
y_train = train_df[label_col].values

X_test = test_df[feature_columns].values
y_test = test_df[label_col].values

# Handle any NaN values by replacing with 0
X_train = np.nan_to_num(X_train, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

# Train Gradient Boosting model
print("\nTraining Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,
    random_state=42,
    verbose=1
)
gb_model.fit(X_train, y_train)

# Make predictions
y_train_pred = gb_model.predict(X_train)
y_test_pred = gb_model.predict(X_test)

# Calculate errors for training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("\n" + "="*60)
print("TRAINING SET METRICS")
print("="*60)
print(f"Mean Squared Error (MSE): {train_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {train_rmse:.2f}")
print(f"Mean Absolute Error (MAE): {train_mae:.2f}")
print(f"R² Score: {train_r2:.4f}")

# Calculate errors for test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "="*60)
print("TEST SET METRICS")
print("="*60)
print(f"Mean Squared Error (MSE): {test_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {test_rmse:.2f}")
print(f"Mean Absolute Error (MAE): {test_mae:.2f}")
print(f"R² Score: {test_r2:.4f}")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*60)
print(feature_importance.head(10).to_string(index=False))

# Create visualization for training and test set predictions
fig = plt.figure(figsize=(12, 10))

# Training Set Plots
# Plot 1: Predicted vs True values for training set
plt.subplot(2, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, edgecolors='k', linewidth=0.5, s=30)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('True Total Gold Difference', fontsize=12)
plt.ylabel('Predicted Total Gold Difference', fontsize=12)
plt.title('Training Set: Predicted vs True Values', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Predicted and True values over training samples (sampled for visibility)
plt.subplot(2, 2, 2)
# Sample indices for better visualization if training set is large
sample_size = min(100, len(y_train))
sample_indices = np.linspace(0, len(y_train)-1, sample_size, dtype=int)
plt.plot(sample_indices, y_train[sample_indices], 'o-', label='True Values', alpha=0.7, markersize=4)
plt.plot(sample_indices, y_train_pred[sample_indices], 's-', label='Predicted Values', alpha=0.7, markersize=4)
plt.xlabel('Training Sample Index', fontsize=12)
plt.ylabel('Total Gold Difference', fontsize=12)
plt.title(f'Training Set: True vs Predicted (Sampled {sample_size} points)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Test Set Plots
# Plot 3: Predicted vs True values for test set
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('True Total Gold Difference', fontsize=12)
plt.ylabel('Predicted Total Gold Difference', fontsize=12)
plt.title('Test Set: Predicted vs True Values', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Predicted and True values over test samples
plt.subplot(2, 2, 4)
test_indices = np.arange(len(y_test))
plt.plot(test_indices, y_test, 'o-', label='True Values', alpha=0.7, markersize=4)
plt.plot(test_indices, y_test_pred, 's-', label='Predicted Values', alpha=0.7, markersize=4)
plt.xlabel('Test Sample Index', fontsize=12)
plt.ylabel('Total Gold Difference', fontsize=12)
plt.title('Test Set: True vs Predicted Over Time', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_boosting_results.png', dpi=300, bbox_inches='tight')
print("\n" + "="*60)
print("Visualization saved as 'gradient_boosting_results.png'")
print("="*60)

# Create feature importance plot
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 20 Feature Importances in Gradient Boosting', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('gradient_boosting_feature_importance.png', dpi=300, bbox_inches='tight')
print("Feature importance plot saved as 'gradient_boosting_feature_importance.png'")
print("="*60)

plt.show()

