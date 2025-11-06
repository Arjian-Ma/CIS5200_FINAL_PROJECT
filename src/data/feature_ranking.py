#!/usr/bin/env python3
"""
Recursive Feature Elimination (RFE) with XGBoost
- Iteratively removes least important features
- Tracks performance at each step
- Combines x_position and y_position into single position features
- Aggregates player-specific features for analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RECURSIVE FEATURE ELIMINATION WITH GRADIENT BOOSTING")
print("="*80)

# Load base data
print("\n[Step 1/7] Loading base feature data...")
df = pd.read_parquet("Data/processed/featured_data.parquet")
print(f"  ✓ Loaded base data: {df.shape}")

# Load spatial features
print("\n[Step 2/7] Loading spatial features...")
spatial_file = "Data/processed/featured_data_with_scores.parquet"
if os.path.exists(spatial_file):
    df_spatial = pd.read_parquet(spatial_file)
    print(f"  ✓ Loaded spatial data: {df_spatial.shape}")
    
    # Extract spatial feature columns
    spatial_features = []
    for col in df_spatial.columns:
        if any(keyword in col for keyword in ['Spatial', 'CentroidDist', 'MinInterTeamDist', 
                                                'EngageDiff', 'FrontlineOverlap', 'RadialVelocityDiff']):
            spatial_features.append(col)
    
    print(f"  ✓ Found {len(spatial_features)} spatial feature columns:")
    for sf in spatial_features[:5]:  # Show first 5
        print(f"      - {sf}")
    if len(spatial_features) > 5:
        print(f"      ... and {len(spatial_features) - 5} more")
    
    # Merge spatial features with base data
    print("  Merging spatial features...")
    merge_keys = ['match_id', 'frame_idx']
    df = df.merge(df_spatial[merge_keys + spatial_features], on=merge_keys, how='left')
    print(f"  ✓ Merged data: {df.shape}")
else:
    print(f"  ⚠ Warning: Spatial feature file not found")
    print(f"  Continuing with base features only...")

print(f"  ✓ Total data shape: {df.shape}")

# Remove specific features as requested
print("\n[Step 3/7] Removing specific features...")
features_to_remove = [
    'Total_Gold_Difference_Last_Time_Frame',
    'Total_Xp_Difference',
    'Total_Xp_Difference_Last_Time_Frame',
]

for feat in features_to_remove:
    if feat in df.columns:
        print(f"  - {feat}")

df_clean = df.drop(columns=[f for f in features_to_remove if f in df.columns], errors='ignore')
print(f"  ✓ Cleaned data: {df_clean.shape}")

# Prepare features and target
print("\n[Step 4/7] Preparing features and target...")
metadata_cols = ['match_id', 'frame_idx', 'timestamp']
target_col = 'Total_Gold_Difference'

feature_cols = [col for col in df_clean.columns if col not in metadata_cols + [target_col]]
print(f"  ✓ Total features: {len(feature_cols)}")

X = df_clean[feature_cols].fillna(0)
y = df_clean[target_col]

# Split data
print("\n[Step 5/7] Splitting data (80/20 train/test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  ✓ Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================================
# RECURSIVE FEATURE ELIMINATION (RFE)
# ============================================================================

print("\n[Step 6/7] " + "="*72)
print("STARTING RECURSIVE FEATURE ELIMINATION")
print("="*80)

# RFE Parameters
drop_percentage = 0.15  # Drop 15% of features each iteration
min_features = 20  # Stop when we reach this many features
n_estimators = 50  # XGBoost rounds per iteration

# Track RFE progress
rfe_history = []
current_features = list(X.columns)

print(f"\nRFE Configuration:")
print(f"  Initial features: {len(current_features)}")
print(f"  Drop percentage per iteration: {drop_percentage*100:.0f}%")
print(f"  Minimum features to keep: {min_features}")
print(f"  XGBoost estimators: {n_estimators}")
print(f"  Training samples: {X_train.shape[0]:,}")
print(f"  Test samples: {X_test.shape[0]:,}")

iteration = 0
best_test_rmse = float('inf')
best_iteration = 0
best_features = current_features.copy()

print("\n" + "-"*80)
print(f"{'Iter':<6} {'Features':<10} {'Train RMSE':<15} {'Test RMSE':<15} {'Test R²':<12} {'Action':<20}")
print("-"*80)

while len(current_features) > min_features:
    iteration += 1
    
    print(f"\n[Iteration {iteration}] Training with {len(current_features)} features...", end=' ', flush=True)
    
    # Get current feature subset
    X_train_subset = X_train[current_features]
    X_test_subset = X_test[current_features]
    
    # Train Gradient Boosting model
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train_subset, y_train)
    print("✓", flush=True)
    
    # Evaluate
    y_train_pred = model.predict(X_train_subset)
    y_test_pred = model.predict(X_test_subset)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Track best model
    if test_rmse < best_test_rmse:
        best_test_rmse = test_rmse
        best_iteration = iteration
        best_features = current_features.copy()
        action = "✓ New best"
    else:
        action = ""
    
    # Store history
    rfe_history.append({
        'iteration': iteration,
        'n_features': len(current_features),
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'features': current_features.copy()
    })
    
    print(f"{iteration:<6} {len(current_features):<10} {train_rmse:<15.2f} {test_rmse:<15.2f} {test_r2:<12.4f} {action:<20}")
    
    # Get feature importances and aggregate by base feature
    feature_importance = model.feature_importances_
    importance_df_iter = pd.DataFrame({
        'feature': current_features,
        'importance': feature_importance
    })
    
    # Aggregate importance by base feature (combine player-specific features)
    base_importance = defaultdict(lambda: {'total_importance': 0, 'features': []})
    
    for _, row in importance_df_iter.iterrows():
        feat = row['feature']
        imp = row['importance']
        
        # Determine base feature name
        if feat.startswith('Player') and '_' in feat:
            parts = feat.split('_', 1)
            if len(parts) == 2:
                base = parts[1]
                # Combine x_position and y_position
                if 'x_position' in base or 'y_position' in base:
                    base = base.replace('x_position', 'position').replace('y_position', 'position')
            else:
                base = feat
        else:
            # Handle non-player features
            if 'x_position' in feat.lower() or 'y_position' in feat.lower():
                base = feat.replace('x_position', 'position').replace('y_position', 'position').replace('X_position', 'position').replace('Y_position', 'position')
            else:
                base = feat
        
        base_importance[base]['total_importance'] += imp
        base_importance[base]['features'].append(feat)
    
    # Convert to sorted list by total importance
    base_sorted = sorted(base_importance.items(), key=lambda x: x[1]['total_importance'], reverse=False)
    
    # Drop least important base features (which drops ALL related player features)
    target_drop_count = max(1, int(len(current_features) * drop_percentage))
    features_to_drop = []
    
    for base_name, base_data in base_sorted:
        if len(features_to_drop) + len(base_data['features']) <= target_drop_count:
            features_to_drop.extend(base_data['features'])
        else:
            # Check if adding this base feature would go below min_features
            if len(current_features) - len(features_to_drop) - len(base_data['features']) >= min_features:
                features_to_drop.extend(base_data['features'])
            break
        
        if len(current_features) - len(features_to_drop) <= min_features:
            break
    
    if len(features_to_drop) == 0:
        break
    
    # Log what's being dropped
    # Group dropped features by base
    dropped_by_base = defaultdict(list)
    for feat in features_to_drop:
        if feat.startswith('Player') and '_' in feat:
            parts = feat.split('_', 1)
            if len(parts) == 2:
                base = parts[1]
                if 'x_position' in base or 'y_position' in base:
                    base = base.replace('x_position', 'position').replace('y_position', 'position')
            else:
                base = feat
        else:
            if 'x_position' in feat.lower() or 'y_position' in feat.lower():
                base = feat.replace('x_position', 'position').replace('y_position', 'position').replace('X_position', 'position').replace('Y_position', 'position')
            else:
                base = feat
        dropped_by_base[base].append(feat)
    
    print(f"[Iteration {iteration}] Dropping {len(features_to_drop)} features ({len(dropped_by_base)} base feature groups):")
    for base, feats in list(dropped_by_base.items())[:3]:  # Show first 3
        print(f"    - {base}: {len(feats)} features")
    if len(dropped_by_base) > 3:
        print(f"    ... and {len(dropped_by_base) - 3} more base groups")
    
    # Update current features
    current_features = [f for f in current_features if f not in features_to_drop]

print("-"*80)
print(f"\n✓ RFE Complete!")
print(f"  Best iteration: {best_iteration}")
print(f"  Best test RMSE: {best_test_rmse:.2f}")
print(f"  Best feature count: {len(best_features)}")

# Train final model with best features
print(f"\n[Step 7/7] " + "="*72)
print("TRAINING FINAL MODEL WITH BEST FEATURES")
print("="*80)

X_train_best = X_train[best_features]
X_test_best = X_test[best_features]

model = GradientBoostingRegressor(
    n_estimators=100,  # More estimators for final model
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    verbose=0
)

print(f"\nTraining final model with {len(best_features)} features and 100 estimators...")
print("  Progress: ", end='', flush=True)

# Train with simple progress indicator
for i in range(0, 101, 10):
    if i == 0:
        continue
    temp_model = GradientBoostingRegressor(
        n_estimators=i,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    temp_model.fit(X_train_best, y_train)
    print(f"{i}% ", end='', flush=True)

model.fit(X_train_best, y_train)
print("✓ Complete!")

# Get feature importance from best model
print("\nExtracting feature importance from best model...")
feature_importance = model.feature_importances_
feature_names = best_features

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"✓ Extracted importance for {len(feature_names)} features")

# Aggregate player-specific features
print("\nAggregating player-specific features...")

aggregated_importance = defaultdict(float)
feature_to_base_mapping = {}  # Maps individual features to their base name

for idx, row in importance_df.iterrows():
    feature_name = row['feature']
    importance_val = row['importance']
    
    # Check if it's a player-specific feature (e.g., Player1_kills, Player2_kills)
    if feature_name.startswith('Player') and '_' in feature_name:
        # Extract the base feature name (e.g., "kills" from "Player1_kills")
        parts = feature_name.split('_', 1)
        if len(parts) == 2:
            player_num = parts[0]  # e.g., "Player1"
            base_feature = parts[1]  # e.g., "kills"
            
            # Special handling for x_position and y_position
            if 'x_position' in base_feature or 'y_position' in base_feature:
                # Combine x and y position under "position"
                position_base = base_feature.replace('x_position', 'position').replace('y_position', 'position')
                aggregated_importance[position_base] += importance_val
                feature_to_base_mapping[feature_name] = position_base
            else:
                # Aggregate under the base feature name
                aggregated_importance[base_feature] += importance_val
                feature_to_base_mapping[feature_name] = base_feature
        else:
            aggregated_importance[feature_name] += importance_val
            feature_to_base_mapping[feature_name] = feature_name
    else:
        # Non-player-specific features
        # Also handle x_position/y_position at team level
        if 'x_position' in feature_name.lower() or 'y_position' in feature_name.lower():
            position_base = feature_name.replace('x_position', 'position').replace('y_position', 'position').replace('X_position', 'position').replace('Y_position', 'position')
            aggregated_importance[position_base] += importance_val
            feature_to_base_mapping[feature_name] = position_base
        else:
            aggregated_importance[feature_name] += importance_val
            feature_to_base_mapping[feature_name] = feature_name

# Convert to dataframe and sort
aggregated_df = pd.DataFrame([
    {'feature': k, 'importance': v} 
    for k, v in aggregated_importance.items()
]).sort_values('importance', ascending=False).reset_index(drop=True)

print(f"✓ Aggregated into {len(aggregated_df)} unique feature groups")

# Calculate percentage contribution
total_importance = aggregated_df['importance'].sum()
aggregated_df['percentage'] = (aggregated_df['importance'] / total_importance) * 100
aggregated_df['cumulative_percentage'] = aggregated_df['percentage'].cumsum()

# Display top features
print("\n" + "="*60)
print("TOP 30 FEATURE GROUPS (Aggregated)")
print("="*60)
print(f"{'Rank':<6} {'Feature':<45} {'Importance':<12} {'%':<8} {'Cumul %':<10}")
print("-"*90)

for idx, row in aggregated_df.head(30).iterrows():
    print(f"{idx+1:<6} {row['feature']:<45} {row['importance']:<12.6f} {row['percentage']:<8.2f} {row['cumulative_percentage']:<10.2f}")

# Save full ranking to CSV
output_csv = 'results/feature_importance_aggregated.csv'
aggregated_df.to_csv(output_csv, index=False)
print(f"\n✓ Saved full ranking to: {output_csv}")

# Save top 50 for easy viewing
print(f"\nTop 50 features saved to: results/feature_importance_top50.csv")
aggregated_df.head(50).to_csv('results/feature_importance_top50.csv', index=False)

# Visualizations
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# 0. RFE Progress Plot
print("\n[1/4] Generating RFE progress plot...")
fig_rfe = plt.figure(figsize=(16, 10))
gs = fig_rfe.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: RMSE vs Number of Features
ax_rmse = fig_rfe.add_subplot(gs[0, :])
rfe_df = pd.DataFrame(rfe_history)
ax_rmse.plot(rfe_df['n_features'], rfe_df['train_rmse'], 'b-o', label='Train RMSE', linewidth=2, markersize=6)
ax_rmse.plot(rfe_df['n_features'], rfe_df['test_rmse'], 'r-o', label='Test RMSE', linewidth=2, markersize=6)
ax_rmse.axvline(x=len(best_features), color='g', linestyle='--', linewidth=2, label=f'Best: {len(best_features)} features', alpha=0.7)
ax_rmse.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax_rmse.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax_rmse.set_title('RFE: Performance vs Number of Features', fontsize=14, fontweight='bold', pad=15)
ax_rmse.legend(fontsize=11)
ax_rmse.grid(True, alpha=0.3)
ax_rmse.invert_xaxis()  # More features on left, fewer on right

# Plot 2: R² Score
ax_r2 = fig_rfe.add_subplot(gs[1, 0])
ax_r2.plot(rfe_df['n_features'], rfe_df['test_r2'], 'g-o', linewidth=2, markersize=6)
ax_r2.axvline(x=len(best_features), color='r', linestyle='--', linewidth=2, alpha=0.7)
ax_r2.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
ax_r2.set_ylabel('Test R² Score', fontsize=11, fontweight='bold')
ax_r2.set_title('R² Score vs Features', fontsize=12, fontweight='bold')
ax_r2.grid(True, alpha=0.3)
ax_r2.invert_xaxis()

# Plot 3: Improvement over iterations
ax_iter = fig_rfe.add_subplot(gs[1, 1])
improvements = [rfe_df['test_rmse'].iloc[0] - rmse for rmse in rfe_df['test_rmse']]
colors = ['green' if imp > 0 else 'red' for imp in improvements]
ax_iter.bar(rfe_df['iteration'], improvements, color=colors, alpha=0.7)
ax_iter.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax_iter.set_xlabel('RFE Iteration', fontsize=11, fontweight='bold')
ax_iter.set_ylabel('RMSE Improvement', fontsize=11, fontweight='bold')
ax_iter.set_title('Improvement vs Baseline', fontsize=12, fontweight='bold')
ax_iter.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Recursive Feature Elimination Analysis\nBest: {len(best_features)} features | RMSE: {best_test_rmse:.2f}', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('results/rfe_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: results/rfe_analysis.png")
plt.close()

# Save RFE history
rfe_df.to_csv('results/rfe_history.csv', index=False)
print(f"  ✓ Saved: results/rfe_history.csv")

# 1. Top 30 features bar chart
print("\n[2/4] Generating feature importance plots...")
fig, axes = plt.subplots(2, 1, figsize=(16, 14))

top_n = 30
top_features = aggregated_df.head(top_n)

ax1 = axes[0]
bars = ax1.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.8)
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['feature'], fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel('Feature Importance (Gain)', fontsize=12, fontweight='bold')
ax1.set_title(f'Top {top_n} Feature Groups by Importance (Aggregated)', 
              fontsize=14, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3)

# Add percentage labels
for i, (imp, pct) in enumerate(zip(top_features['importance'], top_features['percentage'])):
    ax1.text(imp, i, f' {pct:.1f}%', va='center', fontsize=8, color='darkblue')

# 2. Cumulative importance
ax2 = axes[1]
x_pos = range(len(aggregated_df))
ax2.plot(x_pos, aggregated_df['cumulative_percentage'], 'b-', linewidth=2.5, marker='o', markersize=3)
ax2.axhline(y=80, color='r', linestyle='--', linewidth=2, label='80% threshold', alpha=0.7)
ax2.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90% threshold', alpha=0.7)
ax2.fill_between(x_pos, 0, aggregated_df['cumulative_percentage'], alpha=0.2)

ax2.set_xlabel('Number of Feature Groups', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Importance (%)', fontsize=12, fontweight='bold')
ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_ylim([0, 105])

# Find how many features for 80% and 90%
n_80 = (aggregated_df['cumulative_percentage'] >= 80).idxmax() + 1
n_90 = (aggregated_df['cumulative_percentage'] >= 90).idxmax() + 1
ax2.text(0.5, 0.95, f'{n_80} features → 80%\n{n_90} features → 90%',
         transform=ax2.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/feature_importance_gradient_boosting.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: results/feature_importance_gradient_boosting.png")
plt.close()

# 3. Create a category-based analysis
print("\n[3/4] Generating category breakdown plot...")

category_mapping = {
    # Combat stats
    'kills': 'Combat',
    'deaths': 'Combat',
    'assists': 'Combat',
    'damage_dealt_to_champions': 'Combat',
    'total_damage_dealt': 'Combat',
    'physical_damage_dealt': 'Combat',
    'magic_damage_dealt': 'Combat',
    'true_damage_dealt': 'Combat',
    'damage_taken': 'Combat',
    'physical_damage_taken': 'Combat',
    'magic_damage_taken': 'Combat',
    'true_damage_taken': 'Combat',
    
    # Economy
    'total_gold': 'Economy',
    'current_gold': 'Economy',
    'gold_per_second': 'Economy',
    'gold_earned': 'Economy',
    
    # Farm
    'minions_killed': 'Farm',
    'jungle_minions_killed': 'Farm',
    'total_minions_killed': 'Farm',
    
    # Vision
    'ward': 'Vision',
    'vision': 'Vision',
    
    # XP & Level
    'level': 'XP/Level',
    'xp': 'XP/Level',
    
    # Objectives
    'turret': 'Objectives',
    'inhibitor': 'Objectives',
    'dragon': 'Objectives',
    'baron': 'Objectives',
    'rift': 'Objectives',
    'elite': 'Objectives',
    'building': 'Objectives',
    
    # Stats
    'ability_power': 'Stats',
    'armor': 'Stats',
    'magic_resist': 'Stats',
    'attack_damage': 'Stats',
    'attack_speed': 'Stats',
    'movement_speed': 'Stats',
    'health': 'Stats',
    'max_health': 'Stats',
    'ability_haste': 'Stats',
    
    # Spatial features (NEW)
    'spatial': 'Spatial',
    'centroiddist': 'Spatial',
    'mininterteamdist': 'Spatial',
    'engagediff': 'Spatial',
    'frontlineoverlap': 'Spatial',
    'radialvelocitydiff': 'Spatial',
    'position': 'Spatial',
    'distance': 'Spatial',
}

def categorize_feature(feature_name):
    feature_lower = feature_name.lower()
    for keyword, category in category_mapping.items():
        if keyword in feature_lower:
            return category
    return 'Other'

aggregated_df['category'] = aggregated_df['feature'].apply(categorize_feature)

# Aggregate by category
category_importance = aggregated_df.groupby('category')['importance'].sum().sort_values(ascending=False)
category_percentage = (category_importance / category_importance.sum()) * 100

print("\n" + "="*60)
print("FEATURE IMPORTANCE BY CATEGORY")
print("="*60)
print(f"{'Category':<20} {'Importance':<15} {'Percentage':<10}")
print("-"*50)
for cat, imp in category_importance.items():
    pct = category_percentage[cat]
    print(f"{cat:<20} {imp:<15.4f} {pct:<10.2f}%")

# Pie chart for categories
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
colors = plt.cm.Set3(range(len(category_importance)))
wedges, texts, autotexts = ax.pie(
    category_importance.values,
    labels=category_importance.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    textprops={'fontsize': 11}
)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax.set_title('Feature Importance by Category', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/feature_importance_by_category.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: results/feature_importance_by_category.png")

# Model performance (Final model with best features)
print("\n[4/4] Evaluating final model performance...")
print("\n" + "="*80)
print("FINAL MODEL PERFORMANCE (WITH BEST FEATURES)")
print("="*80)

y_train_pred = model.predict(X_train_best)
y_test_pred = model.predict(X_test_best)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nFinal Model ({len(best_features)} features, 100 estimators):")
print(f"  Train RMSE: {train_rmse:.2f}")
print(f"  Test RMSE:  {test_rmse:.2f}")
print(f"\n  Train MAE:  {train_mae:.2f}")
print(f"  Test MAE:   {test_mae:.2f}")
print(f"\n  Train R²:   {train_r2:.4f}")
print(f"  Test R²:    {test_r2:.4f}")

print(f"\nImprovement from initial to best:")
print(f"  Initial features: {rfe_df['n_features'].iloc[0]}")
print(f"  Best features: {len(best_features)}")
print(f"  Feature reduction: {rfe_df['n_features'].iloc[0] - len(best_features)} ({(1 - len(best_features)/rfe_df['n_features'].iloc[0])*100:.1f}%)")
print(f"  RMSE improvement: {rfe_df['test_rmse'].iloc[0] - best_test_rmse:.2f}")

# Save best features list
best_features_df = pd.DataFrame({'feature': best_features})
best_features_df.to_csv('results/best_features_rfe.csv', index=False)
print(f"\n✓ Saved best features list to: results/best_features_rfe.csv")

# Spatial features analysis
print("\n" + "="*60)
print("SPATIAL FEATURES ANALYSIS")
print("="*60)

spatial_keywords = ['Spatial', 'CentroidDist', 'MinInterTeamDist', 
                    'EngageDiff', 'FrontlineOverlap', 'RadialVelocityDiff']

spatial_feature_importance = aggregated_df[
    aggregated_df['feature'].apply(lambda x: any(kw.lower() in x.lower() for kw in spatial_keywords))
].copy()

if len(spatial_feature_importance) > 0:
    print(f"\nFound {len(spatial_feature_importance)} spatial feature groups")
    print(f"Total spatial importance: {spatial_feature_importance['importance'].sum():.4f}")
    print(f"Spatial contribution: {spatial_feature_importance['percentage'].sum():.2f}%")
    
    print(f"\nTop 15 Spatial Features:")
    print(f"{'Rank':<6} {'Feature':<50} {'Importance':<12} {'%':<8}")
    print("-"*80)
    for idx, row in spatial_feature_importance.head(15).iterrows():
        overall_rank = aggregated_df.index.get_loc(idx) + 1
        print(f"#{overall_rank:<5} {row['feature']:<50} {row['importance']:<12.6f} {row['percentage']:<8.2f}")
    
    # Save spatial features separately
    spatial_feature_importance.to_csv('results/spatial_feature_importance.csv', index=False)
    print(f"\n✓ Saved spatial feature analysis to: results/spatial_feature_importance.csv")
else:
    print("\n⚠ No spatial features found in the dataset")

print("\n" + "="*80)
print("RECURSIVE FEATURE ELIMINATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. results/rfe_analysis.png - RFE progress visualization")
print("  2. results/rfe_history.csv - Full RFE iteration history")
print("  3. results/best_features_rfe.csv - List of best features selected")
print("  4. results/feature_importance_aggregated.csv - Full feature ranking")
print("  5. results/feature_importance_top50.csv - Top 50 features")
print("  6. results/feature_importance_gradient_boosting.png - Feature importance plots")
print("  7. results/feature_importance_by_category.png - Category breakdown")
if len(spatial_feature_importance) > 0:
    print("  8. results/spatial_feature_importance.csv - Spatial features analysis")

print(f"\n✨ Best model uses {len(best_features)} features with test RMSE of {best_test_rmse:.2f}")
print(f"✨ This represents a {(1 - len(best_features)/rfe_df['n_features'].iloc[0])*100:.1f}% reduction in features!")

