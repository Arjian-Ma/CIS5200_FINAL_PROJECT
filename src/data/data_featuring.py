#!/usr/bin/env python3
"""
Script to transform xy_rows.csv into featured DataFrame
Aggregates 10 players per timestamp into a single row with team differences
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

def calculate_spatial_features(df):
    """
    Calculate spatial dynamic features for team positioning and movement
    
    Args:
        df: DataFrame with player position data
        
    Returns:
        DataFrame with added spatial features
    """
    print("\nCalculating spatial dynamic features...")
    
    # Initialize spatial feature columns
    df['CentroidDist'] = np.nan
    df['MinInterTeamDist'] = np.nan
    df['EngagedDiff'] = np.nan
    df['FrontlineOverlap'] = np.nan
    df['RadialVelocityDiff'] = np.nan
    
    # Process each timeframe with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating spatial features", unit="frame"):
        try:
            # Extract positions for both teams
            # Team A (Blue): Players 1-5 - CONSISTENT with build_xy_dataframe.py (team=1, Riot's team ID 100)
            team_a_positions = []
            for i in range(1, 6):
                x_col = f'Player{i}_X_Position'
                y_col = f'Player{i}_Y_Position'
                if x_col in df.columns and y_col in df.columns:
                    x, y = row[x_col], row[y_col]
                    if not pd.isna(x) and not pd.isna(y):
                        team_a_positions.append([x, y])
            
            # Team B (Red): Players 6-10 - CONSISTENT with build_xy_dataframe.py (team=-1, Riot's team ID 200)
            team_b_positions = []
            for i in range(6, 11):
                x_col = f'Player{i}_X_Position'
                y_col = f'Player{i}_Y_Position'
                if x_col in df.columns and y_col in df.columns:
                    x, y = row[x_col], row[y_col]
                    if not pd.isna(x) and not pd.isna(y):
                        team_b_positions.append([x, y])
            
            if len(team_a_positions) < 2 or len(team_b_positions) < 2:
                continue
                
            team_a_positions = np.array(team_a_positions)
            team_b_positions = np.array(team_b_positions)
            
            # 1. CentroidDist: Distance between team centroids
            centroid_a = np.mean(team_a_positions, axis=0)
            centroid_b = np.mean(team_b_positions, axis=0)
            centroid_dist = np.linalg.norm(centroid_a - centroid_b)
            df.at[idx, 'CentroidDist'] = centroid_dist
            
            # 2. MinInterTeamDist: Closest enemy pair distance
            distances = cdist(team_a_positions, team_b_positions)
            min_inter_team_dist = np.min(distances)
            df.at[idx, 'MinInterTeamDist'] = min_inter_team_dist
            
            # 3. EngagedDiff: Number advantage at contact (d0 ≈ 900)
            d0 = 900
            engaged_a = 0
            engaged_b = 0
            
            # Count engaged players in team A
            for pos_a in team_a_positions:
                min_dist_to_enemy = np.min([np.linalg.norm(pos_a - pos_b) for pos_b in team_b_positions])
                if min_dist_to_enemy < d0:
                    engaged_a += 1
            
            # Count engaged players in team B
            for pos_b in team_b_positions:
                min_dist_to_enemy = np.min([np.linalg.norm(pos_b - pos_a) for pos_a in team_a_positions])
                if min_dist_to_enemy < d0:
                    engaged_b += 1
            
            engaged_diff = engaged_a - engaged_b
            df.at[idx, 'EngagedDiff'] = engaged_diff
            
            # 4. FrontlineOverlap: Projection overlap on centroid axis
            # Calculate centroid axis vector
            u = centroid_b - centroid_a
            u_norm = np.linalg.norm(u)
            if u_norm > 0:
                u = u / u_norm  # Normalize
                
                # Project team A positions onto axis
                proj_a = np.dot(team_a_positions, u)
                # Project team B positions onto axis (negative direction)
                proj_b = np.dot(team_b_positions, -u)
                
                # Find forward-most players
                forward_a = np.max(proj_a)
                forward_b = np.max(proj_b)
                
                # Calculate overlap
                overlap = forward_a + forward_b - centroid_dist
                df.at[idx, 'FrontlineOverlap'] = overlap
            
            # 5. RadialVelocityDiff: Average forward velocity difference
            # This requires previous frame data, so we'll calculate it differently
            # For now, we'll use a simplified version based on current positions
            # In a real implementation, you'd need to track previous positions
            
            # Simplified RVD: Use position relative to centroid
            team_a_radial = np.mean([np.dot(pos - centroid_a, u) for pos in team_a_positions])
            team_b_radial = np.mean([np.dot(pos - centroid_b, -u) for pos in team_b_positions])
            radial_velocity_diff = team_a_radial - team_b_radial
            df.at[idx, 'RadialVelocityDiff'] = radial_velocity_diff
            
        except Exception as e:
            # Skip problematic rows
            continue
    
    print(f"\n✅ Spatial features calculated for {len(df)} rows")
    return df

def create_featured_dataframe(input_csv_path, output_csv_path=None):
    """
    Transform xy_rows.csv where each timestamp has 10 rows (1 per player)
    into a DataFrame where each timestamp has 1 row with aggregated team differences
    
    Args:
        input_csv_path (str): Path to xy_rows.csv
        output_csv_path (str): Optional path to save the featured DataFrame as CSV
    
    Returns:
        pandas.DataFrame: Featured DataFrame with aggregated team differences
    """
    
    print(f"Loading data from {input_csv_path}...")
    if input_csv_path.lower().endswith('.parquet'):
        df = pd.read_parquet(input_csv_path)
    else:
        df = pd.read_csv(input_csv_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Total rows: {len(df)}")
    print(f"Expected timeframes: {len(df) // 10}")
    
    # Group by match_id, frame_idx, and timestamp
    grouped = df.groupby(['match_id', 'frame_idx', 'timestamp'])
    
    featured_rows = []
    
    print("\\nProcessing timeframes...")
    
    # Add progress bar for processing timeframes
    for i, (group_key, group_df) in enumerate(tqdm(grouped, desc="Processing timeframes", unit="timeframe")):
        match_id, frame_idx, timestamp = group_key
        
        # Calculate team differences (team 1 minus team -1)
        # For each metric, sum(value * team) gives the difference
        # 
        # CONSISTENCY CHECK:
        # - team = 1 for participants 1-5 (Blue team, Riot's team ID 100)
        # - team = -1 for participants 6-10 (Red team, Riot's team ID 200)
        # - Formula: (value * team).sum() = Blue_total - Red_total
        # - Positive difference = Blue ahead, Negative difference = Red ahead
        # - This aligns with: Blue wins (winningTeam=100) → participants 1-5 have Y_won=1
        
        # Extract winning label (same for all frames in a match)
        # Y_won: 1 if participant's team won, 0 otherwise
        # Blue team (participants 1-5, team=1) have Y_won=1 when Blue wins
        # Red team (participants 6-10, team=-1) have Y_won=1 when Red wins
        # We check if any Blue participant has Y_won=1 to determine if Blue won
        blue_team_rows = group_df[group_df['team'] == 1]
        if len(blue_team_rows) > 0:
            # Check if Blue team won (any Blue participant has Y_won=1)
            blue_won = blue_team_rows['Y_won'].iloc[0] == 1 if 'Y_won' in blue_team_rows.columns else 0
            # Winning label: 1 if Blue won, 0 if Red won
            winning_label = 1 if blue_won else 0
        else:
            # Fallback: check if any participant has Y_won=1
            if 'Y_won' in group_df.columns:
                winning_label = 1 if group_df['Y_won'].iloc[0] == 1 and group_df['team'].iloc[0] == 1 else 0
            else:
                winning_label = 0
        
        # 1. Total_Gold_Difference
        total_gold_diff = (group_df['Y_total_gold'] * group_df['team']).sum()
        
        # 2. Total_Xp_Difference
        total_xp_diff = (group_df['Y_total_xp'] * group_df['team']).sum()
        
        # 5. Total_Minions_Killed_Difference
        total_minions_killed_diff = (group_df['X_minions_killed'] * group_df['team']).sum()
        
        # 6. Total_Jungle_Minions_Killed_Difference
        total_jungle_minions_killed_diff = (group_df['X_jungle_minions_killed'] * group_df['team']).sum()
        
        # Damage Profile (7-18): Each damage metric * team
        damage_metrics = {
            'Magic_Damage_Done_Diff': (group_df['X_dmg_magicDamageDone'] * group_df['team']).sum(),
            'Magic_Damage_Done_To_Champions_Diff': (group_df['X_dmg_magicDamageDoneToChampions'] * group_df['team']).sum(),
            'Magic_Damage_Taken_Diff': (group_df['X_dmg_magicDamageTaken'] * group_df['team']).sum(),
            'Physical_Damage_Done_Diff': (group_df['X_dmg_physicalDamageDone'] * group_df['team']).sum(),
            'Physical_Damage_Done_To_Champions_Diff': (group_df['X_dmg_physicalDamageDoneToChampions'] * group_df['team']).sum(),
            'Physical_Damage_Taken_Diff': (group_df['X_dmg_physicalDamageTaken'] * group_df['team']).sum(),
            'Total_Damage_Done_Diff': (group_df['X_dmg_totalDamageDone'] * group_df['team']).sum(),
            'Total_Damage_Done_To_Champions_Diff': (group_df['X_dmg_totalDamageDoneToChampions'] * group_df['team']).sum(),
            'Total_Damage_Taken_Diff': (group_df['X_dmg_totalDamageTaken'] * group_df['team']).sum(),
            'True_Damage_Done_Diff': (group_df['X_dmg_trueDamageDone'] * group_df['team']).sum(),
            'True_Damage_Done_To_Champions_Diff': (group_df['X_dmg_trueDamageDoneToChampions'] * group_df['team']).sum(),
            'True_Damage_Taken_Diff': (group_df['X_dmg_trueDamageTaken'] * group_df['team']).sum(),
        }
        
        # Performance/Scoreline (19-20)
        # 19. Total_Kill_Difference - uses Y_total_kills (cumulative)
        total_kill_diff = (group_df['Y_total_kills'] * group_df['team']).sum()
        
        # 20. Total_Assist_Difference - uses X_evt_assists (incremental, needs cumsum later)
        assist_diff_increment = (group_df['X_evt_assists'] * group_df['team']).sum()
        
        # Vision/Map_Control (21-23)
        # 21. Total_Ward_Placed_Difference - incremental, needs cumsum
        ward_placed_diff_increment = (group_df['X_evt_wards_placed'] * group_df['team']).sum()
        
        # 22. Total_Ward_Killed_Difference - incremental, needs cumsum
        ward_killed_diff_increment = (group_df['X_evt_wards_killed'] * group_df['team']).sum()
        
        # 23. Time_Enemy_Spent_Controlled_Difference - cumulative
        time_controlled_diff = (group_df['X_time_enemy_spent_controlled'] * group_df['team']).sum()
        
        # Objectives (24-25)
        # 24. Elite_Monster_Killed_Difference - incremental, needs cumsum
        elite_monster_diff_increment = (group_df['X_evt_elite_monsters'] * group_df['team']).sum()
        
        # 25. Buildings_Taken_Difference - incremental, needs cumsum
        buildings_diff_increment = (group_df['X_evt_buildings'] * group_df['team']).sum()
        
        # Individual Player Stats (26-269)
        # For each of the 10 players, extract their individual stats
        player_stats = {}
        
        for participant_id in range(1, 11):
            # Get the row for this specific player
            player_row = group_df[group_df['participantId'] == participant_id]
            
            if len(player_row) == 0:
                # If player not found, fill with NaN
                prefix = f'Player{participant_id}_'
                player_stats[f'{prefix}Team'] = np.nan
                player_stats[f'{prefix}Attack_Damage'] = np.nan
                player_stats[f'{prefix}Attack_Speed'] = np.nan
                player_stats[f'{prefix}Ability_Power'] = np.nan
                player_stats[f'{prefix}Ability_Haste'] = np.nan
                player_stats[f'{prefix}Armor_Pen_Percent'] = np.nan
                player_stats[f'{prefix}Armor_Pen'] = np.nan
                player_stats[f'{prefix}Bonus_Armor_Pen_Percent'] = np.nan
                player_stats[f'{prefix}Magic_Pen'] = np.nan
                player_stats[f'{prefix}Magic_Pen_Percent'] = np.nan
                player_stats[f'{prefix}Bonus_Magic_Pen_Percent'] = np.nan
                player_stats[f'{prefix}Armor'] = np.nan
                player_stats[f'{prefix}Magic_Resist'] = np.nan
                player_stats[f'{prefix}Health_Percentage'] = np.nan
                player_stats[f'{prefix}Health_Regen'] = np.nan
                player_stats[f'{prefix}Life_Steal'] = np.nan
                player_stats[f'{prefix}Omnivamp'] = np.nan
                player_stats[f'{prefix}Physical_Vamp'] = np.nan
                player_stats[f'{prefix}Spell_Vamp'] = np.nan
                player_stats[f'{prefix}Power_Percent'] = np.nan
                player_stats[f'{prefix}Power_Regen'] = np.nan
                player_stats[f'{prefix}Movement_Speed'] = np.nan
                player_stats[f'{prefix}X_Position'] = np.nan
                player_stats[f'{prefix}Y_Position'] = np.nan
            else:
                player_row = player_row.iloc[0]  # Get the first (and only) row
                prefix = f'Player{participant_id}_'
                
                # Team
                player_stats[f'{prefix}Team'] = player_row['team']
                
                # Offensive Stats (28-37)
                player_stats[f'{prefix}Attack_Damage'] = player_row['X_champ_attackDamage']
                player_stats[f'{prefix}Attack_Speed'] = player_row['X_champ_attackSpeed']
                player_stats[f'{prefix}Ability_Power'] = player_row['X_champ_abilityPower']
                player_stats[f'{prefix}Ability_Haste'] = player_row['X_champ_abilityHaste']
                player_stats[f'{prefix}Armor_Pen_Percent'] = player_row['X_champ_armorPenPercent']
                player_stats[f'{prefix}Armor_Pen'] = player_row['X_champ_armorPen']
                player_stats[f'{prefix}Bonus_Armor_Pen_Percent'] = player_row['X_champ_bonusArmorPenPercent']
                player_stats[f'{prefix}Magic_Pen'] = player_row['X_champ_magicPen']
                player_stats[f'{prefix}Magic_Pen_Percent'] = player_row['X_champ_magicPenPercent']
                player_stats[f'{prefix}Bonus_Magic_Pen_Percent'] = player_row['X_champ_bonusMagicPenPercent']
                
                # Defensive Stats (38-39)
                player_stats[f'{prefix}Armor'] = player_row['X_champ_armor']
                player_stats[f'{prefix}Magic_Resist'] = player_row['X_champ_magicResist']
                
                # Health Stats (40-41)
                health_max = player_row['X_champ_healthMax']
                player_stats[f'{prefix}Health_Percentage'] = (player_row['X_champ_health'] / health_max) if health_max > 0 else 0
                player_stats[f'{prefix}Health_Regen'] = player_row['X_champ_healthRegen']
                
                # Sustain Stats (42-45)
                player_stats[f'{prefix}Life_Steal'] = player_row['X_champ_lifesteal']
                player_stats[f'{prefix}Omnivamp'] = player_row['X_champ_omnivamp']
                player_stats[f'{prefix}Physical_Vamp'] = player_row['X_champ_physicalVamp']
                player_stats[f'{prefix}Spell_Vamp'] = player_row['X_champ_spellVamp']
                
                # Power/Mana Stats (46-47)
                power_max = player_row['X_champ_powerMax']
                player_stats[f'{prefix}Power_Percent'] = (player_row['X_champ_power'] / power_max) if power_max > 0 else 0
                player_stats[f'{prefix}Power_Regen'] = player_row['X_champ_powerRegen']
                
                # Movement and Position (48-50)
                player_stats[f'{prefix}Movement_Speed'] = player_row['X_champ_movementSpeed']
                player_stats[f'{prefix}X_Position'] = player_row['X_pos_x']
                player_stats[f'{prefix}Y_Position'] = player_row['X_pos_y']
        
        # Create row dictionary
        row = {
            'match_id': match_id,
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            # Winning label (1 if Blue team won, 0 if Red team won)
            # Consistent with team mapping: Blue (participants 1-5, team=1) wins → label=1
            'Y_won': winning_label,
            # Basic Economy/Progression
            'Total_Gold_Difference': total_gold_diff,
            'Total_Xp_Difference': total_xp_diff,
            # Will be filled in next step
            'Total_Gold_Difference_Last_Time_Frame': np.nan,
            'Total_Xp_Difference_Last_Time_Frame': np.nan,
            # Minions
            'Total_Minions_Killed_Difference': total_minions_killed_diff,
            'Total_Jungle_Minions_Killed_Difference': total_jungle_minions_killed_diff,
        }
        
        # Add damage profile
        row.update(damage_metrics)
        
        # Add Performance/Scoreline
        row['Total_Kill_Difference'] = total_kill_diff
        row['Total_Assist_Difference_Increment'] = assist_diff_increment
        
        # Add Vision/Map_Control
        row['Total_Ward_Placed_Difference_Increment'] = ward_placed_diff_increment
        row['Total_Ward_Killed_Difference_Increment'] = ward_killed_diff_increment
        row['Time_Enemy_Spent_Controlled_Difference'] = time_controlled_diff
        
        # Add Objectives
        row['Elite_Monster_Killed_Difference_Increment'] = elite_monster_diff_increment
        row['Buildings_Taken_Difference_Increment'] = buildings_diff_increment
        
        # Add all player stats
        row.update(player_stats)
        
        featured_rows.append(row)
    
    # Create DataFrame
    print(f"\nCreating featured DataFrame...")
    print(f"Processing {len(featured_rows)} timeframes...")
    featured_df = pd.DataFrame(featured_rows)
    print(f"Created DataFrame with shape: {featured_df.shape}")
    
    # Fill in "Last Time Frame" columns (shift by 1 within each match)
    print("Calculating last time frame differences...")
    featured_df['Total_Gold_Difference_Last_Time_Frame'] = featured_df.groupby('match_id')['Total_Gold_Difference'].shift(1)
    featured_df['Total_Xp_Difference_Last_Time_Frame'] = featured_df.groupby('match_id')['Total_Xp_Difference'].shift(1)
    
    # Calculate cumulative sums for incremental event columns
    print("Calculating cumulative sums for incremental columns...")
    featured_df['Total_Assist_Difference'] = featured_df.groupby('match_id')['Total_Assist_Difference_Increment'].cumsum()
    featured_df['Total_Ward_Placed_Difference'] = featured_df.groupby('match_id')['Total_Ward_Placed_Difference_Increment'].cumsum()
    featured_df['Total_Ward_Killed_Difference'] = featured_df.groupby('match_id')['Total_Ward_Killed_Difference_Increment'].cumsum()
    featured_df['Elite_Monster_Killed_Difference'] = featured_df.groupby('match_id')['Elite_Monster_Killed_Difference_Increment'].cumsum()
    featured_df['Buildings_Taken_Difference'] = featured_df.groupby('match_id')['Buildings_Taken_Difference_Increment'].cumsum()
    
    # Drop the increment columns as they were just temporary
    featured_df = featured_df.drop(columns=[
        'Total_Assist_Difference_Increment',
        'Total_Ward_Placed_Difference_Increment',
        'Total_Ward_Killed_Difference_Increment',
        'Elite_Monster_Killed_Difference_Increment',
        'Buildings_Taken_Difference_Increment'
    ])
    
    # Reorder columns for better organization
    column_order = [
        'match_id', 'frame_idx', 'timestamp',
        # Winning label (column 4)
        'Y_won',
        # Economy/Progression (columns 5-10)
        'Total_Gold_Difference',
        'Total_Xp_Difference',
        'Total_Gold_Difference_Last_Time_Frame',
        'Total_Xp_Difference_Last_Time_Frame',
        'Total_Minions_Killed_Difference',
        'Total_Jungle_Minions_Killed_Difference',
        # Damage Profile (columns 10-21)
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
        # Performance/Scoreline (columns 22-23)
        'Total_Kill_Difference',
        'Total_Assist_Difference',
        # Vision/Map_Control (columns 24-26)
        'Total_Ward_Placed_Difference',
        'Total_Ward_Killed_Difference',
        'Time_Enemy_Spent_Controlled_Difference',
        # Objectives (columns 27-28)
        'Elite_Monster_Killed_Difference',
        'Buildings_Taken_Difference',
    ]
    
    # Add individual player stats (columns 29-268)
    # 24 stats per player × 10 players = 240 columns
    player_stat_names = [
        'Team',
        'Attack_Damage',
        'Attack_Speed',
        'Ability_Power',
        'Ability_Haste',
        'Armor_Pen_Percent',
        'Armor_Pen',
        'Bonus_Armor_Pen_Percent',
        'Magic_Pen',
        'Magic_Pen_Percent',
        'Bonus_Magic_Pen_Percent',
        'Armor',
        'Magic_Resist',
        'Health_Percentage',
        'Health_Regen',
        'Life_Steal',
        'Omnivamp',
        'Physical_Vamp',
        'Spell_Vamp',
        'Power_Percent',
        'Power_Regen',
        'Movement_Speed',
        'X_Position',
        'Y_Position',
    ]
    
    for player_id in range(1, 11):
        for stat_name in player_stat_names:
            column_order.append(f'Player{player_id}_{stat_name}')
    
    featured_df = featured_df[column_order]
    
    # Calculate spatial features (requires player position columns)
    print("\n=== Calculating Spatial Features ===")
    featured_df = calculate_spatial_features(featured_df)
    
    # Add spatial features to column order (after player stats)
    spatial_features = ['CentroidDist', 'MinInterTeamDist', 'EngagedDiff', 'FrontlineOverlap', 'RadialVelocityDiff']
    for feature in spatial_features:
        if feature not in column_order:
            column_order.append(feature)
    
    # Reorder to ensure spatial features are at the end
    featured_df = featured_df[column_order]
    
    # Fill NaN values with 0 (especially important for timestamp 0 where some values may be NaN)
    # Only fill numeric columns, exclude identifier columns and Y_won (label column)
    print("\nFilling NaN values with 0...")
    identifier_cols = ['match_id', 'frame_idx', 'timestamp']
    label_cols = ['Y_won'] if 'Y_won' in featured_df.columns else []
    exclude_cols = identifier_cols + label_cols
    numeric_cols = [col for col in featured_df.columns if col not in exclude_cols]
    
    nan_counts_before = featured_df[numeric_cols].isna().sum().sum()
    featured_df[numeric_cols] = featured_df[numeric_cols].fillna(0)
    nan_counts_after = featured_df[numeric_cols].isna().sum().sum()
    print(f"  Filled {nan_counts_before} NaN values in numeric columns (remaining: {nan_counts_after})")
    
    # Fill NaN in Y_won with 0 (if any exist, though they shouldn't)
    if 'Y_won' in featured_df.columns:
        y_won_nan = featured_df['Y_won'].isna().sum()
        if y_won_nan > 0:
            print(f"  Warning: {y_won_nan} NaN values in Y_won label column, filling with 0")
            featured_df['Y_won'] = featured_df['Y_won'].fillna(0).astype(int)
    
    # Check for any remaining NaNs in identifier columns (should be rare)
    id_nan_counts = featured_df[identifier_cols].isna().sum().sum()
    if id_nan_counts > 0:
        print(f"  Warning: {id_nan_counts} NaN values found in identifier columns (match_id, frame_idx, timestamp)")
    
    print(f"\n=== Featured DataFrame Summary ===")
    print(f"Shape: {featured_df.shape}")
    print(f"Columns: {list(featured_df.columns)}")
    print(f"\nFirst 5 rows:")
    print(featured_df.head())
    
    print(f"\nColumn Groups:")
    print("- Identifiers (3): match_id, frame_idx, timestamp")
    print("- Winning Label (1): Y_won (1 if Blue team won, 0 if Red team won)")
    print("- Economy/Progression (6): Total_Gold_Difference, Total_Xp_Difference, etc.")
    print("- Damage Profile (12): Magic, Physical, True damage metrics")
    print("- Performance/Scoreline (2): Total_Kill_Difference, Total_Assist_Difference")
    print("- Vision/Map_Control (3): Total_Ward_Placed_Difference, Total_Ward_Killed_Difference, etc.")
    print("- Objectives (2): Elite_Monster_Killed_Difference, Buildings_Taken_Difference")
    print("- Individual Player Stats (240): 24 stats × 10 players")
    print(f"  - Per player: Team, Attack stats, Defensive stats, Health, Sustain, Power, Movement, Position")
    print("- Spatial Dynamic Features (5): CentroidDist, MinInterTeamDist, EngagedDiff, FrontlineOverlap, RadialVelocityDiff")
    
    # Save to file if path provided
    if output_csv_path:
        print(f"\nSaving featured DataFrame to: {output_csv_path}")
        if output_csv_path.lower().endswith('.parquet'):
            featured_df.to_parquet(output_csv_path, index=False)
        else:
            featured_df.to_csv(output_csv_path, index=False)
            print(f"✅ Successfully saved {len(featured_df)} rows to {output_csv_path}")
    
    return featured_df

def main():
    """Main function to run the feature engineering"""
    
    input_file = "data/raw/xy_rows.parquet"
    output_file = "data/processed/featured_data.parquet"
    
    print("=== League of Legends Feature Engineering ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    try:
        # Create featured DataFrame
        featured_df = create_featured_dataframe(input_file, output_file)
    except FileNotFoundError as e:
        print(f"❌ Error: Input file not found: {input_file}")
        print("Please make sure you have run the build_xy_dataframe.py script first to create xy_rows.csv")
        return
    except Exception as e:
        print(f"❌ Error during feature engineering: {e}")
        return
    
    if featured_df is not None:
        print("\n=== Sample Statistics ===")
        print("\nGold Difference Statistics:")
        print(featured_df['Total_Gold_Difference'].describe())
        
        print("\nXP Difference Statistics:")
        print(featured_df['Total_Xp_Difference'].describe())
        
        print("\nSpatial Features Statistics:")
        spatial_cols = ['CentroidDist', 'MinInterTeamDist', 'EngagedDiff', 'FrontlineOverlap', 'RadialVelocityDiff']
        if all(col in featured_df.columns for col in spatial_cols):
            print(featured_df[spatial_cols].describe())
        
        print("\nData shape transformation:")
        print(f"Input: ~{len(featured_df) * 10} rows (10 players per timeframe)")
        print(f"Output: {len(featured_df)} rows (1 row per timeframe)")
        
        print("\n✅ Feature engineering completed successfully!")

if __name__ == "__main__":
    main()

