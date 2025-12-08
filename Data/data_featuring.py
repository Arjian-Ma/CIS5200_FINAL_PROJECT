#!/usr/bin/env python3
"""
Script to transform xy_rows.csv into featured DataFrame
Aggregates 10 players per timestamp into a single row with team differences
"""

import pandas as pd
import numpy as np

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
    df = pd.read_csv(input_csv_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Total rows: {len(df)}")
    print(f"Expected timeframes: {len(df) // 10}")
    
    # Group by match_id, frame_idx, and timestamp
    grouped = df.groupby(['match_id', 'frame_idx', 'timestamp'])
    
    featured_rows = []
    
    print("\\nProcessing timeframes...")
    for i, (group_key, group_df) in enumerate(grouped):
        match_id, frame_idx, timestamp = group_key
        
        if i % 100 == 0:
            print(f"Processing timeframe {i+1}...")
        
        # Calculate team differences (team 1 minus team -1)
        # For each metric, sum(value * team) gives the difference
        
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
    print(f"\\nCreating featured DataFrame...")
    featured_df = pd.DataFrame(featured_rows)
    
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
        # Economy/Progression (columns 4-9)
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
    
    print(f"\\n=== Featured DataFrame Summary ===")
    print(f"Shape: {featured_df.shape}")
    print(f"Columns: {list(featured_df.columns)}")
    print(f"\\nFirst 5 rows:")
    print(featured_df.head())
    
    print(f"\\nColumn Groups:")
    print("- Identifiers (3): match_id, frame_idx, timestamp")
    print("- Economy/Progression (6): Total_Gold_Difference, Total_Xp_Difference, etc.")
    print("- Damage Profile (12): Magic, Physical, True damage metrics")
    print("- Performance/Scoreline (2): Total_Kill_Difference, Total_Assist_Difference")
    print("- Vision/Map_Control (3): Total_Ward_Placed_Difference, Total_Ward_Killed_Difference, etc.")
    print("- Objectives (2): Elite_Monster_Killed_Difference, Buildings_Taken_Difference")
    print("- Individual Player Stats (240): 24 stats × 10 players")
    print(f"  - Per player: Team, Attack stats, Defensive stats, Health, Sustain, Power, Movement, Position")
    
    # Save to CSV if path provided
    if output_csv_path:
        featured_df.to_csv(output_csv_path, index=False)
        print(f"\\nFeatured DataFrame saved to: {output_csv_path}")
    
    return featured_df

def main():
    """Main function to run the feature engineering"""
    
    input_file = "xy_rows.csv"
    output_file = "featured_data.csv"
    
    print("=== League of Legends Feature Engineering ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Create featured DataFrame
    featured_df = create_featured_dataframe(input_file, output_file)
    
    if featured_df is not None:
        print("\\n=== Sample Statistics ===")
        print("\\nGold Difference Statistics:")
        print(featured_df['Total_Gold_Difference'].describe())
        
        print("\\nXP Difference Statistics:")
        print(featured_df['Total_Xp_Difference'].describe())
        
        print("\\nData shape transformation:")
        print(f"Input: ~{len(featured_df) * 10} rows (10 players per timeframe)")
        print(f"Output: {len(featured_df)} rows (1 row per timeframe)")
        
        print("\\n✅ Feature engineering completed successfully!")

if __name__ == "__main__":
    main()

