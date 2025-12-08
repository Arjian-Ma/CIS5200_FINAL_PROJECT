#!/usr/bin/env python3
"""
Script to calculate player scores from individual player statistics
Assigns scores for different profiles: Offensive, Defensive, Sustain, Resources, Mobility
Then calculates an overall player score
"""

import pandas as pd
import numpy as np

def calculate_offensive_score(player_prefix, df):
    """
    Calculate offensive score based on attack and ability stats
    Weighted combination of offensive capabilities
    """
    # Normalize and weight offensive stats
    # Physical damage dealers: Attack Damage, Attack Speed, Armor Penetration
    # Magic damage dealers: Ability Power, Ability Haste, Magic Penetration
    
    physical_score = (
        df[f'{player_prefix}Attack_Damage'] * 0.3 +  # Base damage
        df[f'{player_prefix}Attack_Speed'] * 100 * 0.2 +  # Attack speed (scaled)
        df[f'{player_prefix}Armor_Pen'] * 0.15 +
        df[f'{player_prefix}Armor_Pen_Percent'] * 0.15 +
        df[f'{player_prefix}Bonus_Armor_Pen_Percent'] * 0.2
    )
    
    magic_score = (
        df[f'{player_prefix}Ability_Power'] * 0.3 +  # Base ability power
        df[f'{player_prefix}Ability_Haste'] * 0.2 +  # Cooldown reduction
        df[f'{player_prefix}Magic_Pen'] * 0.15 +
        df[f'{player_prefix}Magic_Pen_Percent'] * 0.15 +
        df[f'{player_prefix}Bonus_Magic_Pen_Percent'] * 0.2
    )
    
    # Take the maximum of physical or magic (players typically specialize in one)
    offensive_score = np.maximum(physical_score, magic_score)
    
    return offensive_score

def calculate_defensive_score(player_prefix, df):
    """
    Calculate defensive score based on armor, magic resist, health
    """
    defensive_score = (
        df[f'{player_prefix}Armor'] * 0.3 +  # Physical defense
        df[f'{player_prefix}Magic_Resist'] * 0.3 +  # Magic defense
        df[f'{player_prefix}Health_Percentage'] * 100 * 0.3 +  # Current health (scaled to 0-100)
        df[f'{player_prefix}Health_Regen'] * 0.1  # Health regeneration
    )
    
    return defensive_score

def calculate_sustain_score(player_prefix, df):
    """
    Calculate sustain score based on lifesteal and vamp stats
    """
    sustain_score = (
        df[f'{player_prefix}Life_Steal'] * 100 * 0.3 +  # Lifesteal (scaled)
        df[f'{player_prefix}Omnivamp'] * 100 * 0.4 +  # Omnivamp (most versatile)
        df[f'{player_prefix}Physical_Vamp'] * 100 * 0.15 +
        df[f'{player_prefix}Spell_Vamp'] * 100 * 0.15
    )
    
    return sustain_score

def calculate_resources_score(player_prefix, df):
    """
    Calculate resources score based on power/mana availability
    """
    resources_score = (
        df[f'{player_prefix}Power_Percent'] * 50 * 0.6 +  # Current resource (scaled to 0-50)
        df[f'{player_prefix}Power_Regen'] * 0.4  # Resource regeneration
    )
    
    return resources_score

def calculate_mobility_score(player_prefix, df):
    """
    Calculate mobility score based on movement speed
    Position is not included in mobility score as it's contextual
    """
    # Movement speed is the primary indicator of mobility
    # Typical range is 300-500, we normalize it
    mobility_score = df[f'{player_prefix}Movement_Speed']
    
    return mobility_score

def calculate_overall_player_score(player_prefix, df):
    """
    Calculate overall player score from all profile scores
    Weighted combination emphasizing offensive and defensive capabilities
    """
    offensive = df[f'{player_prefix}Offensive_Score']
    defensive = df[f'{player_prefix}Defensive_Score']
    sustain = df[f'{player_prefix}Sustain_Score']
    resources = df[f'{player_prefix}Resources_Score']
    mobility = df[f'{player_prefix}Mobility_Score']
    
    # Overall score with different weights
    # Offensive and Defensive are most important, followed by Sustain
    overall_score = (
        offensive * 0.35 +  # Offensive capability
        defensive * 0.30 +  # Defensive capability
        sustain * 0.15 +    # Sustain
        resources * 0.10 +  # Resource management
        mobility * 0.10     # Mobility
    )
    
    return overall_score

def add_player_scores(input_csv_path, output_csv_path=None):
    """
    Add player score columns to the featured DataFrame
    
    Args:
        input_csv_path (str): Path to featured_data.csv
        output_csv_path (str): Optional path to save the scored DataFrame
    
    Returns:
        pandas.DataFrame: DataFrame with player scores added
    """
    
    print(f"Loading featured data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    print(f"Original shape: {df.shape}")
    
    # Calculate scores for each player
    print("\\nCalculating player scores...")
    
    for player_id in range(1, 11):
        player_prefix = f'Player{player_id}_'
        
        print(f"Processing Player {player_id}...")
        
        # Calculate profile scores
        df[f'{player_prefix}Offensive_Score'] = calculate_offensive_score(player_prefix, df)
        df[f'{player_prefix}Defensive_Score'] = calculate_defensive_score(player_prefix, df)
        df[f'{player_prefix}Sustain_Score'] = calculate_sustain_score(player_prefix, df)
        df[f'{player_prefix}Resources_Score'] = calculate_resources_score(player_prefix, df)
        df[f'{player_prefix}Mobility_Score'] = calculate_mobility_score(player_prefix, df)
        
        # Calculate overall player score
        df[f'{player_prefix}Overall_Score'] = calculate_overall_player_score(player_prefix, df)
    
    # Drop individual player stats columns (keep only scores)
    print("\\nRemoving individual player stat columns (keeping only scores)...")
    
    # Identify columns to drop (all Player stats except scores)
    player_stat_cols_to_drop = []
    for player_id in range(1, 11):
        prefix = f'Player{player_id}_'
        # Drop all player columns except the score columns
        for col in df.columns:
            if col.startswith(prefix) and not col.endswith('_Score'):
                player_stat_cols_to_drop.append(col)
    
    print(f"Dropping {len(player_stat_cols_to_drop)} individual player stat columns...")
    df = df.drop(columns=player_stat_cols_to_drop)
    
    print(f"\\n=== Scored DataFrame Summary ===")
    print(f"New shape: {df.shape}")
    print(f"Retained columns: Team aggregates (28) + Player scores (60) + Team score aggregates (9)")
    
    # Show sample scores for Player 1
    print("\\n=== Sample Scores (Player 1, first 5 frames) ===")
    score_cols = [
        'frame_idx',
        'Player1_Offensive_Score',
        'Player1_Defensive_Score',
        'Player1_Sustain_Score',
        'Player1_Resources_Score',
        'Player1_Mobility_Score',
        'Player1_Overall_Score'
    ]
    print(df[score_cols].head())
    
    # Show statistics for Player 1 overall score
    print("\\n=== Player 1 Overall Score Statistics ===")
    print(df['Player1_Overall_Score'].describe())
    
    # Calculate team scores (sum of 5 players per team)
    print("\\n=== Calculating Team Scores ===")
    
    # Blue team (Players 1-5)
    blue_offensive = sum(df[f'Player{i}_Offensive_Score'] for i in range(1, 6))
    blue_defensive = sum(df[f'Player{i}_Defensive_Score'] for i in range(1, 6))
    blue_overall = sum(df[f'Player{i}_Overall_Score'] for i in range(1, 6))
    
    # Red team (Players 6-10)
    red_offensive = sum(df[f'Player{i}_Offensive_Score'] for i in range(6, 11))
    red_defensive = sum(df[f'Player{i}_Defensive_Score'] for i in range(6, 11))
    red_overall = sum(df[f'Player{i}_Overall_Score'] for i in range(6, 11))
    
    # Add team scores
    df['Blue_Team_Offensive_Score'] = blue_offensive
    df['Blue_Team_Defensive_Score'] = blue_defensive
    df['Blue_Team_Overall_Score'] = blue_overall
    
    df['Red_Team_Offensive_Score'] = red_offensive
    df['Red_Team_Defensive_Score'] = red_defensive
    df['Red_Team_Overall_Score'] = red_overall
    
    # Calculate team score differences
    df['Team_Offensive_Score_Diff'] = df['Blue_Team_Offensive_Score'] - df['Red_Team_Offensive_Score']
    df['Team_Defensive_Score_Diff'] = df['Blue_Team_Defensive_Score'] - df['Red_Team_Defensive_Score']
    df['Team_Overall_Score_Diff'] = df['Blue_Team_Overall_Score'] - df['Red_Team_Overall_Score']
    
    print(f"Added 9 team aggregate score columns")
    print(f"Final shape: {df.shape}")
    
    # Show team score differences
    print("\\n=== Team Score Differences (first 5 frames) ===")
    team_diff_cols = [
        'frame_idx',
        'Team_Offensive_Score_Diff',
        'Team_Defensive_Score_Diff',
        'Team_Overall_Score_Diff'
    ]
    print(df[team_diff_cols].head())
    
    # Save to CSV if path provided
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f"\\nScored DataFrame saved to: {output_csv_path}")
    
    return df

def main():
    """Main function to run the scoring"""
    
    input_file = "featured_data.csv"
    output_file = "featured_data_with_scores.csv"
    
    print("=== League of Legends Player Scoring System ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    print("Scoring System:")
    print("- Offensive Score: Attack/Ability stats with weighted combination")
    print("- Defensive Score: Armor, Magic Resist, Health")
    print("- Sustain Score: Lifesteal and Vamp stats")
    print("- Resources Score: Power/Mana availability")
    print("- Mobility Score: Movement speed")
    print("- Overall Score: Weighted combination of all profiles")
    print()
    
    # Add player scores
    scored_df = add_player_scores(input_file, output_file)
    
    if scored_df is not None:
        print("\\n✅ Player scoring completed successfully!")
        print(f"Total columns: {len(scored_df.columns)}")
        print(f"- Team aggregate features: 28")
        print(f"- Individual player scores: 60 (6 per player × 10 players)")
        print(f"- Team aggregate scores: 9")
        print(f"- Total: {len(scored_df.columns)}")
        print(f"\\nIndividual player stats (240 columns) have been removed to reduce file size.")

if __name__ == "__main__":
    main()

