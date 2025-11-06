# ESE5380_FINAL


# Riot Games Match Timeline Data Structure

## Raw Timeline Data Structure
```
timeline_data
├── info
│   ├── frames[]                    # Array of game frames (every ~60 seconds)
│   │   ├── timestamp              # Game time in milliseconds
│   │   ├── participantFrames{}    # Stats for all 10 players at this time
│   │   │   ├── "1" to "10"        # Participant IDs
│   │   │   │   ├── championStats
│   │   │   │   │   ├── health, healthMax
│   │   │   │   │   ├── armor, magicResist
│   │   │   │   │   ├── attackDamage, abilityPower
│   │   │   │   │   └── movementSpeed, etc.
│   │   │   │   ├── currentGold, totalGold
│   │   │   │   ├── xp, level
│   │   │   │   ├── damageStats
│   │   │   │   │   ├── totalDamageDone
│   │   │   │   │   ├── totalDamageDoneToChampions
│   │   │   │   │   ├── totalDamageTaken
│   │   │   │   │   ├── magicDamageDone
│   │   │   │   │   ├── physicalDamageDone
│   │   │   │   │   └── trueDamageDone
│   │   │   │   ├── position{x, y}
│   │   │   │   ├── minionsKilled
│   │   │   │   └── jungleMinionsKilled
│   │   │   └── events[]            # Game events in this frame
│   │   │       ├── CHAMPION_KILL
│   │   │       ├── BUILDING_KILL
│   │   │       ├── WARD_PLACED
│   │   │       ├── LEVEL_UP
│   │   │       ├── SKILL_LEVEL_UP
│   │   │       └── GAME_END
│   │   └── gameId
│   └── participants[]              # Player info
│       ├── participantId
│       └── puuid
```

## Processed Data Structure
```
processed_data
├── gold_df                         # DataFrame: timestamp, participant_id, current_gold, total_gold, team
├── xp_df                          # DataFrame: timestamp, participant_id, xp, level, team  
├── health_df                      # DataFrame: timestamp, participant_id, current_health, max_health, health_percentage
├── damage_df                      # DataFrame: timestamp, participant_id, damage stats, damage types
├── position_df                    # DataFrame: timestamp, participant_id, x, y coordinates
├── events_df                      # DataFrame: timestamp, type, participant_id, killer_id, victim_id, etc.
├── participant_info               # DataFrame: participant_id, puuid
├── team_gold                      # Pivot table: Blue vs Red team gold over time
├── team_xp                        # Pivot table: Blue vs Red team XP over time
├── frames_count                   # Number of frames processed
└── match_duration_ms              # Total match duration in milliseconds
```

## Analysis Results Structure
```
analysis_results
├── participant_summary            # DataFrame: Final stats per player
│   ├── participant_id
│   ├── final_gold, final_xp, final_level
│   ├── total_damage, damage_to_champions
│   └── team (Blue/Red)
├── team_stats                     # Dictionary: Team performance metrics
│   ├── blue_total_gold, red_total_gold
│   ├── blue_total_xp, red_total_xp
│   ├── blue_avg_level, red_avg_level
│   └── blue_total_damage, red_total_damage
├── match_duration_minutes         # Match length in minutes
├── total_kills                    # Number of champion kills
├── deaths_by_team                 # Dictionary: Deaths per team
├── towers_destroyed               # Number of towers destroyed
└── events_summary                 # Dictionary: Count of each event type
```



## Data Processing Flow

```
Raw Timeline Data
        ↓
┌─────────────────────────────────────┐
│     process_timeline_data()         │
│  • Extract frames and participants  │
│  • Parse champion stats, gold, XP   │
│  • Collect damage and position data │
│  • Process events (kills, buildings)│
│  • Calculate team-based metrics     │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│         processed_data              │
│  • 6 DataFrames (gold, xp, health, │
│    damage, position, events)        │
│  • Team pivot tables               │
│  • Participant info                │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│    analyze_match_performance()      │
│  • Calculate final stats           │
│  • Compare team performance         │
│  • Count events and kills          │
│  • Generate summary metrics        │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│       analysis_results              │
│  • Participant summaries           │
│  • Team statistics                 │
│  • Match metrics                   │
└─────────────────────────────────────┘
```

## Key Data Transformations

### 1. **Raw → Processed**
- **Nested JSON** → **Flat DataFrames**
- **Participant IDs** → **Team assignments** (Blue: 1-5, Red: 6-10)
- **Milliseconds** → **Time series data**
- **Complex objects** → **Extracted metrics**

### 2. **Processed → Analysis**
- **Time series** → **Final values** (max gold, XP, damage)
- **Individual stats** → **Team aggregates**
- **Event lists** → **Counts and summaries**
- **Raw positions** → **Movement patterns**

### 3. **Data Types**
- **timeline_data**: Raw JSON from Riot API
- **processed_data**: Dictionary of pandas DataFrames
- **analysis_results**: Dictionary of summaries and metrics