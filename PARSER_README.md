# Riot Games Timeline Parser Pipeline

A comprehensive pipeline for collecting and storing Riot Games match timeline data for multiple players.

## Features

- **Batch Processing**: Process multiple players from a DataFrame
- **Rate Limiting**: Respects Riot API rate limits with configurable delays
- **Error Handling**: Robust error handling with retry logic
- **Efficient Storage**: JSON format with match_id as the primary key
- **Progress Tracking**: Detailed logging and statistics
- **Index Creation**: Automatic creation of timeline index for easy access

## Files

- `riot_timeline_parser.py` - Main parser pipeline class
- `riot_parser.py` - Updated with integration functions
- `example_usage.py` - Example usage script
- `data_exploration.ipynb` - Data analysis functions

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas requests
```

### 2. Create Players DataFrame

```python
import pandas as pd

players_df = pd.DataFrame({
    'game_name': ['Arjian', 'Faker', 'Caps'],
    'tag_line': ['NA1', 'KR', 'EUW']
})
```

### 3. Run the Pipeline

```python
from riot_parser import run_timeline_pipeline

parser, results_df, index = run_timeline_pipeline(
    players_df=players_df,
    api_key="YOUR_API_KEY_HERE",
    routing="americas",
    matches_per_player=20,
    output_dir="timeline_data"
)
```

## Detailed Usage

### RiotTimelineParser Class

```python
from riot_timeline_parser import RiotTimelineParser

parser = RiotTimelineParser(
    api_key="RGAPI-your-key-here",
    routing="americas",  # americas, asia, europe, sea
    matches_per_player=20,
    output_dir="timeline_data",
    delay_between_requests=0.1
)
```

### Parameters

- **api_key**: Your Riot Games API key
- **routing**: API routing region (americas, asia, europe, sea)
- **matches_per_player**: Number of recent matches to fetch per player
- **output_dir**: Directory to store timeline JSON files
- **delay_between_requests**: Delay between API requests (seconds)

### Processing Players

```python
# Process single player
result = parser.process_player("Arjian", "NA1")

# Process DataFrame of players
results_df = parser.parse_players_dataframe(players_df)
```

## Output Structure

### Timeline Files
Each match timeline is saved as: `{match_id}_timeline.json`

```
timeline_data/
├── NA1_1234567890_timeline.json
├── NA1_1234567891_timeline.json
├── NA1_1234567892_timeline.json
└── timeline_index.json
```

### Timeline Index
`timeline_index.json` contains metadata for all saved timelines:

```json
{
  "NA1_1234567890": {
    "filepath": "timeline_data/NA1_1234567890_timeline.json",
    "filename": "NA1_1234567890_timeline.json",
    "file_size_bytes": 2048576,
    "modified_time": "2024-01-15T10:30:00",
    "frame_count": 42,
    "duration_ms": 2520000,
    "duration_minutes": 42.0,
    "participants": [...]
  }
}
```

### Processing Results DataFrame

```python
results_df.columns
# ['game_name', 'tag_line', 'puuid', 'match_ids', 'timelines_saved', 'timelines_failed', 'success']
```

## API Rate Limits

The parser includes built-in rate limiting:
- **Personal API Key**: 100 requests per 2 minutes
- **Development API Key**: 100 requests per 2 minutes
- Default delay: 0.1 seconds between requests
- Automatic retry on 429 (rate limit) errors

## Error Handling

The parser handles various error conditions:
- **404**: Player not found
- **429**: Rate limited (automatic retry)
- **Network errors**: Request timeouts and connection issues
- **File I/O errors**: Problems saving timeline data

## Statistics Tracking

The parser tracks comprehensive statistics:

```python
parser.stats
# {
#   'players_processed': 5,
#   'players_failed': 0,
#   'matches_found': 100,
#   'matches_processed': 95,
#   'matches_failed': 5,
#   'total_requests': 210,
#   'start_time': datetime(...),
#   'end_time': datetime(...)
# }
```

## Example Workflow

1. **Prepare Data**: Create DataFrame with player names and tags
2. **Initialize Parser**: Set up with your API key and preferences
3. **Run Pipeline**: Process all players and collect timelines
4. **Analyze Data**: Use the timeline index and saved JSON files
5. **Further Processing**: Use functions from `data_exploration.ipynb`

## Integration with Analysis Functions

The saved timeline data can be used with the analysis functions:

```python
import json
from data_exploration import process_timeline_data, analyze_match_performance

# Load a timeline
with open('timeline_data/NA1_1234567890_timeline.json', 'r') as f:
    timeline_data = json.load(f)

# Process and analyze
processed = process_timeline_data(timeline_data)
analysis = analyze_match_performance(processed)
```

## Tips

- **Start Small**: Test with 5-10 matches per player first
- **Monitor Rate Limits**: Watch for 429 errors in logs
- **Backup Data**: Timeline data is valuable, keep backups
- **Regional Routing**: Use correct routing for player regions
- **API Key Management**: Keep your API key secure and rotate regularly

## Troubleshooting

### Common Issues

1. **403 Forbidden**: API key expired or invalid
2. **429 Too Many Requests**: Reduce `delay_between_requests`
3. **404 Not Found**: Player name/tag incorrect
4. **Timeout Errors**: Increase timeout values or check network

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
