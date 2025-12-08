import pandas as pd
import requests
import json
import os
import time
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import urllib.parse
from collections import deque
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiter to respect Riot API limits:
    - 20 requests per second
    - 100 requests per 2 minutes
    """
    
    def __init__(self, requests_per_second: int = 20, requests_per_2min: int = 100):
        self.requests_per_second = requests_per_second
        self.requests_per_2min = requests_per_2min
        
        # Track request timestamps
        self.request_times = deque()  # For per-second tracking
        self.request_times_2min = deque()  # For 2-minute tracking
        
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        now = datetime.now()
        
        # Clean old requests from tracking
        self._clean_old_requests(now)
        
        # Check per-second limit
        if len(self.request_times) >= self.requests_per_second:
            sleep_time = 1.0 - (now - self.request_times[0]).total_seconds()
            if sleep_time > 0:
                logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s for per-second limit")
                time.sleep(sleep_time)
                now = datetime.now()
                self._clean_old_requests(now)
        
        # Check 2-minute limit
        if len(self.request_times_2min) >= self.requests_per_2min:
            sleep_time = 120.0 - (now - self.request_times_2min[0]).total_seconds()
            if sleep_time > 0:
                logger.warning(f"Rate limit: sleeping {sleep_time:.2f}s for 2-minute limit")
                time.sleep(sleep_time)
                now = datetime.now()
                self._clean_old_requests(now)
        
        # Record this request
        self.request_times.append(now)
        self.request_times_2min.append(now)
    
    def _clean_old_requests(self, now: datetime):
        """Remove old request timestamps"""
        # Remove requests older than 1 second
        while self.request_times and (now - self.request_times[0]).total_seconds() >= 1.0:
            self.request_times.popleft()
        
        # Remove requests older than 2 minutes
        while self.request_times_2min and (now - self.request_times_2min[0]).total_seconds() >= 120.0:
            self.request_times_2min.popleft()
    
    def get_status(self) -> Dict:
        """Get current rate limit status"""
        now = datetime.now()
        self._clean_old_requests(now)
        
        return {
            'requests_last_second': len(self.request_times),
            'requests_last_2min': len(self.request_times_2min),
            'per_second_limit': self.requests_per_second,
            'per_2min_limit': self.requests_per_2min
        }


class RiotTimelineParser:
    """
    Parser pipeline for collecting Riot Games match timeline data for multiple players
    """
    
    def __init__(self, api_key: str, routing: str = "americas", 
                 matches_per_player: int = 20, output_dir: str = "data/raw/timeline_data",
                 delay_between_requests: float = 0.1, start_offset: int = 0):
        """
        Initialize the Riot Timeline Parser
        
        Args:
            api_key: Riot Games API key
            routing: API routing region (americas, asia, europe, sea)
            matches_per_player: Number of recent matches to fetch per player
            output_dir: Directory to store timeline data
            delay_between_requests: Delay between API requests to respect rate limits
            start_offset: Number of most recent matches to skip before fetching
                          (e.g., start_offset=200 means skip first 200, then fetch matches_per_player)
        """
        self.api_key = api_key
        self.routing = routing
        self.matches_per_player = matches_per_player
        self.output_dir = os.path.abspath(output_dir)
        self.delay_between_requests = delay_between_requests
        self.start_offset = start_offset
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(requests_per_second=20, requests_per_2min=100)
        
        self.base_url = f"https://{routing}.api.riotgames.com"
        self.headers = {"X-Riot-Token": api_key}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'players_processed': 0,
            'players_failed': 0,
            'matches_found': 0,
            'matches_processed': 0,
            'matches_failed': 0,
            'matches_skipped_duplicates': 0,
            'total_requests': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Track processed match IDs to avoid duplicates across players
        self.processed_match_ids = set()
        
        # Load existing processed matches from output directory
        self._load_existing_matches()
        
        logger.info(f"Initialized RiotTimelineParser with {matches_per_player} matches per player")
        if self.start_offset > 0:
            logger.info(f"Skipping first {self.start_offset} most recent matches per player")
        logger.info(f"Found {len(self.processed_match_ids)} existing matches in output directory")
    
    def is_match_processed(self, match_id: str) -> bool:
        """
        Check if a match ID has already been processed
        
        Args:
            match_id: Match ID to check
            
        Returns:
            True if already processed, False otherwise
        """
        return match_id in self.processed_match_ids
    
    def mark_match_processed(self, match_id: str):
        """
        Mark a match ID as processed
        
        Args:
            match_id: Match ID to mark as processed
        """
        self.processed_match_ids.add(match_id)

    def _match_file_exists(self, match_id: str) -> bool:
        """
        Check whether a timeline file is already present on disk for the match ID.
        """
        filename = f"{match_id}_timeline.json"
        filepath = os.path.join(self.output_dir, filename)
        return os.path.exists(filepath)
    
    def get_unique_matches(self, match_ids: List[str]) -> List[str]:
        """
        Filter out already processed match IDs
        
        Args:
            match_ids: List of match IDs to filter
            
        Returns:
            List of unique (unprocessed) match IDs
        """
        unique_matches = []
        for match_id in match_ids:
            if self.is_match_processed(match_id) or self._match_file_exists(match_id):
                if not self.is_match_processed(match_id):
                    # Ensure in-memory cache stays in sync with disk contents
                    self.processed_match_ids.add(match_id)
                self.stats['matches_skipped_duplicates'] += 1
                logger.debug(f"Skipping duplicate match (already scraped): {match_id}")
                continue
            unique_matches.append(match_id)
        
        return unique_matches
    
    def get_processed_match_count(self) -> int:
        """
        Get the number of unique matches processed
        
        Returns:
            Number of unique match IDs processed
        """
        return len(self.processed_match_ids)
    
    def _load_existing_matches(self):
        """
        Load existing processed matches from the output directory
        This prevents re-processing matches that already exist
        """
        if not os.path.exists(self.output_dir):
            return
        
        for filename in os.listdir(self.output_dir):
            if not filename.endswith('_timeline.json'):
                continue
            match_id = filename.replace('_timeline.json', '')
            self.processed_match_ids.add(match_id)
    
    def get_puuid_by_riot_id(self, game_name: str, tag_line: str) -> Optional[str]:
        """
        Get PUUID using Account API v1 by Riot ID
        
        Args:
            game_name: Player's game name
            tag_line: Player's tag line
            
        Returns:
            PUUID string if successful, None otherwise
        """
        url = f"{self.base_url}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        
        try:
            # Apply rate limiting before making request
            self.rate_limiter.wait_if_needed()
            
            response = requests.get(url, headers=self.headers, timeout=10)
            self.stats['total_requests'] += 1
            
            if response.status_code == 200:
                data = response.json()
                return data.get('puuid')
            elif response.status_code == 404:
                logger.warning(f"Player not found: {game_name}#{tag_line}")
                return None
            elif response.status_code == 429:
                logger.warning(f"Rate limited for {game_name}#{tag_line}, waiting...")
                # Wait longer for rate limit reset
                time.sleep(5)
                return self.get_puuid_by_riot_id(game_name, tag_line)  # Retry
            else:
                logger.error(f"API error for {game_name}#{tag_line}: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {game_name}#{tag_line}: {e}")
            return None
    
    def get_match_ids(self, puuid: str, count: int = None, queue: int = None, start_offset: int = None) -> List[str]:
        """
        Get recent match IDs for a player
        
        Args:
            puuid: Player's PUUID
            count: Number of matches to fetch (defaults to self.matches_per_player)
            queue: Queue ID to filter matches (e.g., 420 for ranked solo, 440 for ranked flex)
            start_offset: Number of most recent matches to skip (defaults to self.start_offset)
            
        Returns:
            List of match IDs
        """
        if count is None:
            count = self.matches_per_player
        if start_offset is None:
            start_offset = self.start_offset
            
        url = f"{self.base_url}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"start": start_offset, "count": count}
        
        # Add queue filter if specified
        if queue is not None:
            params["queue"] = queue
        
        try:
            # Apply rate limiting before making request
            self.rate_limiter.wait_if_needed()
            
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            self.stats['total_requests'] += 1
            
            if response.status_code == 200:
                match_ids = response.json()
                self.stats['matches_found'] += len(match_ids)
                return match_ids
            elif response.status_code == 429:
                logger.warning(f"Rate limited for PUUID {puuid}, waiting...")
                # Wait longer for rate limit reset
                time.sleep(5)
                return self.get_match_ids(puuid, count, queue, start_offset)  # Retry
            else:
                error_msg = f"Error getting match IDs for PUUID {puuid}: {response.status_code}"
                try:
                    error_body = response.json()
                    if isinstance(error_body, dict) and "status" in error_body:
                        error_msg += f" - {error_body.get('status', {}).get('message', 'Unknown error')}"
                    logger.error(error_msg)
                except:
                    logger.error(f"{error_msg} - Response: {response.text[:200]}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error getting match IDs for PUUID {puuid}: {e}")
            return []
    
    def get_timeline(self, match_id: str) -> Optional[Dict]:
        """
        Get timeline data for a specific match
        
        Args:
            match_id: Match ID to fetch timeline for
            
        Returns:
            Timeline data dictionary if successful, None otherwise
        """
        url = f"{self.base_url}/lol/match/v5/matches/{match_id}/timeline"
        
        try:
            # Apply rate limiting before making request
            self.rate_limiter.wait_if_needed()
            
            response = requests.get(url, headers=self.headers, timeout=20)
            self.stats['total_requests'] += 1
            
            if response.status_code == 200:
                timeline_data = response.json()
                self.stats['matches_processed'] += 1
                return timeline_data
            elif response.status_code == 429:
                logger.warning(f"Rate limited for match {match_id}, waiting...")
                # Wait longer for rate limit reset
                time.sleep(5)
                return self.get_timeline(match_id)  # Retry
            else:
                logger.error(f"Error getting timeline for match {match_id}: {response.status_code}")
                self.stats['matches_failed'] += 1
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error getting timeline for match {match_id}: {e}")
            self.stats['matches_failed'] += 1
            return None
    
    def save_timeline(self, match_id: str, timeline_data: Dict, player_info: Dict = None) -> str:
        """
        Save timeline data to JSON file with player metadata
        
        Args:
            match_id: Match ID (used as filename)
            timeline_data: Timeline data to save
            player_info: Dictionary with player information (game_name, tag_line, puuid)
            
        Returns:
            Path to saved file
        """
        filename = f"{match_id}_timeline.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Add player metadata to timeline data
            if player_info:
                timeline_data['scraped_by'] = {
                    'game_name': player_info.get('game_name'),
                    'tag_line': player_info.get('tag_line'),
                    'puuid': player_info.get('puuid'),
                    'scraped_at': datetime.now().isoformat()
                }
            
            with open(filepath, 'w') as f:
                json.dump(timeline_data, f, indent=2)
            return filepath
        except Exception as e:
            logger.error(f"Error saving timeline for match {match_id}: {e}")
            return None
    
    def process_player(self, game_name: str, tag_line: str, queue: int = None) -> Dict:
        """
        Process a single player: get PUUID, match IDs, and timelines
        
        Args:
            game_name: Player's game name
            tag_line: Player's tag line
            queue: Queue ID to filter matches (e.g., 420 for ranked solo, 440 for ranked flex)
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing player: {game_name}#{tag_line}")
        
        result = {
            'game_name': game_name,
            'tag_line': tag_line,
            'puuid': None,
            'match_ids': [],
            'timelines_saved': [],
            'timelines_failed': [],
            'success': False
        }
        
        # Get PUUID
        puuid = self.get_puuid_by_riot_id(game_name, tag_line)
        if not puuid:
            logger.error(f"Failed to get PUUID for {game_name}#{tag_line}")
            self.stats['players_failed'] += 1
            return result
        
        result['puuid'] = puuid
        
        # Get match IDs
        match_ids = self.get_match_ids(puuid, queue=queue)
        if not match_ids:
            logger.warning(f"No match IDs found for {game_name}#{tag_line}")
            self.stats['players_failed'] += 1
            return result
        
        # Filter out duplicate matches
        unique_match_ids = self.get_unique_matches(match_ids)
        original_count = len(match_ids)
        unique_count = len(unique_match_ids)
        
        if original_count > unique_count:
            logger.info(f"Filtered {original_count - unique_count} duplicate matches for {game_name}#{tag_line}")
        
        result['match_ids'] = match_ids  # Keep original for reference
        result['unique_match_ids'] = unique_match_ids
        
        # Get and save timelines only for unique matches
        player_info = {
            'game_name': game_name,
            'tag_line': tag_line,
            'puuid': puuid
        }
        
        # Process matches with progress bar
        for match_id in tqdm(unique_match_ids, desc=f"Processing matches for {game_name}#{tag_line}", 
                           leave=False, disable=len(unique_match_ids) <= 1):
            timeline_data = self.get_timeline(match_id)
            if timeline_data:
                filepath = self.save_timeline(match_id, timeline_data, player_info)
                if filepath:
                    result['timelines_saved'].append(match_id)
                    # Mark as processed after successful save
                    self.mark_match_processed(match_id)
                else:
                    result['timelines_failed'].append(match_id)
            else:
                result['timelines_failed'].append(match_id)
        
        result['success'] = len(result['timelines_saved']) > 0
        if result['success']:
            self.stats['players_processed'] += 1
        else:
            self.stats['players_failed'] += 1
        
        logger.info(f"Completed {game_name}#{tag_line}: {len(result['timelines_saved'])} timelines saved")
        return result
    
    def parse_players_dataframe(self, df: pd.DataFrame, queue: int = None) -> pd.DataFrame:
        """
        Parse timeline data for all players in a DataFrame
        
        Args:
            df: DataFrame with columns 'game_name' and 'tag_line'
            queue: Queue ID to filter matches (e.g., 420 for ranked solo, 440 for ranked flex)
            
        Returns:
            DataFrame with processing results for each player
        """
        logger.info(f"Starting to process {len(df)} players")
        self.stats['start_time'] = datetime.now()
        
        results = []
        
        # Create progress bar for players
        player_progress = tqdm(df.iterrows(), total=len(df), desc="Processing players", 
                             unit="player", dynamic_ncols=True)
        
        for idx, row in player_progress:
            game_name = row['game_name']
            tag_line = row['tag_line']
            
            # Update progress bar description
            player_progress.set_description(f"Processing {game_name}#{tag_line}")
            
            result = self.process_player(game_name, tag_line, queue)
            results.append(result)
            
            # Update progress bar with stats
            player_progress.set_postfix({
                'Processed': self.stats['players_processed'],
                'Failed': self.stats['players_failed'],
                'Matches': self.stats['matches_processed']
            })
            
            # Progress update (keep existing logging)
            if (idx + 1) % 5 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} players")
        
        self.stats['end_time'] = datetime.now()
        self._print_final_stats()
        
        return pd.DataFrame(results)
    
    def create_timeline_index(self) -> Dict[str, Dict]:
        """
        Create an index of all saved timeline files
        
        Returns:
            Dictionary with match_id as key and metadata as value
        """
        index = {}
        
        # Get list of timeline files first
        timeline_files = [f for f in os.listdir(self.output_dir) if f.endswith('_timeline.json')]
        
        for filename in tqdm(timeline_files, desc="Creating timeline index", unit="file"):
            match_id = filename.replace('_timeline.json', '')
            filepath = os.path.join(self.output_dir, filename)
            
            try:
                # Get file stats
                stat = os.stat(filepath)
                file_size = stat.st_size
                modified_time = datetime.fromtimestamp(stat.st_mtime)
                
                # Load timeline to get basic info
                with open(filepath, 'r') as f:
                    timeline_data = json.load(f)
                
                frame_count = len(timeline_data.get('info', {}).get('frames', []))
                duration_ms = timeline_data.get('info', {}).get('frames', [{}])[-1].get('timestamp', 0) if frame_count > 0 else 0
                
                # Get player information if available
                scraped_by = timeline_data.get('scraped_by', {})
                
                index[match_id] = {
                    'filepath': filepath,
                    'filename': filename,
                    'file_size_bytes': file_size,
                    'modified_time': modified_time.isoformat(),
                    'frame_count': frame_count,
                    'duration_ms': duration_ms,
                    'duration_minutes': duration_ms / (1000 * 60),
                    'participants': timeline_data.get('info', {}).get('participants', []),
                    'scraped_by': scraped_by
                }
                
            except Exception as e:
                logger.error(f"Error indexing file {filename}: {e}")
        
        # Save index
        index_path = os.path.join(self.output_dir, 'timeline_index.json')
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2, default=str)
        
        logger.info(f"Created timeline index with {len(index)} matches")
        return index
    
    def get_rate_limit_status(self) -> Dict:
        """Get current rate limit status"""
        return self.rate_limiter.get_status()
    
    def _print_final_stats(self):
        """Print final processing statistics"""
        duration = self.stats['end_time'] - self.stats['start_time']
        
        logger.info("=== PROCESSING COMPLETE ===")
        logger.info(f"Total time: {duration}")
        logger.info(f"Players processed: {self.stats['players_processed']}")
        logger.info(f"Players failed: {self.stats['players_failed']}")
        logger.info(f"Matches found: {self.stats['matches_found']}")
        logger.info(f"Matches processed: {self.stats['matches_processed']}")
        logger.info(f"Matches skipped (duplicates): {self.stats['matches_skipped_duplicates']}")
        logger.info(f"Matches failed: {self.stats['matches_failed']}")
        logger.info(f"Unique matches processed: {self.get_processed_match_count()}")
        logger.info(f"Total API requests: {self.stats['total_requests']}")
        logger.info(f"Success rate: {self.stats['matches_processed']/max(self.stats['matches_found'], 1)*100:.1f}%")
        logger.info(f"Duplicate avoidance: {self.stats['matches_skipped_duplicates']} matches skipped")
        
        # Rate limit status
        rate_status = self.get_rate_limit_status()
        logger.info(f"Rate limit status: {rate_status['requests_last_second']}/{rate_status['per_second_limit']} per second, {rate_status['requests_last_2min']}/{rate_status['per_2min_limit']} per 2 minutes")


def main():
    """
    Example usage of the RiotTimelineParser
    """
    # # Example DataFrame with players
    # players_df = pd.DataFrame({
    #     'game_name': ['Arjian', 'cant type', 'Cupic'],
    #     'tag_line': ['NA1', '1998', 'Hwei']
    # })

    players_df = pd.read_csv('data/raw/opgg_leaderboard.csv')
    
    # Rename columns to match expected format
    players_df = players_df.rename(columns={
        'gamename': 'game_name',
        'tagline': 'tag_line'
    })
    
    # Initialize parser
    parser = RiotTimelineParser(
        api_key="RGAPI-e0bdd094-90f0-40c7-992a-34de4a8d0978",
        routing="americas",
        matches_per_player=100,  # Number of matches to fetch per player
        start_offset=200,  # Skip first 200 most recent matches, then fetch 100
        output_dir="timeline_data"
        # Rate limiting is now handled automatically by the RateLimiter class
    )
    
    # Process players (420 = Ranked Solo/Duo, 440 = Ranked Flex)
    results_df = parser.parse_players_dataframe(players_df, queue=420)
    
    # Create index
    index = parser.create_timeline_index()
    
    print("Processing complete!")
    print(f"Results saved to: {parser.output_dir}")
    print(f"Index saved to: {os.path.join(parser.output_dir, 'timeline_index.json')}")


if __name__ == "__main__":
    main()
