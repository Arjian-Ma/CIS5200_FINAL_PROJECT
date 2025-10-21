
"""
Riot Games API Parser - Basic Functions for Single Operations

Simple sequential functions for testing and single operations.
Use riot_timeline_parser.py for batch processing multiple players.
"""

import requests
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

TIMEOUT = 10
API_KEY = "RGAPI-6b41f36d-67bb-4648-8d64-35f0842f6c06"
ROUTING = "americas"  # americas | asia | europe | sea
BASE = f"https://{ROUTING}.api.riotgames.com"
HEADERS = {"X-Riot-Token": API_KEY}

# ============================================================================
# CORE API FUNCTIONS
# ============================================================================

def get_puuid_by_riot_id(game_name: str, tag_line: str, api_key: str = API_KEY):
    """
    Get PUUID using Account API v1 by Riot ID (gameName + tagLine)
    This is the newer, recommended approach instead of Summoner API v4
    """
    base_url = "https://americas.api.riotgames.com"
    url = f"{base_url}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    headers = {"X-Riot-Token": api_key}
    
    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status_code": response.status_code, "text": response.text}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def get_match_ids(puuid: str, start: int = 0, count: int = 5):
    """
    Get recent match IDs for a player
    """
    url = f"{BASE}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"start": start, "count": count}
    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    return r.json()  # list[str] of matchIds

def get_timeline(match_id: str):
    """
    Get timeline data for a specific match
    """
    url = f"{BASE}/lol/match/v5/matches/{match_id}/timeline"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()  # TimelineDto

#

# ============================================================================
# MAIN TESTING SECTION
# ============================================================================

def test_single_player_single_match():
    """
    Test function: Get a single player's data and fetch one match timeline
    """
    print("=== Testing Single Player, Single Match ===")
    
    # Step 1: Get PUUID
    print("\n1. Getting PUUID for Arjian#NA1...")
    puuid_result = get_puuid_by_riot_id("Arjian", "NA1", API_KEY)
    
    if puuid_result.get('puuid'):
        puuid = puuid_result['puuid']
        print(f"✅ Success! PUUID: {puuid[:20]}...")
        
        # Step 2: Get recent match IDs
        print("\n2. Getting recent match IDs...")
        match_ids = get_match_ids(puuid, count=3)
        print(f"✅ Found {len(match_ids)} matches: {match_ids[:2]}...")
        
        # Step 3: Get timeline for first match
        if match_ids:
            print(f"\n3. Getting timeline for match: {match_ids[0]}")
            timeline = get_timeline(match_ids[0])
            
            if timeline:
                print("✅ Success! Timeline data retrieved")
                print(f"   - Frames: {len(timeline['info']['frames'])}")
                print(f"   - Duration: {timeline['info']['frames'][-1]['timestamp']/1000/60:.1f} minutes")
                print(f"   - Participants: {len(timeline['info']['participants'])}")
                
                return timeline
            else:
                print("❌ Failed to get timeline")
        else:
            print("❌ No match IDs found")
    else:
        print(f"❌ Failed to get PUUID: {puuid_result}")
    
    return None



def main():
    """
    Main function to run tests
    """
    print("Riot Games API Parser - Testing Functions")
    print("=" * 50)
    
    # Test 1: Single player, single match
    timeline = test_single_player_single_match()
    
    
    print("\n" + "=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    main()