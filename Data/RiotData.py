import numpy as np
import pandas as pd
import requests
import json

# Riot Games API configuration
API_KEY = "RGAPI-e03327e2-11f8-494a-9086-daf01c1a1144"
BASE_URL = "https://americas.api.riotgames.com"
RIOT_ID = "Arjian"
TAG_LINE = "NA1"

def get_riot_account_data():
    """Fetch account data from Riot Games API"""
    url = f"{BASE_URL}/riot/account/v1/accounts/by-riot-id/{RIOT_ID}/{TAG_LINE}"
    headers = {
        "X-Riot-Token": API_KEY
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching account data: {e}")
        return None

def get_lol_match_ids(puuid, start=0, count=20):
    """Fetch League of Legends match IDs for a given PUUID"""
    url = f"{BASE_URL}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    headers = {
        "X-Riot-Token": API_KEY
    }
    params = {
        "start": start,
        "count": count
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching LoL match IDs: {e}")
        return None

def main():
    # Get account data from Riot API
    account_data = get_riot_account_data()
    
    if account_data:
        print("Riot Account Data:")
        print(f"PUUID: {account_data['puuid']}")
        print(f"Game Name: {account_data['gameName']}")
        print(f"Tag Line: {account_data['tagLine']}")
        
        # Get LoL match IDs using the PUUID
        puuid = account_data['puuid']
        print(f"\nFetching LoL match IDs for PUUID: {puuid}")
        
        match_ids = get_lol_match_ids(puuid)
        
        if match_ids is not None:
            if match_ids:  # If list is not empty
                print(f"Found {len(match_ids)} LoL matches:")
                for i, match_id in enumerate(match_ids, 1):
                    print(f"  {i}. {match_id}")
                
                # Convert to pandas DataFrame
                match_df = pd.DataFrame({'match_id': match_ids})
                print(f"\nLoL Match IDs DataFrame:")
                print(match_df)
                
                # Convert to numpy array for analysis
                match_count = len(match_ids)
                avg_id_length = np.mean([len(match_id) for match_id in match_ids])
                data_array = np.array([match_count, avg_id_length])
                print(f"\nMatch analysis (count, avg_id_length): {data_array}")
                
                # Additional analysis
                print(f"\nMatch ID patterns:")
                print(f"- All match IDs start with 'NA1_': {all(mid.startswith('NA1_') for mid in match_ids)}")
                print(f"- Average match ID length: {avg_id_length:.1f} characters")
                print(f"- Most recent match: {match_ids[0] if match_ids else 'N/A'}")
                print(f"- Oldest match: {match_ids[-1] if match_ids else 'N/A'}")
            else:
                print("No LoL matches found for this PUUID")
                print("This could mean:")
                print("- You haven't played any LoL matches yet")
                print("- Your matches are older than the API retention period")
                print("- There might be an issue with the PUUID")
        
        # Convert account data to pandas DataFrame for analysis
        df = pd.DataFrame([account_data])
        print("\nAccount DataFrame:")
        print(df)
        
        # Convert to numpy array for numerical analysis
        puuid_length = len(account_data['puuid'])
        name_length = len(account_data['gameName'])
        tag_length = len(account_data['tagLine'])
        
        data_array = np.array([puuid_length, name_length, tag_length])
        print(f"\nString lengths as numpy array: {data_array}")
    else:
        print("Failed to fetch account data")

if __name__ == "__main__":
    main()

