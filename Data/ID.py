#!/usr/bin/env python3
"""
Script to extract OP.GG leaderboard data and create DataFrame with gamename/tagline split
"""

import re
import pandas as pd

def create_gamename_tagline_dataframe(xml_file_path, output_csv_path=None):
    """
    Extract IDs and create a DataFrame with gamename and tagline columns
    
    Args:
        xml_file_path (str): Path to the input XML file
        output_csv_path (str): Optional path to save as CSV
    
    Returns:
        pandas.DataFrame: DataFrame with gamename and tagline columns
    """
    
    # Pattern to match and extract ID values from <tr id="..." class="">
    pattern = r'<tr id="([^"]*)" class="">'
    
    try:
        with open(xml_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Find all matches and extract the ID values
        matches = re.findall(pattern, content)
        
        # Lists to store gamename and tagline
        gamename_list = []
        tagline_list = []
        rank_list = []
        
        print("Processing IDs and splitting into gamename/tagline...")
        
        for i, id_value in enumerate(matches, 1):
            # Split the ID by the last hyphen
            # Find the last occurrence of '-'
            last_hyphen_index = id_value.rfind('-')
            
            if last_hyphen_index != -1:
                gamename = id_value[:last_hyphen_index]
                tagline = id_value[last_hyphen_index + 1:]
            else:
                # If no hyphen found, treat the whole thing as gamename
                gamename = id_value
                tagline = ""
            
            gamename_list.append(gamename)
            tagline_list.append(tagline)
            rank_list.append(i)
            
            print(f"Rank {i}: {gamename} | {tagline}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'rank': rank_list,
            'gamename': gamename_list,
            'tagline': tagline_list,
            'full_id': matches
        })
        
        print(f"\\nDataFrame created successfully!")
        print(f"Shape: {df.shape}")
        print(f"\\nFirst 10 rows:")
        print(df.head(10))
        
        # Save to CSV if path provided
        if output_csv_path:
            df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"\\nDataFrame saved to: {output_csv_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{xml_file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """Main function to run the extraction"""
    
    xml_file = "OpggId.xml"
    csv_file = "opgg_leaderboard.csv"
    
    print("=== OP.GG Leaderboard DataFrame Creator ===")
    print(f"Input file: {xml_file}")
    print(f"Output file: {csv_file}")
    print()
    
    # Create DataFrame with gamename and tagline
    print("Creating DataFrame with gamename/tagline split...")
    df = create_gamename_tagline_dataframe(xml_file, csv_file)
    
    if df is not None:
        print("\\n=== DataFrame Summary ===")
        print(f"Total entries: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("\\nSample data:")
        print(df.head(10))
        
        print(f"\\nDataFrame successfully created and saved to {csv_file}")
    else:
        print("Failed to create DataFrame")

if __name__ == "__main__":
    main()
