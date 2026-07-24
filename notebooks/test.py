import pybaseball
import pandas as pd
import sys
from pathlib import Path

sys.path.append('../src')
from feature_engineering import format_pitch_results, find_shrink_rate, apply_shrinkage, shrink_features

#Determines the range of relevant data and the two specific players
MATCHUP_YEAR = 2023
PITCHER = "Shohei Ohtani"
BATTER = "Mike Trout"


pitcher_first_name, pitcher_last_name = PITCHER.lower().split(" ")
batter_first_name, batter_last_name = BATTER.lower().split(" ")

batter_path = Path(f'../data/raw/{batter_last_name}_{MATCHUP_YEAR}.csv')
pitcher_path = Path(f'../data/raw/{pitcher_last_name}_{MATCHUP_YEAR}.csv')

if batter_path.is_file() and pitcher_path.is_file():
    batter_stats = pd.read_csv(batter_path)
    pitcher_stats = pd.read_csv(pitcher_path)
else:
    #Fetch the data for the specific pitcher and batter
    pitcher = pybaseball.playerid_lookup(pitcher_last_name, pitcher_first_name)
    pitcher_id = pitcher['key_mlbam'].iloc[0]
    batter = pybaseball.playerid_lookup(batter_last_name, batter_first_name)
    batter_id = batter['key_mlbam'].iloc[0]

    start_year = MATCHUP_YEAR - 5
    end_date = str(MATCHUP_YEAR) + "-03-20"
    start_date = str(start_year) + "-12-01"

    #Query and format the stats for the players
    pitcher_stats = pybaseball.statcast_pitcher(start_date, end_date, pitcher_id)
    batter_stats = pybaseball.statcast_batter(start_date, end_date, batter_id)
    batter_stats = format_pitch_results(batter_stats)
    pitcher_stats = format_pitch_results(pitcher_stats)
    pitcher_stats = pitcher_stats.head(5500)
    batter_stats = batter_stats.head(5500)

    #Sends the player data to unique csv's

    pitcher_stats.to_csv(f'../data/raw/{pitcher_last_name}_{MATCHUP_YEAR}.csv', index=False)
    batter_stats.to_csv(f'../data/raw/{batter_last_name}_{MATCHUP_YEAR}.csv', index=False)

#pitcher_pitch_types = pitcher_stats['pitch_type'].unique()
#filtered_batter_stats = batter_stats[batter_stats['pitch_type'].isin(pitcher_pitch_types)]
#batter_pitches = pd.crosstab(filtered_batter_stats['pitch_type'], filtered_batter_stats["zone"])

#Sends the batter dataframe to smaller dataframes of swings and events in play
batter_swing_adjusted, batter_whiff_adjusted, batter_xwoba_adjusted = shrink_features(batter_stats)
batter_stats = batter_stats.join(batter_swing_adjusted, on=['pitch_type', 'zone'])
batter_stats = batter_stats.join(batter_whiff_adjusted, on=['pitch_type', 'zone'])
batter_stats = batter_stats.join(batter_xwoba_adjusted, on=['pitch_type', 'zone'])

pitcher_swing_adjusted, pitcher_whiff_adjusted, pitcher_xwoba_adjusted = shrink_features(pitcher_stats)
pitcher_stats = pitcher_stats.join(pitcher_swing_adjusted, on=['pitch_type', 'zone'])
pitcher_stats = pitcher_stats.join(pitcher_whiff_adjusted, on=['pitch_type', 'zone'])
pitcher_stats = pitcher_stats.join(pitcher_xwoba_adjusted, on=['pitch_type', 'zone'])

