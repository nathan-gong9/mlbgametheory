import pybaseball
import pandas as pd
import sys

sys.path.append('../src')
from feature_engineering import add_pitch_result_columns
from feature_engineering import apply_shrinkage

SHRINK_SIZE = ???


pitcher = pybaseball.playerid_lookup('ohtani', 'shohei')
pitcher_id = pitcher['key_mlbam'].iloc[0]

batter = pybaseball.playerid_lookup('trout', 'mike')
batter_id = batter['key_mlbam'].iloc[0]

pitcher_stats = pybaseball.statcast_pitcher('2020-03-20', '2023-03-20', pitcher_id)
batter_stats = pybaseball.statcast_batter('2020-03-20', '2023-03-20', batter_id)

pitcher_stats = pitcher_stats[pitcher_stats['game_type'].isin(['R', 'F', 'D', 'L', 'W'])]
batter_stats = batter_stats[batter_stats['game_type'].isin(['R', 'F', 'D', 'L', 'W'])]

#pitcher_stats.to_csv('../data/raw/ohtani_20-23.csv', index=False)
#batter_stats.to_csv('../data/raw/trout_20-23.csv', index=False)

pitcher_pitch_types = pitcher_stats['pitch_type'].unique()

batter_stats = add_pitch_result_columns(batter_stats)
pitcher_stats = add_pitch_result_columns(pitcher_stats)

filtered_batter_stats = batter_stats[batter_stats['pitch_type'].isin(pitcher_pitch_types)]

batter_pitches = pd.crosstab(filtered_batter_stats['pitch_type'], filtered_batter_stats["zone"])
print(batter_pitches)

batter_swing_percentage = batter_stats.groupby(['pitch_type', 'zone'])['swing'].mean()
print(batter_swing_percentage.head())

pitcher_swing_percentage = pitcher_stats.groupby(['pitch_type', 'zone'])['swing'].mean()
print(pitcher_swing_percentage.head())




