import pybaseball
import pandas as pd
import sys

sys.path.append('../src')
from feature_engineering import add_pitch_result_columns
from feature_engineering import find_shrink_rate
from feature_engineering import apply_shrinkage

MATCHUP_YEAR = 2023

pitcher = pybaseball.playerid_lookup('ohtani', 'shohei')
pitcher_id = pitcher['key_mlbam'].iloc[0]

batter = pybaseball.playerid_lookup('trout', 'mike')
batter_id = batter['key_mlbam'].iloc[0]

start_year = MATCHUP_YEAR - 5
end_date = str(MATCHUP_YEAR) + "-03-20"
start_date = str(start_year) + "-12-01"

pitcher_stats = pybaseball.statcast_pitcher(start_date, end_date, pitcher_id)
batter_stats = pybaseball.statcast_batter(start_date, end_date, batter_id)

pitcher_stats = pitcher_stats[pitcher_stats['game_type'].isin(['R', 'F', 'D', 'L', 'W'])]
batter_stats = batter_stats[batter_stats['game_type'].isin(['R', 'F', 'D', 'L', 'W'])]

pitcher_stats = pitcher_stats.sort_values(by='game_date', ascending=False)
batter_stats = batter_stats.sort_values(by='game_date', ascending=False)

batter_stats = add_pitch_result_columns(batter_stats)
pitcher_stats = add_pitch_result_columns(pitcher_stats)

pitcher_stats = pitcher_stats.head(5500)
batter_stats = batter_stats.head(5500)

pitcher_stats.to_csv('../data/raw/ohtani_2023.csv', index=False)
batter_stats.to_csv('../data/raw/trout_2023.csv', index=False)

pitcher_pitch_types = pitcher_stats['pitch_type'].unique()

filtered_batter_stats = batter_stats[batter_stats['pitch_type'].isin(pitcher_pitch_types)]

batter_pitches = pd.crosstab(filtered_batter_stats['pitch_type'], filtered_batter_stats["zone"])

batter_swing_percentage = batter_stats.groupby(['pitch_type', 'zone'])['swing'].agg(['mean', 'count'])
pitcher_swing_percentage = pitcher_stats.groupby(['pitch_type', 'zone'])['swing'].mean()

league_swing_baseline = pd.read_csv('../data/processed/league_swing_baseline.csv', index_col=['pitch_type', 'zone'])
batter_swing_percentage['baseline'] = league_swing_baseline

shrink_rates = pd.read_csv('../data/processed/shrink_rates.csv', index_col=0)
print(shrink_rates)

swing_shrink_rate = shrink_rates.loc["swing"].iloc[0]
print(swing_shrink_rate)

batter_swing_percentage['adjusted'] = apply_shrinkage(batter_swing_percentage['mean'],
    batter_swing_percentage['baseline'],
    batter_swing_percentage['count'],
    swing_shrink_rate)

print(batter_swing_percentage)
