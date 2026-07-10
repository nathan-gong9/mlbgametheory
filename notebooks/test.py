import pybaseball
import pandas as pd

ohtani = pybaseball.playerid_lookup('ohtani', 'shohei')
ohtani_id = ohtani['key_mlbam'].iloc[0]

trout = pybaseball.playerid_lookup('trout', 'mike')
trout_id = trout['key_mlbam'].iloc[0]

ohtani_stats = pybaseball.statcast_pitcher('2020-03-20', '2023-03-20', ohtani_id)
trout_stats = pybaseball.statcast_batter('2020-03-20', '2023-03-20', trout_id)

ohtani_stats = ohtani_stats[ohtani_stats['game_type'].isin(['R', 'F', 'D', 'L', 'W'])]
trout_stats = trout_stats[trout_stats['game_type'].isin(['R', 'F', 'D', 'L', 'W'])]

ohtani_stats.to_csv('../data/raw/ohtani_20-23.csv', index=False)
trout_stats.to_csv('../data/raw/trout_20-23.csv', index=False)

ohtani_pitch_types = ohtani_stats['pitch_type'].value_counts()
ohtani_null_pitches = ohtani_stats['delta_run_exp'].isnull().sum()
ohtani_zones = ohtani_stats['zone'].unique()
ohtani_balls_and_strikes = pd.crosstab(ohtani_stats['pitch_type'], [ohtani_stats['balls'], ohtani_stats['strikes']])

trout_pitch_types = trout_stats['pitch_type'].value_counts()
trout_null_pitches = trout_stats['delta_run_exp'].isnull().sum()
trout_zones = trout_stats['zone'].unique()
trout_balls_and_strikes = pd.crosstab(trout_stats['pitch_type'], [trout_stats['balls'], trout_stats['strikes']])

print(trout_pitch_types)
print(trout_null_pitches)
print(trout_zones)
print(trout_balls_and_strikes)

print(trout_stats['description'].unique())


