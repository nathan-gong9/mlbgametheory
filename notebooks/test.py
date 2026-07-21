import pybaseball
import pandas as pd
import sys
from pathlib import Path

sys.path.append('../src')
from feature_engineering import format_pitch_results
from feature_engineering import find_shrink_rate
from feature_engineering import apply_shrinkage

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

print(batter_stats.dtypes[['zone', 'pitch_type', 'prior_pitch_type', 'prior_zone']])

#pitcher_pitch_types = pitcher_stats['pitch_type'].unique()
#filtered_batter_stats = batter_stats[batter_stats['pitch_type'].isin(pitcher_pitch_types)]
#batter_pitches = pd.crosstab(filtered_batter_stats['pitch_type'], filtered_batter_stats["zone"])

#Sends the batter dataframe to smaller dataframes of swings and events in play
batter_stats_swings = batter_stats[batter_stats['swing'] == True]
batter_stats_contact = batter_stats[batter_stats['in_play'] == True]
batter_stats_contact = batter_stats_contact.dropna(subset=['estimated_woba_using_speedangle'])

#Calculates the batter and pitcher swing, whiff, and xwoba values based on pitch type X zone
batter_swing_percentage = batter_stats.groupby(['pitch_type', 'zone'])['swing'].agg(['mean', 'count'])
pitcher_swing_percentage = pitcher_stats.groupby(['pitch_type', 'zone'])['swing'].mean()
batter_whiff_percentage = batter_stats_swings.groupby(['pitch_type', 'zone'])['whiff'].agg(['mean', 'count'])
batter_whiff_percentage = batter_whiff_percentage.reindex(batter_swing_percentage.index, fill_value=0)
batter_contact_quality = batter_stats_contact.groupby(['pitch_type', 'zone'])['estimated_woba_using_speedangle'].agg(['mean', 'count'])
batter_contact_quality = batter_contact_quality.reindex(batter_swing_percentage.index, fill_value=0)

#Fetches baseline values for swing, whiff, and xwoba rates in specific zone X pitch type buckets
league_swing_baseline = pd.read_csv('../data/processed/league_swing_baseline.csv', index_col=['pitch_type', 'zone'])
league_whiff_baseline = pd.read_csv('../data/processed/league_whiff_baseline.csv', index_col=['pitch_type', 'zone'])
league_xwoba_baseline = pd.read_csv('../data/processed/league_xwoba_baseline.csv', index_col=['pitch_type', 'zone'])
batter_swing_percentage['baseline'] = league_swing_baseline
batter_whiff_percentage['baseline'] = league_whiff_baseline
batter_contact_quality['baseline'] = league_xwoba_baseline

#Apply shrinkage rates for smaller sample sizes, pushing values towards league average
shrink_rates = pd.read_csv('../data/processed/shrink_rates.csv', index_col=0)

swing_shrink_rate = shrink_rates.loc["swing"].iloc[0]
whiff_shrink_rate = shrink_rates.loc["whiff"].iloc[0]
xwoba_shrink_rate = shrink_rates.loc["xwoba"].iloc[0]

batter_swing_percentage['adjusted'] = apply_shrinkage(batter_swing_percentage['mean'],
    batter_swing_percentage['baseline'],
    batter_swing_percentage['count'],
    swing_shrink_rate)

batter_whiff_percentage['adjusted'] = apply_shrinkage(batter_whiff_percentage['mean'],
    batter_whiff_percentage['baseline'],
    batter_whiff_percentage['count'],
    whiff_shrink_rate)

batter_contact_quality['adjusted'] = apply_shrinkage(batter_contact_quality['mean'],
    batter_contact_quality['baseline'],
    batter_contact_quality['count'],
    xwoba_shrink_rate)

swing_adjusted = batter_swing_percentage['adjusted'].rename('swing_rate_adjusted')
whiff_adjusted = batter_whiff_percentage['adjusted'].rename('whiff_rate_adjusted')
xwoba_adjusted = batter_contact_quality['adjusted'].rename('xwoba_adjusted')
batter_stats = batter_stats.join(swing_adjusted, on=['pitch_type', 'zone'])
batter_stats = batter_stats.join(whiff_adjusted, on=['pitch_type', 'zone'])
batter_stats = batter_stats.join(xwoba_adjusted, on=['pitch_type', 'zone'])

print(batter_stats['swing_rate_adjusted'].isna().sum())
print(batter_stats[batter_stats['zone'].isna()]['description'].value_counts())
