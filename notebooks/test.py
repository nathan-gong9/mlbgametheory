import pybaseball
import pandas as pd
import sys
from pathlib import Path

sys.path.append('../src')
from feature_engineering import (format_pitch_results, find_shrink_rate, apply_shrinkage, 
                                shrink_features, measure_location_error, build_outcome_label, 
                                split_data, get_xy, fit_logistic, measure_benchmark, calibrate)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

#Determines the range of relevant data and the two specific players
MATCHUP_YEAR = 2023
PITCHER = "Shohei Ohtani"
BATTER = "Mike Trout"

pitcher_first_name, pitcher_last_name = PITCHER.lower().split(" ")
batter_first_name, batter_last_name = BATTER.lower().split(" ")

batter_path = Path(f'../data/raw/{batter_last_name}_{MATCHUP_YEAR}_batting.csv')
pitcher_path = Path(f'../data/raw/{pitcher_last_name}_{MATCHUP_YEAR}_pitching.csv')

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
    pitcher_stats = pitcher_stats.tail(5500)
    batter_stats = batter_stats.tail(5500)

    #Sends the player data to unique csv's

    pitcher_stats.to_csv(pitcher_path, index=False)
    batter_stats.to_csv(batter_path, index=False)

#pitcher_pitch_types = pitcher_stats['pitch_type'].unique()
#filtered_batter_stats = batter_stats[batter_stats['pitch_type'].isin(pitcher_pitch_types)]
#batter_pitches = pd.crosstab(filtered_batter_stats['pitch_type'], filtered_batter_stats["zone"])

pitcher_stats = build_outcome_label(pitcher_stats)
batter_stats = build_outcome_label(batter_stats)

location_error = measure_location_error(pitcher_stats)
location_error.to_csv('../data/processed/ohtani_location_error.csv', index=False)

pitcher_train, pitcher_test = split_data(pitcher_stats)
batter_train, batter_test = split_data(batter_stats)

#Sends the batter dataframe to smaller dataframes of swings and events in play
batter_swing_adjusted, batter_whiff_adjusted, batter_xwoba_adjusted = shrink_features(batter_train)
pitcher_swing_adjusted, pitcher_whiff_adjusted, pitcher_xwoba_adjusted = shrink_features(pitcher_train)

batter_train = batter_train.join(batter_swing_adjusted, on=['pitch_type', 'zone'])
batter_train = batter_train.join(batter_whiff_adjusted, on=['pitch_type', 'zone'])
batter_train = batter_train.join(batter_xwoba_adjusted, on=['pitch_type', 'zone'])

pitcher_train = pitcher_train.join(pitcher_swing_adjusted, on=['pitch_type', 'zone'])
pitcher_train = pitcher_train.join(pitcher_whiff_adjusted, on=['pitch_type', 'zone'])
pitcher_train = pitcher_train.join(pitcher_xwoba_adjusted, on=['pitch_type', 'zone'])

batter_test = batter_test.join(batter_swing_adjusted, on=['pitch_type', 'zone'])
batter_test = batter_test.join(batter_whiff_adjusted, on=['pitch_type', 'zone'])
batter_test = batter_test.join(batter_xwoba_adjusted, on=['pitch_type', 'zone'])

pitcher_test = pitcher_test.join(pitcher_swing_adjusted, on=['pitch_type', 'zone'])
pitcher_test = pitcher_test.join(pitcher_whiff_adjusted, on=['pitch_type', 'zone'])
pitcher_test = pitcher_test.join(pitcher_xwoba_adjusted, on=['pitch_type', 'zone'])

pitcher_logistic_model, pitcher_scaler, pitcher_probs, pitcher_log_loss = fit_logistic(pitcher_train, pitcher_test)
pitcher_lookup, pitcher_benchmark_probs, pitcher_benchmark_loss = measure_benchmark(pitcher_train, pitcher_test)
print(pitcher_log_loss)
print(pitcher_benchmark_loss)

batter_logistic_model, batter_scaler, batter_probs, batter_log_loss = fit_logistic(batter_train, batter_test)
batter_lookup, batter_benchmark_probs, batter_benchmark_loss = measure_benchmark(batter_train, batter_test)
print(batter_log_loss)
print(batter_benchmark_loss)

pitcher_x_test, pitcher_y_test = get_xy(pitcher_test)

print(calibrate(pitcher_y_test, pitcher_probs, pitcher_logistic_model.classes_, 'swinging_strike'))
