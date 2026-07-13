'''
Queries the Statcast full league data csv's, transforms the data, extracts 
the necessary data, and puts them into necessary variables
'''

import pybaseball
import pandas as pd
import sys

sys.path.append('../src')
from feature_engineering import add_pitch_result_columns
from feature_engineering import find_shrink_rate

filenames = ['../data/raw/2020season.csv', '../data/raw/2021season.csv', 
            '../data/raw/2022season.csv', '../data/raw/2023season.csv', 
            '../data/raw/2024season.csv', '../data/raw/2025season.csv', 
            '../data/raw/2026season.csv',]

league_data = pd.concat([pd.read_csv(f) for f in filenames])
league_data = add_pitch_result_columns(league_data)

league_data_swings = league_data[league_data['swing'] == True]

league_data_swing_zones = league_data.groupby(['pitch_type', 'zone'])['swing'].mean()
league_data_whiff_zones = league_data_swings.groupby(['pitch_type', 'zone'])['whiff'].mean()

league_data_contact = league_data_swings[league_data_swings['whiff'] == False]

league_data.to_csv('../data/processed/processed_league_data.csv')
league_data_swing_zones.to_csv('../data/processed/league_swing_baseline.csv')
league_data_whiff_zones.to_csv('../data/processed/league_whiff_baseline.csv')

shrink_rates = {
    'swing': find_shrink_rate(league_data, 'swing'),
    'whiff': find_shrink_rate(league_data_swings, 'whiff'),
}
pd.Series(shrink_rates).to_csv('../data/processed/shrink_rates.csv')