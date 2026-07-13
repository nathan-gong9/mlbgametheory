'''
Queries full Statcast raw data from current day to 2020
'''

import pybaseball
import pandas as pd
import sys

from pybaseball import cache
cache.enable()

for i in range(2020, 2027):
    start_date = str(i) + "-02-01"
    end_date = str(i) + "-12-30"
    print(f"Pulling {i}...")
    data = pybaseball.statcast(start_date, end_date)
    print(f"Saved {i}, {len(data)} rows")
    filename = '../data/raw/' + str(i) + 'season.csv'
    data.to_csv(filename, index=False)