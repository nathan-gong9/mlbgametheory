import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


OUTCOME_MAP = {
        'ball': 'ball',
        'blocked_ball': 'ball',
        'hit_by_pitch': 'ball',
        'called_strike': 'called_strike',
        'swinging_strike': 'swinging_strike',
        'swinging_strike_blocked': 'swinging_strike',
        'foul_tip': 'swinging_strike',
        'foul': 'foul',
        'hit_into_play': 'in_play',
    }

FEATURES = ['balls', 'strikes', 'release_speed', 'pfx_x', 'pfx_z',
            'swing_rate_adjusted', 'whiff_rate_adjusted', 'xwoba_adjusted']

def format_pitch_results(df):
    """
    Adds pitch result columns to a raw Statcast dataframe,
    based on the 'description' field. Also filters out automatic_ball rows,
    since those aren't real pitches.
    """

    pitch_results = df[df["description"] != "automatic_ball"].copy()
    pitch_results = pitch_results[pitch_results['pitch_type'] != 'UN']
    pitch_results = pitch_results[pitch_results['game_type'].isin(['R', 'F', 'D', 'L', 'W'])]

    swings = ['swinging_strike', 'hit_into_play', 'foul', 'foul_tip', 'swinging_strike_blocked']
    takes = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    whiffs = ['swinging_strike', 'swinging_strike_blocked']
    in_play = ['hit_into_play']

    pitch_results["swing"] = pitch_results['description'].isin(swings)
    pitch_results["take"] = pitch_results['description'].isin(takes)
    pitch_results["whiff"] = pitch_results['description'].isin(whiffs)
    pitch_results["in_play"] = pitch_results['description'].isin(in_play)

    pitch_results = pitch_results.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'])
    pitch_results["prior_pitch_type"] = pitch_results.groupby(['game_pk', 'at_bat_number'])['pitch_type'].shift(1)
    pitch_results["prior_zone"] = pitch_results.groupby(['game_pk', 'at_bat_number'])['zone'].shift(1)

    pitch_results = pitch_results.dropna(subset=["zone"])
    
    return pitch_results

def apply_shrinkage(observed_rate, baseline_rate, pitch_count, shrink_size):
    """
    Shrinks a small-sample observed rate toward a prior (baseline) rate,
    weighted by how much data supports the observed rate.

    Uses a weighted average between the observed rate and the prior:
        shrunk = (n * observed_rate + k * prior_rate) / (n + k)

    As n grows large relative to k, the result approaches observed_rate
    (we trust the data). As n shrinks toward zero, the result approaches
    prior_rate (we fall back on the baseline).

    """
    return (pitch_count * observed_rate + shrink_size * baseline_rate) / (pitch_count + shrink_size)

def find_shrink_rate(data, category):
    """
    Finds the shrink rate of a discrete variable to push small sample size pitch type + zone 
    buckets toward the league average
    """
    averages = data.groupby(['pitch_type', 'zone'])[category].agg(['mean', 'count'])
    averages = averages[averages['count'] > 5]
    mean = (averages['mean'] * averages['count']).sum() / averages['count'].sum()
    variance = averages['mean'].var()
    noise = (mean * (1 - mean) / averages['count']).mean()
    true_variance = variance - noise

    if true_variance <= 0:
        return 10
    shrink_rate = mean * (1 - mean) / true_variance - 1

    return shrink_rate

def find_shrink_rate_continuous(data, category):
    """
    Finds the shrink rate of a continuous variable to push small sample size pitch type + zone 
    buckets toward the league average
    """
    averages = data.groupby(['pitch_type', 'zone'])[category].agg(['mean', 'count', 'var'])
    averages = averages[averages['count'] > 5]
    mean = (averages['mean'] * averages['count']).sum() / averages['count'].sum()
    variance = averages['mean'].var()
    noise = (averages['var'] / averages['count']).mean()
    true_variance = variance - noise

    if true_variance <= 0:
        return 10
    shrink_rate = averages['var'].mean() / true_variance - 1

    return shrink_rate

#applies the shrink rates and appends them to the stats of a player
def shrink_features(df):
    """
    Computes a player's pitch-type x zone rates, shrunk toward a two-stage prior.

    Stage 1: the player's overall rate for that pitch type is shrunk toward the
    league rate for that pitch type x zone. This becomes the prior.
    Stage 2: the player's rate in the specific cell is shrunk toward that prior.

    A cell the player never threw (n=0) falls back to the stage-1 prior rather
    than to the raw league baseline, so a pitcher's own tendency with that pitch
    still informs the estimate.
    """
    df_swings = df[df['swing'] == True]
    df_contact = df[df['in_play'] == True]
    df_contact = df_contact.dropna(subset=['estimated_woba_using_speedangle'])

    #Fetches baseline values for swing, whiff, and xwoba rates in specific zone X pitch type buckets
    league_swing_baseline = pd.read_csv('../data/processed/league_swing_baseline.csv', index_col=['pitch_type', 'zone'])
    league_whiff_baseline = pd.read_csv('../data/processed/league_whiff_baseline.csv', index_col=['pitch_type', 'zone'])
    league_xwoba_baseline = pd.read_csv('../data/processed/league_xwoba_baseline.csv', index_col=['pitch_type', 'zone'])

    #Full pitch_type x zone grid, restricted to the pitch types this player throws.
    #Reindexing onto this creates rows for cells the player never threw.
    player_types = df['pitch_type'].unique()
    full_index = league_swing_baseline.index[
        league_swing_baseline.index.get_level_values('pitch_type').isin(player_types)
    ]

    #Cell-level rates, expanded to the full grid
    df_swing_percentage = df.groupby(['pitch_type', 'zone'])['swing'].agg(['mean', 'count']).reindex(full_index)
    df_whiff_percentage = df_swings.groupby(['pitch_type', 'zone'])['whiff'].agg(['mean', 'count']).reindex(full_index)
    df_contact_quality = df_contact.groupby(['pitch_type', 'zone'])['estimated_woba_using_speedangle'].agg(['mean', 'count']).reindex(full_index)

    #Pitch-type-level rates, ignoring zone. Same source frames as above so the
    #denominators match (whiff over swings, xwoba over balls in play).
    swing_by_type = df.groupby('pitch_type')['swing'].agg(['mean', 'count'])
    whiff_by_type = df_swings.groupby('pitch_type')['whiff'].agg(['mean', 'count'])
    xwoba_by_type = df_contact.groupby('pitch_type')['estimated_woba_using_speedangle'].agg(['mean', 'count'])

    df_swing_percentage['baseline'] = league_swing_baseline
    df_whiff_percentage['baseline'] = league_whiff_baseline
    df_contact_quality['baseline'] = league_xwoba_baseline

    #Apply shrinkage rates for smaller sample sizes, pushing values towards league average
    shrink_rates = pd.read_csv('../data/processed/shrink_rates.csv', index_col=0)

    swing_shrink_rate = shrink_rates.loc["swing"].iloc[0]
    whiff_shrink_rate = shrink_rates.loc["whiff"].iloc[0]
    xwoba_shrink_rate = shrink_rates.loc["xwoba"].iloc[0]

    #Broadcast the pitch-type rates across every zone of that pitch type
    type_level = df_swing_percentage.index.get_level_values('pitch_type')

    swing_type_mean = pd.Series(type_level.map(swing_by_type['mean']), index=df_swing_percentage.index)
    swing_type_count = pd.Series(type_level.map(swing_by_type['count']), index=df_swing_percentage.index).fillna(0)
    whiff_type_mean = pd.Series(type_level.map(whiff_by_type['mean']), index=df_whiff_percentage.index)
    whiff_type_count = pd.Series(type_level.map(whiff_by_type['count']), index=df_whiff_percentage.index).fillna(0)
    xwoba_type_mean = pd.Series(type_level.map(xwoba_by_type['mean']), index=df_contact_quality.index)
    xwoba_type_count = pd.Series(type_level.map(xwoba_by_type['count']), index=df_contact_quality.index).fillna(0)

    #Stage 1: player's pitch-type rate shrunk toward the league cell rate
    swing_prior = apply_shrinkage(swing_type_mean.fillna(0), df_swing_percentage['baseline'],
                                  swing_type_count, swing_shrink_rate)
    whiff_prior = apply_shrinkage(whiff_type_mean.fillna(0), df_whiff_percentage['baseline'],
                                  whiff_type_count, whiff_shrink_rate)
    xwoba_prior = apply_shrinkage(xwoba_type_mean.fillna(0), df_contact_quality['baseline'],
                                  xwoba_type_count, xwoba_shrink_rate)

    #Stage 2: player's cell rate shrunk toward that prior
    df_swing_percentage['adjusted'] = apply_shrinkage(df_swing_percentage['mean'].fillna(0),
        swing_prior,
        df_swing_percentage['count'].fillna(0),
        swing_shrink_rate)

    df_whiff_percentage['adjusted'] = apply_shrinkage(df_whiff_percentage['mean'].fillna(0),
        whiff_prior,
        df_whiff_percentage['count'].fillna(0),
        whiff_shrink_rate)

    df_contact_quality['adjusted'] = apply_shrinkage(df_contact_quality['mean'].fillna(0),
        xwoba_prior,
        df_contact_quality['count'].fillna(0),
        xwoba_shrink_rate)

    swing_adjusted = df_swing_percentage['adjusted'].rename('swing_rate_adjusted')
    whiff_adjusted = df_whiff_percentage['adjusted'].rename('whiff_rate_adjusted')
    xwoba_adjusted = df_contact_quality['adjusted'].rename('xwoba_adjusted')

    return swing_adjusted, whiff_adjusted, xwoba_adjusted

def generate_run_value_table(df, woba_scale=1.0):
    """
    Builds the count-only run expectancy table from league-wide pitch data.
    Returns ['balls', 'strikes', 'run_value'].
    """
    df = df.copy()
    pa_ending = df['events'].notna() & (df['woba_denom'] == 1)

    blend = df['woba_value'].where(~df['in_play'],
                                   df['estimated_woba_using_speedangle'])
    df['woba_blend'] = blend.fillna(df['woba_value']).where(pa_ending)
    lgwoba = df.groupby('game_year')['woba_blend'].transform('mean')
    df['pa_run_value'] = (df['woba_blend'] - lgwoba) / woba_scale
    df['pa_run_value'] = df.groupby(['game_pk', 'at_bat_number'])['pa_run_value'].transform('max')
    deduped = df.drop_duplicates(subset=['game_pk', 'at_bat_number', 'balls', 'strikes'])
    return (deduped.groupby(['balls', 'strikes'])['pa_run_value']
                   .mean()
                   .rename('run_value')
                   .reset_index())

def measure_location_error(df, split_by_count=True, min_n=30):
    """
    Measures how spread out a pitcher's actual pitch locations are,
    per pitch type. Feeds the execution-error work in Phase 7.

    Returns one row per slice with the standard deviation of horizontal
    (plate_x) and vertical (plate_z) location, in feet.
    """
    df = df.copy().dropna(subset=['plate_x', 'plate_z'])

    group_cols = ['pitch_type', 'stand']

    if split_by_count:
        # Pitcher is ahead when strikes > balls; behind when balls > strikes.
        # Separating these strips out some deliberate aiming differences.
        df['count_state'] = np.select(
            [df['strikes'] > df['balls'], df['balls'] > df['strikes']],
            ['ahead', 'behind'],
            default='even'
        )
        group_cols.append('count_state')

    out = df.groupby(group_cols).agg(
        n=('plate_x', 'size'),
        mean_x=('plate_x', 'mean'),
        mean_z=('plate_z', 'mean'),
        sd_x=('plate_x', 'std'),
        sd_z=('plate_z', 'std'),
    ).reset_index()

    out = out[out['n'] >= min_n]

    # Strike zone is ~1.42 ft wide. Anything near 1.0 here is too inflated to use.
    out['sd_x_share_of_zone'] = (out['sd_x'] / 1.42).round(2)

    return out.sort_values(['pitch_type', 'n'], ascending=[True, False])

def build_outcome_label(df):
    '''
    Builds the outcome label for different pitch result events
    '''
    df = df.copy()
    df['outcome'] = df['description'].map(OUTCOME_MAP)
    unmapped = df.loc[df['outcome'].isna(), 'description'].value_counts()
    if not unmapped.empty:
        print(f"Dropping {unmapped.sum()} unmapped pitches:\n{unmapped}")
    df = df.dropna(subset=['outcome'])
    return df

def split_data(df, train_frac = 0.8):
    '''
    splits the data into a training split and testing split
    split is based on chronological order of plate appearances
    '''
    df = df.copy()
    df = df.sort_values(by=["game_date", "game_pk", "at_bat_number", "pitch_number"])

    df['pa_id'] = df['game_pk'].astype(str) + '_' + df['at_bat_number'].astype(str)
    unique_plate_appearances = df['pa_id'].unique()
    data_cutoff = int(len(unique_plate_appearances) * train_frac)

    training_pas = unique_plate_appearances[:data_cutoff]
    testing_pas = unique_plate_appearances[data_cutoff:]

    train_data = df[df["pa_id"].isin(training_pas)]
    test_data = df[df["pa_id"].isin(testing_pas)]

    return train_data, test_data

def get_xy(df):
    '''
    returns the x and y features of a dataset
    '''
    x = df[FEATURES]
    y = df['outcome']
    return x, y

def fit_logistic(train_df, test_df):
    '''
    fits and tests a logistic model on the training and test dataframes
    '''
    x_train, y_train = get_xy(train_df)
    x_test, y_test = get_xy(test_df)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_scaled, y_train)

    probs = model.predict_proba(x_test_scaled)
    loss = log_loss(y_test, probs)
    return model, scaler, probs, loss

def measure_benchmark(train_df, test_df):
    '''
    measures the benchmark performance of only using 
    the balls-strikes counts as the predictor
    '''
    x_test, y_test = get_xy(test_df)

    lookup = train_df.groupby(['balls', 'strikes'])['outcome'].value_counts(normalize=True).unstack().fillna(0.001)
    lookup = lookup.div(lookup.sum(axis=1), axis=0)
    benchmark = test_df[['balls', 'strikes']].join(lookup, on=['balls', 'strikes'])
    benchmark_probs = benchmark.drop(columns=['balls', 'strikes'])
    loss = log_loss(y_test, benchmark_probs)
    return lookup, benchmark_probs, loss

def calibrate(y_test, probs, classes, class_name, n_bins=10):
    '''
    check calibration of model to measure the relative correct prediction rate
    '''
    column = list(classes).index(class_name)
    target = probs[:, column]
    results = y_test == class_name
    results = results.reset_index(drop=True)
    df = pd.DataFrame({'predicted': target, 'actual': results})
    df['bin'] = pd.qcut(df['predicted'], n_bins, labels=False, duplicates='drop')
    calibration = df.groupby('bin').agg(
        predicted=('predicted', 'mean'),
        actual=('actual', 'mean'),
        n=('actual', 'size'),
    )
    return calibration

