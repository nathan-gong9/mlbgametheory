import pandas as pd

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
    df_swings = df[df['swing'] == True]
    df_contact = df[df['in_play'] == True]
    df_contact = df_contact.dropna(subset=['estimated_woba_using_speedangle'])

    #Calculates the df and pitcher swing, whiff, and xwoba values based on pitch type X zone
    df_swing_percentage = df.groupby(['pitch_type', 'zone'])['swing'].agg(['mean', 'count'])
    df_whiff_percentage = df_swings.groupby(['pitch_type', 'zone'])['whiff'].agg(['mean', 'count'])
    df_whiff_percentage = df_whiff_percentage.reindex(df_swing_percentage.index, fill_value=0)
    df_contact_quality = df_contact.groupby(['pitch_type', 'zone'])['estimated_woba_using_speedangle'].agg(['mean', 'count'])
    df_contact_quality = df_contact_quality.reindex(df_swing_percentage.index, fill_value=0)

    #Fetches baseline values for swing, whiff, and xwoba rates in specific zone X pitch type buckets
    league_swing_baseline = pd.read_csv('../data/processed/league_swing_baseline.csv', index_col=['pitch_type', 'zone'])
    league_whiff_baseline = pd.read_csv('../data/processed/league_whiff_baseline.csv', index_col=['pitch_type', 'zone'])
    league_xwoba_baseline = pd.read_csv('../data/processed/league_xwoba_baseline.csv', index_col=['pitch_type', 'zone'])
    df_swing_percentage['baseline'] = league_swing_baseline
    df_whiff_percentage['baseline'] = league_whiff_baseline
    df_contact_quality['baseline'] = league_xwoba_baseline

    #Apply shrinkage rates for smaller sample sizes, pushing values towards league average
    shrink_rates = pd.read_csv('../data/processed/shrink_rates.csv', index_col=0)

    swing_shrink_rate = shrink_rates.loc["swing"].iloc[0]
    whiff_shrink_rate = shrink_rates.loc["whiff"].iloc[0]
    xwoba_shrink_rate = shrink_rates.loc["xwoba"].iloc[0]

    df_swing_percentage['adjusted'] = apply_shrinkage(df_swing_percentage['mean'],
        df_swing_percentage['baseline'],
        df_swing_percentage['count'],
        swing_shrink_rate)

    df_whiff_percentage['adjusted'] = apply_shrinkage(df_whiff_percentage['mean'],
        df_whiff_percentage['baseline'],
        df_whiff_percentage['count'],
        whiff_shrink_rate)

    df_contact_quality['adjusted'] = apply_shrinkage(df_contact_quality['mean'],
        df_contact_quality['baseline'],
        df_contact_quality['count'],
        xwoba_shrink_rate)

    swing_adjusted = df_swing_percentage['adjusted'].rename('swing_rate_adjusted')
    whiff_adjusted = df_whiff_percentage['adjusted'].rename('whiff_rate_adjusted')
    xwoba_adjusted = df_contact_quality['adjusted'].rename('xwoba_adjusted')

    return swing_adjusted, whiff_adjusted, xwoba_adjusted
