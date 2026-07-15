def add_pitch_result_columns(df):
    """
    Adds pitch result columns to a raw Statcast dataframe,
    based on the 'description' field. Also filters out automatic_ball rows,
    since those aren't real pitches.
    """

    pitch_results = df[df["description"] != "automatic_ball"].copy()
    pitch_results = pitch_results[pitch_results['pitch_type'] != 'UN']

    swings = ['swinging_strike', 'hit_into_play', 'foul', 'foul_tip', 'swinging_strike_blocked']
    takes = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    whiffs = ['swinging_strike', 'swinging_strike_blocked']
    in_play = ['hit_into_play']

    pitch_results["swing"] = pitch_results['description'].isin(swings)
    pitch_results["take"] = pitch_results['description'].isin(takes)
    pitch_results["whiff"] = pitch_results['description'].isin(whiffs)
    pitch_results["in_play"] = pitch_results['description'].isin(in_play)

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