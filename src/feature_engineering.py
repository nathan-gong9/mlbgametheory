def add_pitch_result_columns(df):
    """
    Adds pitch result columns to a raw Statcast dataframe,
    based on the 'description' field. Also filters out automatic_ball rows,
    since those aren't real pitches.
    """

    pitch_results = df[df["description"] != "automatic_ball"].copy()

    swings = ['swinging_strike', 'hit_into_play', 'foul', 'foul_tip', 'swinging_strike_blocked']
    takes = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    whiffs = ['swinging_strike', 'swinging_strike_blocked']

    pitch_results["swing"] = pitch_results['description'].isin(swings)
    pitch_results["take"] = pitch_results['description'].isin(takes)
    pitch_results["whiff"] = pitch_results['description'].isin(whiffs)

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