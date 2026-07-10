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