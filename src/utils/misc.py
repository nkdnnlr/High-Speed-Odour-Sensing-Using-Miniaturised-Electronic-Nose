import pandas as pd

def get_trial_df(trial, data_dir, ms_start=None, ms_stop=None):
    # """ Returns the trial data as a pandas dataframe. """
    # Load data file
    trial_path = data_dir.joinpath(trial['condition'], trial['kind'], trial['trial_id']+f'.csv')
    trial_df = pd.read_csv(trial_path, engine='pyarrow')

    # Crop in time
    if ms_start is not None:
        trial_df = trial_df[(trial_df['time_ms']>=ms_start)]
    if ms_stop is not None:
        trial_df = trial_df[(trial_df['time_ms']<ms_stop)]

    return trial_df