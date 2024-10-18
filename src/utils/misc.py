import pandas as pd
import numpy as np
import scipy 
import scipy.io

def get_trial_df(trial, data_dir, ms_start=None, ms_stop=None):
    """ 
    Returns the trial data as a pandas dataframe.

    :param trial: Dictionary with the trial information
    :param data_dir: Path to the data directory
    :param ms_start: Start time in ms
    :param ms_stop: Stop time in ms    
    """
    # Load data file
    trial_path = data_dir.joinpath(trial['condition'], trial['kind'], trial['trial_id']+f'.csv')
    trial_df = pd.read_csv(trial_path, engine='pyarrow')
    # Crop in time
    if ms_start is not None:
        trial_df = trial_df[(trial_df['time_ms']>=ms_start)]
    if ms_stop is not None:
        trial_df = trial_df[(trial_df['time_ms']<ms_stop)]
    return trial_df

# Define a custom encoder function to preserve data types of keys and values in nested dictionaries
def custom_encoder(obj):
    """ 
    Define a custom encoder function to preserve data types of keys and values in nested dictionaries

    :param obj: Object to encode
    :return obj: Encoded object
    """
    if isinstance(obj, dict):
        encoded_dict = {}
        for key, value in obj.items():
            if isinstance(key, (int, float)):
                key = str(key)
            if isinstance(value, (int, float)):
                encoded_dict[key] = value  # Preserve the data type of the value
            elif isinstance(value, dict):
                encoded_dict[key] = custom_encoder(value)  # Recursively encode nested dictionaries
            else:
                encoded_dict[key] = value
        return encoded_dict
    return obj

# Define a custom decoder function to convert str keys back to their original data types in nested dictionaries
def custom_decoder(obj):
    """ 
    Define a custom decoder function to convert str keys back to their original data types in nested dictionaries

    :param obj: Object to decode
    :return obj: Decoded object    
    """
    if isinstance(obj, dict):
        decoded_dict = {}
        for key, value in obj.items():
            original_key = key
            try:
                key = np.int32(key)
            except ValueError:
                try:
                    key = np.float64(key)
                except ValueError:
                    key = str(key)
                    pass
            if isinstance(value, dict):
                decoded_dict[key] = custom_decoder(value)  # Recursively decode nested dictionaries
            else:
                decoded_dict[key] = value
        return decoded_dict
    return obj

def moving_average(data, window_size):
    """ 
    Calculate the moving average of a signal.

    :param data: Signal to process
    :param window_size: Window size for the moving average
    :return: Moving average of the signal
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def subsample(signal, ss_factor=10):
    """
    Subsample a signal by a given factor.

    :param signal: Signal to subsample
    :param ss_factor: Subsampling factor
    :return: Subsampled signal
    """
    return signal[::ss_factor]

def get_pid_plumes(plume_id, datadir_plumes, set_zero=0):
    """
    Load the PID plume data.

    :param plume_id: ID of the plume
    :param datadir_plumes: Path to the plumes directory
    :return pid_gas1: PID data for gas 1
    """
    # Extract components from plume_id
    codes = plume_id.split(sep='_')
    mix = codes[0]=='mix'
    inverse = codes[-1]=='i'
    id = codes[1]
    # Assign directory
    if mix:
        datadir_plumes_AT = datadir_plumes.joinpath("AT_mix")
        datadir_plumes_EB = datadir_plumes.joinpath("EB_mix")
    else:
        datadir_plumes_AT = datadir_plumes.joinpath("AT_50")
        datadir_plumes_EB = datadir_plumes.joinpath("EB_50")
    # Load PID plume data
    if not inverse:
        pid_gas1 = scipy.io.loadmat(datadir_plumes_EB.joinpath(f"plume_{id}.mat"))['plume'][0]
        pid_gas2 = scipy.io.loadmat(datadir_plumes_AT.joinpath(f"plume_{id}.mat"))['plume'][0]
    else:
        pid_gas2 = scipy.io.loadmat(datadir_plumes_EB.joinpath(f"plume_{id}.mat"))['plume'][0]
        pid_gas1 = scipy.io.loadmat(datadir_plumes_AT.joinpath(f"plume_{id}.mat"))['plume'][0]    
    pid_gas1[pid_gas1<=0] = 0
    pid_gas2[pid_gas2<=0] = 0
    return pid_gas1, pid_gas2

def get_stimulus(row_index, df_data, t_ms_start = 0, data_dir_originalplumes=None):
    """
    Generate the stimulus for a given trial.

    :param row_index: Row index of the trial
    :param df_data: DataFrame with the trial data
    :param t_ms_start: Start time in ms
    :param data_dir_originalplumes: Path to the original plumes directory
    :return df_data: DataFrame with the trial data and the stimulus
    """
    trial_id, kind, shape, concentration, gas1, gas2 = row_index['trial_id'], row_index['kind'], row_index['shape'], row_index['concentration'], row_index['gas1'], row_index['gas2']
    dur_ms = 1000 # if not otherwise specified
    fs = 1000  # Sampling frequency
    df_data['EB'], df_data['IA'], df_data['Eu'], df_data['2H'], df_data['b1'], df_data['b2'], df_data['b_comp'] = 0, 0, 0, 0, 0, 0, 1
    if kind == 'plume':
        dur_ms = 5000
        pid_gas1, pid_gas2 = get_pid_plumes(shape, data_dir_originalplumes)
        pid_gas1_normalised = (pid_gas1 - np.min(pid_gas1)) / (np.max(pid_gas1) - np.min(pid_gas1))
        pid_gas2_normalised = (pid_gas2 - np.min(pid_gas2)) / (np.max(pid_gas2) - np.min(pid_gas2))
        ss = len(pid_gas1) // dur_ms
        stimulus_1 = subsample(moving_average(pid_gas1_normalised, window_size=ss), ss)
        stimulus_2 = subsample(moving_average(pid_gas2_normalised, window_size=ss), ss)
        df_data[gas1][(df_data['time_ms']>=t_ms_start) & (df_data['time_ms']<t_ms_start+dur_ms)] = 0.5*stimulus_1
        df_data[gas2][(df_data['time_ms']>=t_ms_start) & (df_data['time_ms']<t_ms_start+dur_ms)] = 0.5*stimulus_2
        # Flow compensation
        df_data['b_comp'] = 1 - df_data[gas1] - df_data[gas2]
    if kind == 'pulse':
        dur_ms = int(float(shape[:-1])*1000)
        df_data[gas1][(df_data['time_ms']>=t_ms_start) & (df_data['time_ms']<=t_ms_start+dur_ms)] = 0.5 * (concentration/5.)
        # Flow compensation
        df_data['b_comp'] = 1 - df_data[gas1]
    if kind == 'corr':
        freq = int(shape[:-2])
        period = fs / freq# * 2
        # Create a time array from 0 to 1000 ms with a 1 kHz resolution
        time = np.arange(0, dur_ms, 1)  # Time in milliseconds
        stimulus = np.where((time % period) < (period / 2), 1, 0)
        df_data[gas1][(df_data['time_ms']>=t_ms_start) & (df_data['time_ms']<t_ms_start+dur_ms)] = 0.5*stimulus
        df_data[gas2][(df_data['time_ms']>=t_ms_start) & (df_data['time_ms']<t_ms_start+dur_ms)] = 0.5*stimulus
        # Flow compensation
        df_data['b_comp'] = 1 - df_data[gas1] - df_data[gas2]

    if kind == 'acorr':
        freq = int(shape[:-2])
        period = fs / freq# * 2
        # Create a time array from 0 to 1000 ms with a 1 kHz resolution
        time = np.arange(0, dur_ms, 1)  # Time in milliseconds
        stimulus = np.where((time % period) < (period / 2), 1, 0)
        df_data[gas1][(df_data['time_ms']>=t_ms_start) & (df_data['time_ms']<t_ms_start+dur_ms)] = 0.5*stimulus
        df_data[gas2][(df_data['time_ms']>=t_ms_start) & (df_data['time_ms']<t_ms_start+dur_ms)] = 0.5*(1-stimulus)
        # Flow compensation
        df_data['b_comp'] = 1 - df_data[gas1] - df_data[gas2]
    df_data['total'] = df_data['EB'] + df_data['IA'] + df_data['Eu'] + df_data['2H'] + df_data['b1'] + df_data['b2'] + df_data['b_comp']
    return df_data