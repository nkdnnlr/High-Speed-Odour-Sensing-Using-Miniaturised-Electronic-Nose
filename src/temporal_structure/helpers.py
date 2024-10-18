import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, fftshift
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from scipy.signal import find_peaks
from math import pi

def cut(t, nose_data, t_start=0, duration=1000):
    """ 
    Cut the data in time

    :param t: time vector
    :param nose_data: data
    :param t_start: start time
    :param duration: duration
    :return: cut data    
    """
    t_duration = t[(t>=t_start) & (t<=duration)]
    nose_data_duration = nose_data[:, (t>=t_start) & (t<=duration)]
    return t_duration, nose_data_duration

def scale(nose_data_duration, modality):
    """
    Scale the data

    :param nose_data_duration: data
    :param modality: modality
    :return: scaled data
    """
    if modality == 'R_gas':
        nose_data_scaled = [np.diff(np.log(nose_data_duration[chn])) for chn in range(4)]
    elif modality == 'T_heat':
        nose_data_scaled = [np.diff(nose_data_duration[chn]) for chn in range(4)]
    elif modality == 'pid':
        nose_data_scaled = nose_data_duration
    return np.array(nose_data_scaled)

def get_trial_df(trial, data_dir, ms_start=-1000, ms_stop=3000, ftype='csv'): 
    """ 
    Get trial dataframe

    :param trial: trial info
    :param data_dir: directory with data
    :param ms_start: start time
    :param ms_stop: stop time
    :param ftype: file type
    :return: trial dataframe    
    """
    # Load data file
    trial_path = data_dir.joinpath(trial['condition'], trial['kind'], trial['trial_id']+f'.{ftype}')
    # Check type of file
    if ftype == 'arrow':
        trial_df = pd.read_feather(trial_path)
    elif ftype == 'csv':
        trial_df = pd.read_csv(trial_path, engine='pyarrow')
    # Crop in time
    if ms_start is not None:
        trial_df = trial_df[(trial_df['time_ms']>=ms_start) & (trial_df['time_ms']<ms_stop)]
    return trial_df

def frequency_filter(x, y, f_min=0, f_max=100):
    """ 
    Filter for reasonable frequencies

    :param x: x values
    :param y: y values
    :param f_min: minimum frequency
    :param f_max: maximum frequency
    :return: filtered x and y values    
    """
    y_filtered = y[(x>=f_min) & (x<=f_max)]
    x_filtered = x[(x>=f_min) & (x<=f_max)]
    return x_filtered, y_filtered

def ffs_to_peaks(ffs):
    """ 
    Get peaks from Fourier features

    :param ffs: Fourier features
    :return: peaks
    """
    fhz = fftfreq(len(ffs), 1e-3)
    N = len(ffs)        
    # Shift
    y = fftshift(ffs)
    x = fftshift(fhz)
    # Filter for reasonable frequencies
    x, y = frequency_filter(x, y, f_min=0, f_max=100)
    # Round to integers
    x = np.round(x, 0)
    # Find peaks
    peaks, _ = find_peaks(np.abs(y), height=0)
    # Sort peak heights
    peaks_sorted = peaks[np.argsort(_['peak_heights'], )]
    # Get peak frequencies, magnitudes, phases
    _freqs_peaks = x[peaks_sorted]
    _magnitudes_peaks = np.abs(y)[peaks_sorted]
    _phases_peaks = np.angle(y)[peaks_sorted]
    # Dischard neg freq values and reverse order
    freqs_peaks = _freqs_peaks[_freqs_peaks>0][::-1]
    magnitudes_peaks = _magnitudes_peaks[_freqs_peaks>0][::-1]
    phases_peaks = _phases_peaks[_freqs_peaks>0][::-1]
    return freqs_peaks, magnitudes_peaks, phases_peaks, x, y

def ffs_to_peaks_allsensors(ffs_all):
    """
    Get peaks from Fourier features for all sensors

    :param ffs_all: Fourier features for all sensors
    :return: peaks for all sensors
    """
    freqs_peaks_all, magnitudes_peaks_all, phases_peaks_all, x_all, y_all = [], [], [], [], []
    # Calculate peaks for each sensor from Fourier feature
    for s, ffs in enumerate(ffs_all):
        # Calculate peaks and sort them
        freqs_peaks, magnitudes_peaks, phases_peaks, x, y = ffs_to_peaks(ffs)
        # Append strongest peak feature to lists
        freqs_peaks_all.append(freqs_peaks)
        magnitudes_peaks_all.append(magnitudes_peaks)
        phases_peaks_all.append(phases_peaks)
        x_all.append(x)
        y_all.append(y)
    return np.array(freqs_peaks_all), np.array(magnitudes_peaks_all), np.array(phases_peaks_all), np.array(x_all), np.array(y_all)    

def trial_nose(trial_df, modality, t_start, t_end, channels=range(4)):
    """
    Get data from trial

    :param trial_df: trial dataframe
    :param modality: modality
    :param t_start: start time
    :param t_end: end time
    :param channels: channels
    :return: data and time    
    """
    trial_df = trial_df[(trial_df['time_ms']>=t_start) & (trial_df['time_ms']<t_end)]
    t = trial_df['time_ms'].values
    nose_data = []
    for chan in channels:
        if modality == 'R_gas':
            R_gas = trial_df[f'{modality}_{chan+1}']
        elif modality == 'T_heat':
            R_gas = trial_df[f'{modality}_{chan+1}']
        else:
            print(f"Channel '{modality}' not implemented.")
        nose_data.append(R_gas)
    
    return np.array(nose_data), t

def fft_nose(nose_data):
    """
    Get Fourier features from data

    :param nose_data: data
    :return: Fourier features
    """
    nose_data_fft = np.array([fft(nose_data[chn]) for chn in range(len(nose_data))])
    return nose_data_fft

def get_fft_features(ffs_trial, f_max=None):
    """ 
    Get features from Fourier features

    :param ffs_trial: Fourier features
    :param f_max: maximum frequency
    :return: features    
    """
    fhz = fftfreq(ffs_trial.shape[1], 1e-3)
    magnitudes_trial = []
    phases_trial = []
    f = None
    # Iterate across sensors and extract magnitude and phase
    for s, ft in enumerate(ffs_trial):
        y = fftshift(ft)
        f = fftshift(fhz)    
        # Handle cutoff frequency if specified
        if f_max is not None:
            f_min = 0
            f, y = np.array(f), np.array(y)
            indices = (f>=f_min) & (f<=f_max)
            f = f[indices]
            y = y[indices]
            magnitudes_trial.append(np.abs(y))
            phases_trial.append(np.angle(y))
        else:
            magnitudes_trial.append(np.abs(y))
            phases_trial.append(np.angle(y))
    # Make numpy arrays and return
    return np.array(f), np.array(magnitudes_trial), np.array(phases_trial)

def get_trial_features(f, magnitudes, phases):
    """
    Get features from trial

    :param f: frequencies
    :param magnitudes: magnitudes
    :param phases: phases
    :return: features    
    """
    freqs_peaks_highest, magnitudes_peaks_highest, phases_peaks_highest = [], [], []
    # Iterate over sensor dimension
    for magnitudes_sensor, phases_sensor in zip(magnitudes, phases):
        # Find peaks
        peaks, properties = find_peaks(magnitudes_sensor, height=0)
        # Sort peak heights
        peaks_sorted = peaks[np.argsort(properties['peak_heights'], )]
        # Get peak frequencies, magnitudes, phases
        _freqs_peaks = f[peaks_sorted]
        _magnitudes_peaks = magnitudes_sensor[peaks_sorted]
        _phases_peaks = phases_sensor[peaks_sorted]
        # Dischard neg freq values and reverse order
        freqs_peaks = _freqs_peaks[_freqs_peaks>0][::-1]
        magnitudes_peaks = _magnitudes_peaks[_freqs_peaks>0][::-1]
        phases_peaks = _phases_peaks[_freqs_peaks>0][::-1]
        # Append
        freqs_peaks_highest.append(freqs_peaks[0]) 
        magnitudes_peaks_highest.append(magnitudes_peaks[0])
        phases_peaks_highest.append(phases_peaks[0])
    return np.array(freqs_peaks_highest).flatten(), np.array(magnitudes_peaks_highest).flatten(), np.array(phases_peaks_highest).flatten()


def get_freqspace_features(trials, data_dir, modality, t_start, t_end, stimulus_start, stimulus_duration, set='training'):
    """ 
    Get features from frequency space

    :param trials: dataframe with trials
    :param data_dir: directory with data
    :param modality: modality to extract features from (e.g. 'R_gas')
    :param t_start: start time of trial
    :param t_end: end time of trial
    :param stimulus_start: start of stimulus
    :param stimulus_duration: duration of stimulus
    :param set: 'training' or 'validation'    
    """
    y_freqs, y_patterns, X_freqs, X_magnitudes, X_phases, X_allfeatures = [], [], [], [], [], []
    for idx, trial in trials.iterrows():
        # Get trial info
        pattern, freq = trial['kind'], int(trial['shape'][:-2])
        # Get dataframe
        trial_df = get_trial_df(trial, data_dir, t_start, t_end)
        # Extract data
        nose_data, t = trial_nose(trial_df, modality, t_start, t_end)
        # Cut & Scale
        t_duration, nose_data_duration = cut(t, nose_data, stimulus_start, stimulus_duration)
        nose_data_scaled = scale(nose_data_duration, modality)
        # FFT                    
        nose_data_fft = fft_nose(nose_data_scaled)
        # Get FFT features
        f, magnitudes_trial, phases_trial = get_fft_features(nose_data_fft)
        # Get trial features
        freqs_peaks, magnitudes_peaks, phases_peaks = get_trial_features(f, magnitudes_trial, phases_trial)
        # Append to lists
        y_freqs.append(freq)
        y_patterns.append(pattern)
        X_freqs.append(freqs_peaks)
        X_magnitudes.append(magnitudes_peaks)
        X_phases.append(phases_peaks)
        X_allfeatures.append(np.concatenate((freqs_peaks, magnitudes_peaks, phases_peaks)).flatten())
    return y_freqs, y_patterns, X_freqs, X_magnitudes, X_phases, X_allfeatures


def get_pid_data(trial, data_dir, t_start, t_end):
    """ 
    Get PID data

    :param trial: trial info
    :param data_dir: directory with data
    :param t_start: start time
    :param t_end: end time
    :return: PID data    
    """
    # Get dataframe
    trial_df = get_trial_df(trial, data_dir, t_start, t_end)    
    # Extract data
    pid_data, t = np.expand_dims(trial_df['pid_V'], axis=0), trial_df['time_ms']
    return pid_data, t

def extract_pids(codes_selected, tmin, tmax, pid_index, data_dir_pid, i=None):
    """
    Extract PID data

    :param codes_selected: selected codes
    :param tmin: start time
    :param tmax: end time
    :param pid_index: PID index
    :param data_dir_pid: directory with PID data
    :param i: index
    :return: PID data and time
    """
    pids_data = []
    pids_t = []
    for code in codes_selected:
        kind, gas1, gas2, freq = code.split('_')
        if kind == "acor":
            kind = "acorr"
        sd = pid_index[((pid_index['gas1']==gas1) & (pid_index['gas2']==gas2) & (pid_index['shape']==str(freq)+"Hz")) & (pid_index['kind']==kind)]
        if i is None:
            pids_data.append([get_pid_data(sd.iloc[j], data_dir_pid, t_start=tmin, t_end=tmax)[0][0] for j in range(len(sd))])
            pids_t.append([get_pid_data(sd.iloc[j], data_dir_pid, t_start=tmin, t_end=tmax)[1] for j in range(len(sd))])
        else:
            pids_data.append(get_pid_data(sd.iloc[i], data_dir_pid, t_start=tmin, t_end=tmax)[0][0])
            pids_t.append(get_pid_data(sd.iloc[i], data_dir_pid, t_start=tmin, t_end=tmax)[1])
    return pids_data, pids_t

def get_freqspace_features_pid(trials, data_dir, modality, t_start, t_end, stimulus_start, stimulus_duration, set='training'):
    """
    Get features from frequency space

    :param trials: dataframe with trials
    :param data_dir: directory with data
    :param modality: modality to extract features from (e.g. 'R_gas')
    :param t_start: start time of trial
    :param t_end: end time of trial
    :param stimulus_start: start of stimulus
    :param stimulus_duration: duration of stimulus
    :param set: 'training' or 'validation'
    """
    y_freqs, y_patterns, X_freqs, X_magnitudes, X_phases, X_allfeatures = [], [], [], [], [], []
    for idx, trial in trials.iterrows():
        # Get trial info
        pattern, freq = trial['kind'], int(trial['shape'][:-2])
        # Get dataframe
        trial['condition'] = ''
        trial_df = get_trial_df(trial, data_dir, t_start, t_end, ftype='csv')
        # Extract data
        pid_data, t = np.expand_dims(trial_df['pid_V'], axis=0), trial_df['time_ms']
        # Normalise
        pid_data = (pid_data - trial['pid_min'])/(trial['pid_max'] - trial['pid_min'])
        # Cut & Scale
        t_duration, pid_data_duration = cut(t, pid_data, stimulus_start, stimulus_duration)
        pid_data_duration = scale(pid_data_duration, modality)
        # FFT                    
        nose_data_fft = fft_nose(pid_data_duration)
        # Get FFT features
        f, magnitudes_trial, phases_trial = get_fft_features(nose_data_fft)
        # Get trial features
        freqs_peaks, magnitudes_peaks, phases_peaks = get_trial_features(f, magnitudes_trial, phases_trial)
        # Append to lists
        y_freqs.append(freq)
        y_patterns.append(pattern)
        X_freqs.append(freqs_peaks)
        X_magnitudes.append(magnitudes_peaks)
        X_phases.append(phases_peaks)
        X_allfeatures.append(np.concatenate((freqs_peaks, magnitudes_peaks, phases_peaks)).flatten())
    return y_freqs, y_patterns, X_freqs, X_magnitudes, X_phases, X_allfeatures

def get_peak_features(codes_selected, ffs_selected):
    """
    Get peak features

    :param codes_selected: selected codes
    :param ffs_selected: selected Fourier features
    :return: peak features
    """
    peak_features = {}
    for i, (ffs_all, code_selected) in enumerate(zip(ffs_selected, codes_selected)):
        peak_features[code_selected] = {'frequency': [], 'magnitude': [], 'phase': []}
        for ffs in ffs_all:
            peak_features[code_selected]['frequency'].append([])
            peak_features[code_selected]['magnitude'].append([])
            peak_features[code_selected]['phase'].append([])
            fhz = fftfreq(len(ffs[0]), 1e-3)
            N = len(ffs[0])  
            freqs_peaks_all, magnitudes_peaks_all, phases_peaks_all, x_all, y_all = ffs_to_peaks_allsensors(ffs)   
            freqs_peak_dominant = [freq[0] for freq in freqs_peaks_all]
            magnitudes_peak_dominant = [magnitude[0] for magnitude in magnitudes_peaks_all]
            phases_peak_dominant = [phase[0] for phase in phases_peaks_all]
            for peak_freq, peak_magnitude, peak_phase in zip(freqs_peak_dominant, magnitudes_peak_dominant, phases_peak_dominant):
                peak_features[code_selected]['frequency'][-1].append(1/peak_freq)
                peak_features[code_selected]['magnitude'][-1].append(peak_magnitude)
                peak_features[code_selected]['phase'][-1].append(peak_phase)
    return peak_features

def get_scores(y_dict, pred_dict, metric='accuracy'):
    """
    Get scores

    :param y_dict: dictionary with true labels
    :param pred_dict: dictionary with predicted labels
    :param metric: metric to calculate scores
    :return: scores
    """
    scores = []
    for f, y in y_dict.items():
        y_pred = np.array(pred_dict[f]).flatten()
        if metric == 'accuracy':
            score = accuracy_score(y, y_pred)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y, y_pred)
        elif metric == 'f1_score':
            score = f1_score(y, y_pred, average="weighted")
        else:
            print(f"'{metric}' not implemented.")
        scores.append(score)
    return list(y_dict.keys()), scores

def get_winner(model_ensemble, axis=0):
    """
    Get winner

    :param model_ensemble: ensemble of models
    :param axis: axis
    :return: winner
    """
    if axis==0:
        preds, counts = np.unique(model_ensemble, return_counts=True)
        # argsort with random tie breaking
        randomnumbers = np.random.random(counts.size)
        ind = np.lexsort((randomnumbers, counts))
        return preds[ind][-1]
    elif axis==1:
        winner_arr = []
        for pred_arr in np.array(model_ensemble).T:
            preds, counts = np.unique(pred_arr, return_counts=True)
            # argsort with random tie breaking
            randomnumbers = np.random.random(counts.size)
            ind = np.lexsort((randomnumbers, counts))
            winner_arr.append(preds[ind][-1])
        return winner_arr
    else:
        return NotImplementedError