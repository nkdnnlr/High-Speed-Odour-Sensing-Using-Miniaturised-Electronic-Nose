import numpy as np
import scipy.io
import pandas as pd
from sklearn.metrics._classification import accuracy_score, f1_score

def split_traintest_time(selected, f_test=0.1, time_split=True, stratified=False):
    """
    Splits the selected data into train and test sets. Time-batched or not. Stratified or not.

    :param _type_ selected: ATTENTION: assumes selected with pre-reset_index
    :param float f_test: _description_, defaults to 0.1
    :param bool time_split: _description_, defaults to True
    :param bool stratified: _description_, defaults to False
    """
    # Reset index of (underlying) dataframe
    selected = selected.reset_index()
    
    # Define splits
    f_trainval = 1-f_test
    n_all = len(selected)
    n_test = int(f_test*n_all)
    idx_trainval = int(f_trainval*n_all)

    if time_split:
        if not stratified:
            # Not stratified: just check index
            selected_gas = selected.copy(deep=True).query(f"gas1 != 'None'").reset_index() # resetting index
            selected_train = selected_gas.query(f"level_0 < {idx_trainval}")
            selected_test = selected_gas.query(f"level_0 >= {idx_trainval}")
        else:
            # Stratified: check index & make sure that each class has same number of elements
            gases = selected["gas1"]
            gases_unique, gases_counts = np.unique(gases, return_counts=True)

            selected_train = selected[0:0]
            selected_test = selected[0:0]
            for gas, count in zip(gases_unique, gases_counts):
                selected_gas = selected.copy(deep=True).query(f"gas1 == '{gas}'").reset_index() #  resetting index 
                selected_train = pd.concat([selected_train, selected_gas[selected_gas.index < (count*f_trainval)]])
                selected_test = pd.concat([selected_test, selected_gas[selected_gas.index >= (count*f_trainval)]])
    else:
        if not stratified:
            # Not stratified: just check index
            selected_gas = selected.copy(deep=True).query(f"gas1 != 'None'").sample(frac=1).reset_index() # Shuffling & resetting index
            selected_train = selected_gas.query(f"level_0 < {idx_trainval}")
            selected_test = selected_gas.query(f"level_0 >= {idx_trainval}")
        else:
            # Stratified: check index & make sure that each class has same number of elements
            gases = selected['gas1'].to_numpy()
            gases_unique, gases_counts = np.unique(gases, return_counts=True)

            selected_train = selected[0:0]
            selected_test = selected[0:0]
            for gas, count in zip(gases_unique, gases_counts):
                selected_gas = selected.copy(deep=True).query(f"gas1 == '{gas}'").sample(frac=1).reset_index() # Shuffling & resetting index
                selected_train = pd.concat([selected_train, selected_gas[selected_gas.index < (count*f_trainval)]])
                selected_test = pd.concat([selected_test, selected_gas[selected_gas.index >= (count*f_trainval)]])
    return selected_train, selected_test

def get_kfold_indices(selected, nfolds=5, stratified=True):
    """
    Gets kfold indices for time series (time-batched). Stratified (not strictly time-batched) or not.

    :param _type_ selected: ATTENTION: assumes selected with pre-reset_index
    :param int nfolds: _description_, defaults to 5
    :param bool stratified: _description_, defaults to False
    """
    kfold_indices_train, kfold_indices_val = {}, {}
    if not stratified:
        # Get number of validation samples
        n_val = int(len(selected)/nfolds)
        # iterate over folds
        for k in range(nfolds):
            # Define start & stop index of val split
            i_start_val = k*n_val
            i_stop_val = (k+1)*n_val
            kfold_indices_train[k], kfold_indices_val[k] = [], []
            # iterate over trials & check if name is in 
            for i, row in enumerate(selected):
                if row.record.name in range(i_start_val, i_stop_val):
                    kfold_indices_val[k].append(i)
                else:
                    kfold_indices_train[k].append(i)
            # Make numpy arrays for convenience         
            kfold_indices_train[k], kfold_indices_val[k] = np.array(kfold_indices_train[k]), np.array(kfold_indices_val[k])
    else:
        gases = selected["gas1"].to_numpy()
        gases_unique, gases_counts = np.unique(gases, return_counts=True)
        # iterate over folds
        for k in range(nfolds):
            kfold_indices_train[k], kfold_indices_val[k] = [], []
            # iterate over gases
            for gas, count in zip(gases_unique, gases_counts):
                n_val = int(count/nfolds)
                # Define gas-specific start & stop index of val split
                i_start_val = k*n_val
                i_stop_val = (k+1)*n_val
                # iterate over trials
                i_gas = 0
                i = 0
                for _, row in selected.iterrows():#enumerate(selected):
                    # print(i)
                    if row.gas1 == gas:
                        if i_gas in range(i_start_val, i_stop_val):
                            kfold_indices_val[k].append(i)
                            # kfold_indices_val[k].append(gas)
                        else:
                            kfold_indices_train[k].append(i)                    
                            # kfold_indices_train[k].append(gas)
                        i_gas += 1
                    i += 1
            # Make numpy arrays for convenience         
            kfold_indices_train[k], kfold_indices_val[k] = np.array(kfold_indices_train[k]), np.array(kfold_indices_val[k])
    return kfold_indices_train, kfold_indices_val

def get_trial_df(trial, data_dir, ms_start=-1000, ms_stop=3000, ftype='csv'):
    """ 
    Returns the trial data as a pandas dataframe.

    :param _type_ trial: data of the trial
    :param _type_ data_dir: directory of the data
    :param int ms_start: start time in ms, defaults to -1000
    :param int ms_stop: stop time in ms, defaults to 3000
    :param str ftype: file type, defaults to 'csv'
    @return _type_ trial_df: trial data as a pandas dataframe   
    """
    # Load data file
    trial_path = data_dir.joinpath(trial['condition'], trial['kind'], trial['trial_id']+f'.{ftype}')
    # Check type of file
    if ftype == 'arrow':
        trial_df = pd.read_feather(trial_path)
    elif ftype == 'csv':
        trial_df = pd.read_csv(trial_path, engine='pyarrow')
    # Crop in time
    trial_df = trial_df[(trial_df['time_ms']>=ms_start) & (trial_df['time_ms']<ms_stop)]
    return trial_df

def get_cycles(chandata, iloc0, cstep_l, cstep_r, cycle_phase=0):
    """ 
    Extracts heater cycles from the trial data.
    Cycles start at step == cycle_phase.

    :param _type_ chandata: channel data
    :param int iloc0: index of the trial start
    :param _type_ cstep_l: left channel steps
    :param _type_ cstep_r: right channel steps
    :param int cycle_phase: cycle phase, defaults to 0
    @return _type_ cycles: heater cycles    
    """
    cycles = []
    cs_l = HeaterCycles.find_cycles(cstep_l, iloc0, cycle_phase)
    cs_r = HeaterCycles.find_cycles(cstep_r, iloc0, cycle_phase)
    if cs_l:
        cycles.append(HeaterCycles(chandata, cs_l, range(1,5)))
        t_start_0th = cs_l[0][cs_l[-1]]     # phase of start-pulse-cycle wrt trial start
    elif cs_r:
        cycles.append(HeaterCycles(chandata, cs_r, range(5,9)))
        t_start_0th = cs_r[0][cs_r[-1]]     # phase of start-pulse-cycle wrt trial start 
    else:
        return None, None
    return cycles, t_start_0th

def flatten_list(l):
    """
    Flattens a list of lists.

    :param _type_ l: list of lists
    @return _type_ [item for sublist in l for item in sublist]: flattened list
    """
    return [item for sublist in l for item in sublist]

def unison_shuffled(lists):
    """
    Shuffles multiple lists in unison.

    :param _type_ lists: lists to shuffle
    @return _type_ shuffled_lists: shuffled lists
    """
    # Check same length
    assert all(len(lst) == len(lists[0]) for lst in lists[1:])
    # Shuffle
    p = np.random.permutation(len(lists[0]))
    shuffled_lists = [lst[p] for lst in lists]
    return shuffled_lists

def score_temporalclassification(gas1_all, predicted_all, phases_all, true_offset):
    """
    Scores the temporal classification.

    :param _type_ gas1_all: true gas
    :param _type_ predicted_all: predicted gas
    :param _type_ phases_all: phases
    :param int true_offset: true offset
    @return _type_ acc: accuracy
    @return _type_ f1: f1 score
    @return _type_ detection_rate: detection rate
    @return _type_ phases_onset: phases onset
    @return _type_ phases_offset: phases offset
    @return _type_ confidence_mean: mean confidence
    """
    gas1_all_array = np.array(gas1_all)
    predicted_all_array = np.array(predicted_all)
    phases_all_array = np.array(phases_all)

    gas1_ensemble, events_detected, pred_ensemble, phases_onset, phases_offset, confidence_all = [], [], [], [], [], []
    for gas1_arr, pred_arr, phase_arr in zip(gas1_all_array, predicted_all_array, phases_all_array):
        if gas1_arr[0] == 'blank':
            continue
        # Check if non-blank odour is detected
        if detect_event(pred_arr):
            # When multiple non-blank predictions, take the most frequent one as the 'predicted gas'
            pred, confidence = predict_ensemble(pred_arr, n=1)

            # Get prediction onset (of any non-blank gas) wrt true gas onset 
            # and prediction offset (of any non-blank gas) wrt true gas offset
            t_onset, t_offset = get_onset_offset(pred_arr, phase_arr, true_offset)

            events_detected.append(1)
            phases_onset.append(t_onset)
            phases_offset.append(t_offset)
            gas1_ensemble.append(gas1_arr[0])
            pred_ensemble.append(pred)
            confidence_all.append(confidence)

        else:
            events_detected.append(0)
            gas1_ensemble.append(gas1_arr[0])
            pred_ensemble.append('blank')
            confidence_all.append(1)
    # Calculate scores
    acc = accuracy_score(gas1_ensemble, pred_ensemble)
    f1 = f1_score(gas1_ensemble, pred_ensemble, average='weighted')
    detection_rate = np.mean(events_detected)
    confidence_mean = np.mean(confidence_all)
    return acc, f1, detection_rate, phases_onset, phases_offset, confidence_mean

def score_temporalclassification_multiclass(gas1_all, gas2_all, predicted_all, phases_all):
    """  
    Scores the temporal classification for multiple classes.

    :param _type_ gas1_all: true gas 1
    :param _type_ gas2_all: true gas 2
    :param _type_ predicted_all: predicted gas
    :param _type_ phases_all: phases
    @return _type_ detection_rate: detection rate
    @return _type_ acc_gas1: accuracy gas 1
    @return _type_ acc_gas2: accuracy gas 2    
    """
    gas1_all_array = np.array(gas1_all)
    gas2_all_array = np.array(gas2_all)
    predicted_all_array = np.array(predicted_all)
    phases_all_array = np.array(phases_all)
    gas1_ensemble, gas2_ensemble, events_detected, pred_ensemble, phases_onset, phases_offset, confidence_all = [], [], [], [], [], [], []
    for gas1_arr, gas2_arr, pred_arr, phase_arr in zip(gas1_all_array, gas2_all_array, predicted_all_array, phases_all_array):
        if gas1_arr[0] == 'blank':
            continue
        if detect_event(pred_arr):
            events_detected.append(1)
            gas1 = gas1_arr[0]
            gas2 = gas2_arr[0]
            pred, confidence = predict_ensemble(pred_arr, n=2)
            gas1_ensemble.append(gas1)
            gas2_ensemble.append(gas2)
            pred_ensemble.append(pred)
            confidence_all.append(confidence)
        else:
            events_detected.append(0)
            gas1_ensemble.append(gas1)
            gas2_ensemble.append(gas2)
            pred_ensemble.append(['blank'])
            confidence_all.append([1])
    tp_gas1, tp_gas2, tn_gas1, tn_gas2 = 0, 0, 0, 0
    for gas1, gas2, pred, confidence in zip(gas1_ensemble, gas2_ensemble, pred_ensemble, confidence_all):
        if gas1 in pred:
            tp_gas1 += 1
        else:
            tn_gas1 += 1
        if gas2 in pred:
            tp_gas2 += 1
        else:
            tn_gas2 += 1         
    detection_rate = np.mean(events_detected)
    acc_gas1 = tp_gas1 / (tp_gas1 + tn_gas1)
    acc_gas2 = tp_gas2 / (tp_gas2 + tn_gas2)
    return detection_rate, acc_gas1, acc_gas2

def detect_event(arr):
    """
    Check if non-blank elements are present in the array. Return false if True. Return True if False.

    :param _type_ arr: array
    @return _type_ False: if non-blank elements are present
    """
    values, counts = np.unique(arr, return_counts=True)
    values, counts = values[values != 'blank'], counts[values != 'blank']
    if len(counts)==0:
        return False
    else:
        return True
    
def predict_ensemble(arr, n=1):
    """
    Predicts the ensemble.

    :param _type_ arr: array
    :param int n: number of predictions, defaults to 1
    @return _type_ values[0], counts[0]/np.sum(counts): prediction, confidence
    """    
    values, counts = np.unique(arr, return_counts=True)
    values, counts = values[values != 'blank'], counts[values != 'blank']
    counts, values = (np.array(t) for t in zip(*sorted(zip(counts, values), reverse=True)))
    if n==1:
        return values[0], counts[0]/np.sum(counts)
    else:
        return values[:n], counts[:n]/np.sum(counts)

def get_onset_offset(pred_arr, phase_arr, true_offset):
    """
    Gets the onset and offset of the prediction.

    :param _type_ pred_arr: predicted array
    :param _type_ phase_arr: phase array
    :param int true_offset: true offset
    @return _type_ phase_onset: phase onset
    @return _type_ phase_offset: phase offset
    """
    phase_delta = phase_arr[1]-phase_arr[0]
    phase_onset = phase_arr[pred_arr != 'blank'][0]+phase_delta
    phase_offset = phase_arr[phase_arr>true_offset][pred_arr[phase_arr>true_offset] == 'blank'][0]+phase_delta
    return phase_onset, phase_offset

def vote_winner(predicted_feature):
    """
    Votes for the winner.

    :param _type_ predicted_feature: predicted feature
    @return _type_ np.array([result.to_numpy()]): winner
    """
    # Create a list of dataframes
    dfs = [pd.DataFrame(predicted) for predicted in predicted_feature]
    # Concatenate the dataframes and apply the 'mode' method along the rows
    result = pd.concat(dfs).groupby(level=0).apply(lambda x: x.mode().iloc[0])
    # Convert back to numpy and return
    return np.array([result.to_numpy()])

class HeaterCycles:
    """ A collection of heater cycles. """
    @classmethod
    def find_cycles(cls, steps, iloc0, cycle_start, min_cycles=3):
        """ 
        Finds heater cycles in the steps array.

        :param _type_ steps: steps array
        :param int iloc0: index of the trial start
        :param int cycle_start: cycle start
        :param int min_cycles: minimum number of cycles, defaults to 3
        @return _type_ None: if no cycles found
        """
       # Find cycle starts:
        start_ilocs = np.flatnonzero(steps == cycle_start)
        # Are there any cycles?
        if len(start_ilocs) < 2: return None
        # Find the cycle periods:
        periods = start_ilocs[1:] - start_ilocs[:-1]
        # Drop the last cycle, it might be incomplete:
        start_ilocs = start_ilocs[:-1]
        # Drop trivial cycles of length 1:
        valid = periods > 1
        if np.count_nonzero(valid) < min_cycles: return None
        start_ilocs = start_ilocs[valid]
        periods = periods[valid]
        # Are the periods regular?
        if len(np.unique(periods)) == 1:
            period = periods[0]
        else:
            period = None # irregular periods
        end_ilocs = start_ilocs + periods
        # Locate cycle 0 (the one containing the trial's start time):
        c0_num = end_ilocs.searchsorted(iloc0)
        return (start_ilocs, end_ilocs, period, c0_num)

    def __init__(self, chandata, cs, chans):
        self.chandata = chandata
        self.start_ilocs, self.end_ilocs, self.period, self.c0_num = cs
        self.chans = chans
    
    def __repr__(self):
        p = self.period or "<irregular>"
        r = object.__repr__(self)
        r += ": %d cycles of length %s on gas channels %s" % (len(self), p, list(self.chans))
        return r
    
    def __len__(self):
        return len(self.start_ilocs)

    def ilocs(self, c_num):
        """ Gets the data indices (ilocs) for cycle #c_num.
            0 refers to the cycle that contains the trial start time.
            Negative numbers are cycles before c0_num, not cycles from the trial end!
        """
        c = self.c0_num + c_num
        if c < 0: raise IndexError("Cycle %d out of bounds" % c_num)
        return range(self.start_ilocs[c], self.end_ilocs[c])
    
    def __getitem__(self, cycles):
        if isinstance(cycles, slice):
            if cycles.start is not None:
                cfirst_num = cycles.start
            else:
                cfirst_num = -self.c0_num
            
            if cycles.stop is not None:
                clast_num = cycles.stop - 1
            else:
                clast_num = len(self) - 1 - self.c0_num
        else:
            cfirst_num = clast_num = int(cycles)

        for c in (cfirst_num, clast_num):
            if self.c0_num + c < 0: raise IndexError("Cycle %d out of bounds" % c)

        cfirst_iloc = self.start_ilocs[self.c0_num + cfirst_num]
        clast_iloc = self.end_ilocs[self.c0_num + clast_num]
        return CycleRange(self, cfirst_num, clast_num, slice(cfirst_iloc, clast_iloc))

class CycleRange:
    """ A range of cycles. """
    def __init__(self, cycles, cfirst_num, clast_num, iloc_slice):
        self.cycles = cycles
        self.cfirst_num = cfirst_num
        self.clast_num = clast_num
        self.iloc_slice = iloc_slice
    
    def __len__(self):
        n = self.clast_num - self.cfirst_num + 1
        return n

    def cycle_nums(self):
        return np.arange(self.cfirst_num, self.clast_num+1)
        
    def __getitem__(self, col):
        data = self.cycles.chandata[col][self.iloc_slice].to_numpy()
        return data.reshape((len(self), self.cycles.period))
    