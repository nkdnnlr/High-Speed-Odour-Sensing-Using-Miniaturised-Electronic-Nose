from functools import partial
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd
from sklearn import svm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

import src.pulse_classification.helpers as utils

def norm01(x, endpoints=True, midpoint=True):
    """
    Normalize data to 0-1.

    :param _type_ x: data
    :param bool endpoints: endpoints, defaults to True
    :param bool midpoint: midpoint, defaults to True
    :return _type_: normalized data
    """
    x = x.copy()
    # Subtraction by offset (line between endpoints)
    if endpoints:
        for n in range(x.shape[0]):
            x[n, :] -= np.linspace(x[n, 0], x[n, -1], x.shape[1])
    # Division by midpoint
    if midpoint:
        imid = x.shape[1] // 2
        x /= x[:, imid : imid + 1]
    return x


class TrialData:
    def __init__(self, trial, sensors, batch, params) -> None:

        self.trial = trial
        self.sensors = sensors
        self.batch = batch
        self.params = params
        self.ms_before = -1*params['ms_start'] #2000 

        self.period = params["period"]
        self.ncycles = params["ncycles"]
        self.fnorm = partial(
            norm01, endpoints=params["endpoints"], midpoint=params["midpoint"]
        )

        self.features_length = {
            "heat": self.period,
            "dc_hot": 1,
            "constant_raw": self.period,
            "constant_subtracted": self.period,
            "cycle_raw": self.period,
            "cycle_signature": self.period,
            "constant_cycle_signature": self.period,
        }

    def get_trial_nose(self, data_dir, ms_start=-1000, ms_stop=3000):
        """ 
        Load the trial data for the nose sensors

        :param _type_ data_dir: data directory
        :param int ms_start: start time, defaults to -1000
        :param int ms_stop: stop time, defaults to 3000
        """
        self.trial_df = utils.get_trial_df(self.trial, data_dir, ms_start=ms_start, ms_stop=ms_stop)

    def _load_colum(self, col, **kwargs):
        """  
        Load the trial data for a given column

        :param str col: column
        :return _type_: trial data
        """
        cdata = self.trial._load_column(self.channel, col, **kwargs) # load the entire column
        tdata = cdata[self.istart:self.istop] # restrict to the trial start/stop times

        # Some columns need tweaks because they have a special datatype:
        if col == "control_cycle_name":
            # reinterpret the U32 data as a 4-char code:
            tdata = np.frombuffer(tdata, dtype="S4")
        
        elif col == "control_cycle_step":
            cname = self["control_cycle_name"]
            if cname[0] == b'dual':
                # reinterpret the 32-bit step as two 16-bit steps (left/right)
                tdata = np.frombuffer(tdata,
                            dtype=np.dtype([("left", "<u2"),("right", "<u2")]))
        return tdata
    

    def get_cycles(self, phase=25):
        """
        Get cycles

        :param int phase: phase, defaults to 25
        """
        cstep_l = self.trial_df["control_cycle_step_left"]
        cstep_r = self.trial_df["control_cycle_step_right"]

        cycles, t_start_0th = utils.get_cycles(chandata=self.trial_df, iloc0=self.ms_before, cstep_l=cstep_l, cstep_r=cstep_r, cycle_phase=phase)
        self.phase_0th = t_start_0th + self.params["ms_start"]

        self.c_before = cycles[self.batch - 1][-2]
        self.c_after = cycles[self.batch - 1][0 : self.ncycles]  
        
    def get_gas1(self):
        return self.trial["gas1"]

    def get_gas2(self):
        return self.trial["gas2"]

    def get_feature(self, feature="cycle_raw", kind="gas"):
        """
        Get feature

        :param str feature: feature, defaults to "cycle_raw"
        :param str kind: kind, defaults to "gas"
        :return _type_: feature
        """
        data_array = np.ma.zeros(
            (len(self.sensors), self.ncycles, self.features_length[feature])
        )

        if feature == "heat":
            kind = "heat"

        # Check if gas or heat
        if kind == "gas":
            _col = "R_gas_"  #%d" % s
        elif kind == "heat":
            _col = "T_heat_"  #%d" % s
        else:
            print(f"'{kind}' does not exist!")

        # Check feature
        for si, s in enumerate(self.sensors):
            col = _col + str(s)

            if feature == "heat":
                data_array[si, :] = self.c_after[col]
            elif feature == "dc_hot":
                data_array[si, :] = self.cycle_dc(
                    col, self.c_before, self.c_after, phase=1, subtract_prev=False
                ).reshape([-1, 1])

            elif feature == "constant_raw":
                data_array[si, :] = self.cycle_features(
                    col,
                    self.c_before,
                    self.c_after,
                    logscale=False,
                    normalize=False,
                    subtract_prev=False,
                )
            elif feature == "constant_subtracted":
                data_array[si, :] = self.cycle_features(
                    col,
                    self.c_before,
                    self.c_after,
                    logscale=False,
                    normalize=False,
                    subtract_prev=True,
                )                            
            elif feature == "cycle_raw":
                data_array[si, :] = self.cycle_features(
                    col,
                    self.c_before,
                    self.c_after,
                    logscale=False,
                    normalize=False,
                    subtract_prev=False,
                )
            elif feature == "cycle_signature":
                data_array[si, :] = self.cycle_features(
                    col,
                    self.c_before,
                    self.c_after,
                    logscale=True,
                    normalize=True, # Check
                    subtract_prev=True,
                )
            elif feature == "constant_cycle_signature":
                if si in [4,5,6,7]:
                    data_array[si, :] = self.cycle_features(
                        col,
                        self.c_before,
                        self.c_after,
                        logscale=True,
                        normalize=True, # Check
                        subtract_prev=True,
                    )
                elif si in [0,1,2,3]:
                    data_array[si, :] = self.cycle_features(
                        col,
                        self.c_before,
                        self.c_after,
                        logscale=True,
                        normalize=False, # Check
                        subtract_prev=True,
                    )
                else:
                    print("Something weird", si)            
            else:
                print(f"'{feature}' does not exist!")
        return data_array

    def cycle_features(
        self, col, c_before, c_after, logscale=True, normalize=True, subtract_prev=True
    ):
        """
        Create feature from cycle

        params: str col: column (e.g. 'R_gas_1')
        params: _type_ c_before: cycle data before
        params: _type_ c_after: cycle data after
        params: bool logscale: logscale, defaults to True
        params: bool normalize: normalize, defaults to True
        params: bool subtract_prev: subtract previous, defaults to True
        return: _type_: feature
        """

        # Select cycles before and after trigger, where 'before' should only be one cycle
        before = c_before[col]
        after = c_after[col]

        # Make sure that entry is valid
        if np.min(after) < 20:
            return np.ma.masked_all_like(after)

        if logscale:
            # Apply logarithmic scaling to 'before' and 'after'
            before = self.gas_signal(before)
            after = self.gas_signal(after)

        if normalize:
            # Apply cycle wise normalization to 'before' and 'after'
            before = self.fnorm(before)
            after = self.fnorm(after)

        # # Check that data is ... (?)
        # valid = after > 200.0

        if not subtract_prev:
            return after
        else:
            # Subtract the processed 'before' cycles from the data, and return
            return after - before

    def cycle_dc(self, col, c_before, c_after, phase=1, subtract_prev=True):
        """
        Get DC values from cycle

        :param str col: column, e.g. 'R_gas_1'
        :param _type_ c_before: cycle data before
        :param _type_ c_after: cycle data after
        :param int phase: phase, defaults to 1. (0 for cold, 1 for high)
        :param bool subtract_prev: subtract previous, defaults to True
        :return _type_: DC values
        """
        before = c_before[col][:, int(phase * self.period / 2)]
        after = c_after[col][:, int(phase * self.period / 2)]

        if not subtract_prev:
            return after
        else:
            # Subtract the processed 'before' cycles from the data, and return
            return after - before

    @staticmethod
    def gas_signal(x):
        """
        Gas signal, log(1/x)

        :param _type_ x: data
        :return _type_: gas signal
        """
        return np.log(1 / x)

    @staticmethod
    def high_pass(xs, alpha):
        """ 
        High pass filter

        :param _type_ xs: data
        :param float alpha: alpha value
        :return _type_: filtered data        
        """
        ys = np.copy(xs)
        z = np.copy(ys[0, :])
        for i in range(ys.shape[0]):
            z += (ys[i, :] - z) * alpha
            ys[i, :] -= z
        return ys

class Analysis:
    def __init__(
        self,
        data_dir,
        params
    ) -> None:
        self.params = params
        self.gases = params["gases"]
        self.period = params["period"]
        self.ncycles = params["ncycles"]
        self.cycles_gas = params["cycles_gas"]
        self.cycles_blank = params["cycles_blank"]
        self.data_dir = data_dir
        self.ms_start = params["ms_start"]
        self.ms_on = params["ms_on"]
        self.ms_off = params["ms_off"]
        self.ms_end = params["ms_end"]

    # New
    def get_trial_data(self, selected, sensors, batch=1, concentration=None):
        """
        Get trial data

        :param _type_ selected: selected trials
        :param _type_ sensors: list of sensors
        :param int batch: batch number, defaults to 1
        :param int concentration: concentration, defaults to None
        :return _type_: trial data
        """
        # Query
        if concentration is None:
            trials = selected
        else:
            trials = selected.query(f"concentration == {concentration}")

        trials = trials.reset_index(drop=True)

        trial_data_all = {}
        for ti, trial in trials.iterrows():
            t = TrialData(
                    trial, sensors=sensors, batch=batch, params=self.params
                )
            t.get_trial_nose(self.data_dir, self.ms_start, self.ms_end)
            t.get_cycles()
            
            trial_data_all[ti] = t

        return trial_data_all

    # New
    def get_X_y_ph(self, trial_data_all, feature, params, idxs=None, ignore_labels=False):
        """
        Get X, y, and phase

        :param _type_ trial_data_all: trial data
        :param str feature: feature to use
        :param _type_ params: parameter dictionary
        :param _type_ idxs: indices, defaults to None
        :param bool ignore_labels: ignore labels, defaults to False
        :return _type_: X, y, and phase
        """
        # Get parameters from params
        delay = params["delay"]
        on_buffer = params["on_buffer"]
        off_buffer = params["off_buffer"]
        ms_on = params["ms_on"]
        ms_off = params["ms_off"]
        period = params["period"]
        ms_end = params["ms_end"]

        # Check trial indices
        if idxs is None:
            _idxs = range(len(trial_data_all))
        else:
            _idxs = idxs

        # Get data for all trials trials
        X_all, y_gas1_all, y_gas2_all, phases_all, phases_rejected_all = [], [], [], [], []
        for idx in _idxs:
            X, y_gas1, y_gas2, phases, phases_rejected = [], [], [], [], []

            # Get gas and data features
            gas1 = trial_data_all[idx].get_gas1()
            gas2 = trial_data_all[idx].get_gas2()

            data_feature = trial_data_all[idx].get_feature(feature=feature)
            # Reshape (thanks ChatGPT)
            data_feature = data_feature.transpose(1, 0, 2).reshape(data_feature.shape[1], -1)

            # Get phase
            phase = np.array([trial_data_all[idx].phase_0th + m*period for m in range(len(data_feature))])            

            if ignore_labels:
                # Iterate through trial and select data
                for i, ph in enumerate(phase):
                        y_gas1.append(gas1)
                        y_gas2.append(gas2)
                        X.append(data_feature[i])
                        phases.append(ph)                    
            else:
                # Iterate through trial and select data that is either 'gas' or 'blank', reject the rest
                for i, ph in enumerate(phase):
                    # if (ph >= tgas_start+delay+on_buffer) and (ph < tgas_end+delay-period):
                    if (ph >= ms_on+delay+on_buffer) and (ph < ms_off+delay-period):
                        y_gas1.append(gas1)
                        X.append(data_feature[i])
                        phases.append(ph)
                    # elif (ph >= tgas_end+delay+off_buffer) and (ph < t_end+delay-period):
                    elif (ph >= ms_off+delay+off_buffer) and (ph < ms_end+delay-period):
                        y_gas1.append("blank")              
                        X.append(data_feature[i])
                        phases.append(ph)
                    else:
                        phases_rejected.append(ph)
            y_gas1_all.append(np.array(y_gas1))
            y_gas2_all.append(np.array(y_gas2))
            X_all.append(np.array(X))
            phases_all.append(np.array(phases))
            phases_rejected_all.append(np.array(phases_rejected))
        return X_all, y_gas1_all, y_gas2_all, phases_all, phases_rejected_all

    @staticmethod
    def sensors_list2str(sensors):
        """ 
        Convert list of sensors to string

        :param _type_ sensors: list of sensors
        :return _type_: string        
        """

        return "".join([str(s) for s in sensors])

    @staticmethod
    def get_model(classifier):
        """
        Get model

        :param str classifier: classifier to use
        :return _type_: model
        """
        if classifier == 'LinearSVC':
            model = make_pipeline(
                StandardScaler(), svm.LinearSVC(class_weight="balanced"), verbose=False
            )
        elif classifier == 'SVC_linear_kernel':
            model = make_pipeline(
                StandardScaler(), svm.SVC(kernel='linear', probability=True, class_weight="balanced", tol=1e-4, max_iter=10000), verbose=False
            )
        elif classifier == 'SVC_rbf_gridsearchcv':
            print("SVC_rbf")
            param_grid = [
                {'C': [1, 10, 100, 1000, 10000], 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000, 10000], 'gamma': [0.001, 0.0001, 0.00001], 'kernel': ['rbf']},
                ]
            grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)

            model = make_pipeline(
                StandardScaler(), grid,
            )

        elif classifier == 'SVC_rbf':
            model = make_pipeline(
                StandardScaler(), svm.SVC(kernel='rbf', probability=False, gamma=0.0001, C=1000, class_weight='balanced'), verbose=True
            )           
            
        elif classifier == 'kNN':
            model = make_pipeline(
                StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='uniform'), verbose=False
            )
        return model

    @staticmethod
    def train_classifier(X_train, y_train, classifier='kNN'):
        """ 
        Train classifier

        :param _type_ X_train: training data
        :param _type_ y_train: training labels
        :param str classifier: classifier to use, defaults to 'kNN'
        :return _type_: trained model        
        """
        model = Analysis.get_model(classifier)
        model.fit(X_train, y_train)
        return model


    @staticmethod
    def test_classifier(model, X_test, y_test, labels, metric="accuracy"):
        """
        Test classifier

        :param _type_ model: trained model
        :param _type_ X_test: test data
        :param _type_ y_test: test labels
        :param _type_ labels: list of labels
        :param str metric: metric to evaluate, defaults to "accuracy"        
        """
        y_test_pred = model.predict(X_test)
        if metric == "f1_score":
            return f1_score(y_test, y_test_pred, average="weighted"), confusion_matrix(y_test, y_test_pred, labels=labels), labels
        elif metric == "accuracy":
            return accuracy_score(y_test, y_test_pred), confusion_matrix(y_test, y_test_pred, labels=labels), labels


def train_validate_static(analysis, trial_data, kfold_indices_train, kfold_indices_val, sensors, features_all, params, verbose=False):
    """
    Train and validate static classifiers for all sensors and features

    :param _type_ analysis: class instance of Analysis
    :param _type_ trial_data: trial data
    :param _type_ kfold_indices_train: indices train
    :param _type_ kfold_indices_val: indices validation
    :param _type_ sensors: list of sensors
    :param _type_ features_all: list of features
    :param _type_ params: parameter dictionary
    :param bool verbose: verbose, defaults to False
    :return _type_: all models
    """
    classifier = params["classifier"]
    all_models = {}
    for sensors, feature in tqdm(zip([sensors]*4, features_all), total=len(features_all), desc="C=100%"):
        if verbose:
            print(sensors, feature)
        # Prepare
        sensor_str = analysis.sensors_list2str(sensors)
        all_models[feature] = {}
        all_models[feature][sensor_str] = {}

        # k-fold krossvalidation
        # Train
        for k, idx_train in kfold_indices_train.items():
            # Get X, y, phase
            X_train, y_train, y_gas2, phases_train, _ = analysis.get_X_y_ph(trial_data_all=trial_data, feature=feature, params=params, idxs=idx_train)
            # Flatten
            X_train_flat, y_train_flat, phases_train_flat = np.array(utils.flatten_list(X_train)), np.array(utils.flatten_list(y_train)), np.array(utils.flatten_list(phases_train))
            # Shuffle
            X_train_flat, y_train_flat, phases_train_flat = utils.unison_shuffled([X_train_flat, y_train_flat, phases_train_flat])
            # Train
            all_models[feature][sensor_str][k] = analysis.train_classifier(X_train_flat, y_train_flat, classifier)
        # Validate
        accs = []
        for k, idx_val in kfold_indices_val.items():
            # Get X, y, phase
            X_val, y_val, y_gas2, phases_val, _ = analysis.get_X_y_ph(trial_data_all=trial_data, feature=feature, params=params, idxs=idx_val)
            # Flatten
            X_val_flat, y_val_flat, phases_val_flat = np.array(utils.flatten_list(X_val)), np.array(utils.flatten_list(y_val)), np.array(utils.flatten_list(phases_val))
            # Shuffle
            X_val_flat, y_val_flat, phases_val_flat = utils.unison_shuffled([X_val_flat, y_val_flat, phases_val_flat])
            # Predict
            pred = all_models[feature][sensor_str][k].predict(X_val_flat)
            # Get accuracy and confusion matrix
            acc = accuracy_score(pred, y_val_flat)
            cm = confusion_matrix(pred, y_val_flat)
            accs.append(acc)
        if verbose:
            print(np.mean(accs), np.std(accs))
    return all_models

def validate_static_concentrationvars(all_models, analysis, concentrations, index_df_pulses_1s_c1234_train, sensors, features_all, params, verbose=False):
    """ 
    Validate static classifiers for all sensors and features

    :param _type_ all_models: all models
    :param _type_ analysis: class instance of Analysis
    :param _type_ concentrations: list of concentrations
    :param _type_ index_df_pulses_1s_c1234_train: index dataframe
    :param _type_ sensors: list of sensors
    :param _type_ features_all: list of features
    :param _type_ params: parameter dictionary
    :param bool verbose: verbose, defaults to False
    :return _type_: validation scores, confusion matrices, labels
    """

    scores_val, confmats_val, labels_val = {}, {}, {}
    for concentration in concentrations:
        # Select indices and dictionary entry for corresponding concentration
        scores_val[concentration], confmats_val[concentration], labels_val[concentration] = {}, {}, {}   
        pulses_1s_cx = index_df_pulses_1s_c1234_train.query(f"concentration == {concentration}")

        # For all sensors-feature pairs
        for sensors, feature in tqdm(zip([sensors]*4, features_all), total=len(features_all), desc=f"C={concentration}%"):
            # Prepare
            sensor_str = analysis.sensors_list2str(sensors)
            models = all_models[feature][sensor_str]
            # Get trial data
            trial_data = analysis.get_trial_data(pulses_1s_cx, sensors=sensors)
            # Get X, y, phase
            X_val, y_val, y_gas2, phases_val, _ = analysis.get_X_y_ph(trial_data_all=trial_data, feature=feature, params=params)
            # Flatten
            X_val_flat, y_val_flat, phases_val_flat = np.array(utils.flatten_list(X_val)), np.array(utils.flatten_list(y_val)), np.array(utils.flatten_list(phases_val))
            # k-fold crossvalidation
            scores_val[concentration][feature], confmats_val[concentration][feature], labels_val[concentration][feature] = {}, {}, {}
            for k, model in models.items():
                scores_val[concentration][feature][k], confmats_val[concentration][feature][k], labels_val[concentration][feature][k] = analysis.test_classifier(model, X_val_flat, y_val_flat, labels=analysis.gases, metric='accuracy')
                confmats_val[concentration][feature][k] = confmats_val[concentration][feature][k].tolist()
            if verbose:
                # Print validation scores
                print(np.mean(list(scores_val[concentration][feature].values())), np.std(list(scores_val[concentration][feature].values())))
    return scores_val, confmats_val, labels_val

def test_static_concentrationvars(all_models, analysis, concentrations, index_df_pulses_1s_c1234_test, index_df_pulses_1s_c5_test, sensors, features_all, params, verbose=False):
    """
    Test static classifiers for all sensors and features

    :param _type_ all_models: all models
    :param _type_ analysis: class instance of Analysis
    :param _type_ concentrations: list of concentrations
    :param _type_ index_df_pulses_1s_c1234_test: index dataframe
    :param _type_ index_df_pulses_1s_c5_test: index dataframe
    :param _type_ sensors: list of sensors
    :param _type_ features_all: list of features
    :param _type_ params: parameter dictionary
    :param bool verbose: verbose, defaults to False
    :return _type_: test scores, confusion matrices, labels
    """
    scores_test, confmats, labels = {}, {}, {}
    # Concentration 1-5
    for concentration in concentrations:
        # Select indices and dictionary entry for corresponding concentration
        scores_test[concentration], confmats[concentration], labels[concentration] = {}, {}, {}
        if concentration != 100:    
            pulses_1s_cx = index_df_pulses_1s_c1234_test.query(f"concentration == {concentration}")
        else:
            pulses_1s_cx = index_df_pulses_1s_c5_test

        # For all sensors-feature pairs
        for sensors, feature in tqdm(zip([sensors]*4, features_all), total=len(features_all), desc=f"C={concentration}%"):
            # Prepare
            sensor_str = analysis.sensors_list2str(sensors)
            models = all_models[feature][sensor_str]
            # Get trial data
            trial_data = analysis.get_trial_data(pulses_1s_cx, sensors=sensors)
            # Get X, y, phase
            X_test, y_test, y_gas2, phases_test, _ = analysis.get_X_y_ph(trial_data_all=trial_data, feature=feature, params=params)
            # Flatten
            X_test_flat, y_test_flat, phases_test_flat = np.array(utils.flatten_list(X_test)), np.array(utils.flatten_list(y_test)), np.array(utils.flatten_list(phases_test))
            # k-fold crossvalidation
            scores_test[concentration][feature], confmats[concentration][feature], labels[concentration][feature] = {}, {}, {}
            for k, model in models.items():
                scores_test[concentration][feature][k], confmats[concentration][feature][k], labels[concentration][feature][k] = analysis.test_classifier(model, X_test_flat, y_test_flat, labels=analysis.gases, metric='accuracy')
                confmats[concentration][feature][k] = confmats[concentration][feature][k].tolist()
            # Print test scores
            if verbose:
                print(np.mean(list(scores_test[concentration][feature].values())), np.std(list(scores_test[concentration][feature].values())))
    return scores_test, confmats, labels

def train_dynamic(kfold_indices_train, trial_data_trainval, analysis, feature, params):
    """
    Train dynamic classifiers

    :param _type_ kfold_indices_train: indices train
    :param _type_ trial_data_trainval: trial data
    :param _type_ analysis: class instance of Analysis
    :param str feature: feature to use
    :param _type_ params: parameter dictionary
    :return _type_: all models
    """
    # k-fold crossvalidation
    classifier = params["classifier"]
    all_models = {}
    for k, idx_train in tqdm(kfold_indices_train.items()):
        # Get X, y, phase
        X_train, y_train, _, phases_train, _ = analysis.get_X_y_ph(trial_data_all=trial_data_trainval, feature=feature, params=params, idxs=idx_train)
        # Flatten
        X_train_flat, y_train_flat, phases_train_flat = np.array(utils.flatten_list(X_train)), np.array(utils.flatten_list(y_train)), np.array(utils.flatten_list(phases_train))
        # Shuffle
        X_train_flat, y_train_flat, phases_train_flat = utils.unison_shuffled([X_train_flat, y_train_flat, phases_train_flat])
        # Train
        all_models[k] = analysis.train_classifier(X_train_flat, y_train_flat, classifier)
    return all_models

def validate_dynamic(kfold_indices_val, all_models, analysis, trial_data_trainval, selected_shortpulses_dict_trainval, sensors, feature, params, results_dir_single_pulses):
    """
    Validate dynamic classifiers

    :param _type_ kfold_indices_val: indices validation
    :param _type_ all_models: all models
    :param _type_ analysis: class instance of Analysis
    :param _type_ trial_data_trainval: trial data
    :param _type_ selected_shortpulses_dict_trainval: selected short pulses
    :param _type_ sensors: list of sensors
    :param str feature: feature to use
    :param _type_ params: parameter dictionary
    :param _type_ results_dir_single_pulses: results directory
    :return _type_: validation scores
    """   
    # Create empty lists for all widths
    widths_val, all_f1_val, all_acc_val, all_detection_val, all_phases_onset_val, all_phases_offset_val, all_confidence_mean_val = [], [], [], [], [], [], []

    # Validate on 1s pulses, dynamic / over time
    width_ms = 1000
    # k-fold crossvalidation
    y_val_all, y_pred_val_all, phases_val_all = [], [], []
    for k, idx_val in kfold_indices_val.items():
        model = all_models[k]
        # Get X, y, phase
        X_val, y_val, _, phases_val, _ = analysis.get_X_y_ph(trial_data_all=trial_data_trainval, feature=feature, params=params, idxs=idx_val, ignore_labels=True)
        # Iterate over trials, and predict
        y_pred_val = [] 
        for X, y, ph in zip(X_val, y_val, phases_val):
            y_pred_val.append(model.predict(X))
        y_val_all.append(y_val)
        y_pred_val_all.append(y_pred_val)
        phases_val_all.append(phases_val)

    # Save results
    np.save(results_dir_single_pulses.joinpath(f'y_val_all_{width_ms}.npy'), y_val_all)
    np.save(results_dir_single_pulses.joinpath(f'y_pred_val_all_{width_ms}.npy'), y_pred_val_all)
    np.save(results_dir_single_pulses.joinpath(f'phases_val_all_{width_ms}.npy'), phases_val_all)

    # Load again (should be same as before)
    width_ms = 1000
    y_val_all = np.load(results_dir_single_pulses.joinpath(f'y_val_all_{width_ms}.npy'))
    y_pred_val_all = np.load(results_dir_single_pulses.joinpath(f'y_pred_val_all_{width_ms}.npy'))
    phases_val_all = np.load(results_dir_single_pulses.joinpath(f'phases_val_all_{width_ms}.npy'))

    # Collect results for 1s pulses
    accs, f1s, detection_rates, phases_onsets, phases_offsets, confidence_means = [], [], [], [], [], []
    for y_val, y_pred_val, phases_val in zip(y_val_all, y_pred_val_all, phases_val_all):
        acc, f1, detection_rate, phases_onset, phases_offset, confidence_mean = utils.score_temporalclassification(y_val, y_pred_val, phases_val, width_ms)
        accs.append(acc)
        f1s.append(f1)
        detection_rates.append(detection_rate)
        phases_onsets.append(phases_onset)
        phases_offsets.append(phases_offset)
        confidence_means.append(confidence_mean)
    # get means for each
    acc = np.mean(accs)
    f1 = np.mean(f1s)
    detection_rate = np.mean(detection_rates)
    phases_onset = np.mean(phases_onsets, axis=0)
    phases_offset = np.mean(phases_offsets, axis=0)
    confidence_mean = np.mean(confidence_means)

    # Append to lists
    widths_val.append(width_ms)
    all_f1_val.append(f1)
    all_acc_val.append(acc)
    all_detection_val.append(detection_rate)
    all_phases_onset_val.append(phases_onset)
    all_phases_offset_val.append(phases_offset)

    # Validate on shorter pulses over time (i.e. on 10ms, 20ms, 50ms, 100ms, 200ms, 500ms)
    for width, pulses_short in selected_shortpulses_dict_trainval.items():

        # get trial data
        trial_data_short_val = analysis.get_trial_data(pulses_short, sensors=sensors)
        width_ms = int(float(width[:-1])*1000)

        # k-fold crossvalidation
        y_val_all, y_pred_val_all, phases_val_all = [], [], []
        for k, idx_val in kfold_indices_val.items():
            model = all_models[k]
            # Get X, y, phase
            X_val, y_val, _, phases_val, _ = analysis.get_X_y_ph(trial_data_all=trial_data_short_val, feature=feature, params=params, ignore_labels=True)
            # Iterate over trials, and predict
            y_pred_val = [] 
            for X in X_val:
                y_pred_val.append(model.predict(X))
            y_val_all.append(y_val)
            y_pred_val_all.append(y_pred_val)
            phases_val_all.append(phases_val)
        # majority vote of all models
        y_pred_val_winner = utils.vote_winner(y_pred_val_all)

        # Save results
        np.save(results_dir_single_pulses.joinpath(f'y_val_all_{width_ms}.npy'), y_val_all)
        np.save(results_dir_single_pulses.joinpath(f'y_pred_val_all_{width_ms}.npy'), y_pred_val_winner)
        np.save(results_dir_single_pulses.joinpath(f'phases_val_all_{width_ms}.npy'), phases_val_all)

        # Collect results for shorter pulses. 
        accs, f1s, detection_rates, phases_onsets, phases_offsets, confidence_means = [], [], [], [], [], []
        for y_val, y_pred_val, phases_val in zip(y_val_all, y_pred_val_winner, phases_val_all):
            acc, f1, detection_rate, phases_onset, phases_offset, confidence_mean = utils.score_temporalclassification(y_val, y_pred_val, phases_val, width_ms)
            accs.append(acc)
            f1s.append(f1)
            detection_rates.append(detection_rate)
            phases_onsets.append(phases_onset)
            phases_offsets.append(phases_offset)
            confidence_means.append(confidence_mean)
        # get means for each
        acc = np.mean(accs)
        f1 = np.mean(f1s)
        detection_rate = np.mean(detection_rates)
        phases_onset = np.mean(phases_onsets, axis=0)
        phases_offset = np.mean(phases_offsets, axis=0)
        confidence_mean = np.mean(confidence_means)
        
        
        widths_val.append(int(width_ms))
        all_f1_val.append(f1)
        all_acc_val.append(acc)
        all_detection_val.append(detection_rate)
        all_phases_onset_val.append(phases_onset)
        all_phases_offset_val.append(phases_offset)

    return widths_val, all_f1_val, all_acc_val, all_detection_val, all_phases_onset_val, all_phases_offset_val

def test_dynamic(kfold_indices_val, all_models, analysis, trial_data_test, selected_shortpulses_dict_test, sensors, feature, params, results_dir_single_pulses):
    """
    Test dynamic classifiers

    :param _type_ kfold_indices_val: indices validation
    :param _type_ all_models: all models
    :param _type_ analysis: class instance of Analysis
    :param _type_ trial_data_test: trial data
    :param _type_ selected_shortpulses_dict_test: selected short pulses
    :param _type_ sensors: list of sensors
    :param str feature: feature to use
    :param _type_ params: parameter dictionary
    :param _type_ results_dir_single_pulses: results directory
    """
    # Create empty lists for all widths
    widths_test, all_f1_test, all_acc_test, all_detection_test, all_phases_onset_test, all_phases_offset_test, all_confidence_mean_test = [], [], [], [], [], [], []

    # Test on 1s pulses over time
    width_ms = 1000
    # Use ensemble of crossvalidation models
    y_test_all, y_pred_test_all, phases_test_all = [], [], []
    for k, _ in kfold_indices_val.items():
        model = all_models[k]
        # Get X, y, phase
        X_test, y_test, _, phases_test, _ = analysis.get_X_y_ph(trial_data_all=trial_data_test, feature=feature, params=params, idxs=None, ignore_labels=True)
        # Iterate over trials, and predict
        y_pred_test = [] 
        for X, y, ph in zip(X_test, y_test, phases_test):
            y_pred_test.append(model.predict(X))
        y_test_all.append(y_test)
        y_pred_test_all.append(y_pred_test)
        phases_test_all.append(phases_test)

    # Convert to numpy arrays
    phases_test_all, y_test_all, y_pred_test_all = np.array(phases_test_all), np.array(y_test_all), np.array(y_pred_test_all)

    # Majority vote across models
    y_pred_test_winner = utils.vote_winner(y_pred_test_all)

    # Save results
    np.save(results_dir_single_pulses.joinpath(f'y_test_all_{width_ms}.npy'), y_test_all)
    np.save(results_dir_single_pulses.joinpath(f'y_pred_test_all_{width_ms}.npy'), y_pred_test_winner)
    np.save(results_dir_single_pulses.joinpath(f'phases_test_all_{width_ms}.npy'), phases_test_all)

    width_ms = 1000
    # y_test_all = np.load(results_dir_single_pulses.joinpath(f'y_test_all_{width_ms}.npy'), allow_pickle=True)
    # y_pred_test_winner = np.load(results_dir_single_pulses.joinpath(f'y_pred_test_all_{width_ms}.npy'), allow_pickle=True)
    # phases_test_all = np.load(results_dir_single_pulses.joinpath(f'phases_test_all_{width_ms}.npy'), allow_pickle=True)

    # Collect results for 1s pulses
    accs, f1s, detection_rates, phases_onsets, phases_offsets, confidence_means = [], [], [], [], [], []
    for y_test, y_pred_test, phases_test in zip(y_test_all, y_pred_test_winner, phases_test_all):
        acc, f1, detection_rate, phases_onset, phases_offset, confidence_mean = utils.score_temporalclassification(y_test, y_pred_test, phases_test, width_ms)
        accs.append(acc)
        f1s.append(f1)
        detection_rates.append(detection_rate)
        phases_onsets.append(phases_onset)
        phases_offsets.append(phases_offset)
        confidence_means.append(confidence_mean)
    # get means for each
    acc = np.mean(accs)
    f1 = np.mean(f1s)
    detection_rate = np.mean(detection_rates)
    phases_onset = np.mean(phases_onsets, axis=0)
    phases_offset = np.mean(phases_offsets, axis=0)
    confidence_mean = np.mean(confidence_means)

    # Append to lists
    widths_test.append(width_ms)
    all_f1_test.append(f1)
    all_acc_test.append(acc)
    all_detection_test.append(detection_rate)
    all_phases_onset_test.append(phases_onset)
    all_phases_offset_test.append(phases_offset)

    # Test on shorter pulses over time (i.e. on 10ms, 20ms, 50ms, 100ms, 200ms, 500ms)
    for width, pulses_short in selected_shortpulses_dict_test.items():

        # get trial data
        trial_data_short = analysis.get_trial_data(pulses_short, sensors=sensors)

        width_ms = int(float(width[:-1])*1000)
        # Use ensemble of crossvalidation models
        y_test_all, y_pred_test_all, phases_test_all = [], [], []
        for k, _ in kfold_indices_val.items():
            model = all_models[k]
            # Get X, y, phase
            X_test, y_test, _, phases_test, _ = analysis.get_X_y_ph(trial_data_all=trial_data_short, feature=feature, params=params, ignore_labels=True, idxs=None)
            # Iterate over trials, and predict
            y_pred_test = [] 
            for X in X_test:
                y_pred_test.append(model.predict(X))
            y_test_all.append(y_test)
            y_pred_test_all.append(y_pred_test)
            phases_test_all.append(phases_test)

        y_pred_test_winner = utils.vote_winner(y_pred_test_all)

        # Save results
        np.save(results_dir_single_pulses.joinpath(f'y_test_all_{width_ms}.npy'), y_test_all)
        np.save(results_dir_single_pulses.joinpath(f'y_pred_test_all_{width_ms}.npy'), y_pred_test_winner)
        np.save(results_dir_single_pulses.joinpath(f'phases_test_all_{width_ms}.npy'), phases_test_all)

        # Collect results for shorter pulses
        accs, f1s, detection_rates, phases_onsets, phases_offsets, confidence_means = [], [], [], [], [], []
        for y_test, y_pred_test, phases_test in zip(y_test_all, y_pred_test_winner, phases_test_all):
            acc, f1, detection_rate, phases_onset, phases_offset, confidence_mean = utils.score_temporalclassification(y_test, y_pred_test, phases_test, width_ms)
            accs.append(acc)
            f1s.append(f1)
            detection_rates.append(detection_rate)
            phases_onsets.append(phases_onset)
            phases_offsets.append(phases_offset)
            confidence_means.append(confidence_mean)
        # get means for each
        acc = np.mean(accs)
        f1 = np.mean(f1s)
        detection_rate = np.mean(detection_rates)
        phases_onset = np.mean(phases_onsets, axis=0)
        phases_offset = np.mean(phases_offsets, axis=0)
        confidence_mean = np.mean(confidence_means)
        
        widths_test.append(int(width_ms))
        all_f1_test.append(f1)
        all_acc_test.append(acc)
        all_detection_test.append(detection_rate)
        all_phases_onset_test.append(phases_onset)
        all_phases_offset_test.append(phases_offset)

    return widths_test, all_f1_test, all_acc_test, all_detection_test, all_phases_onset_test, all_phases_offset_test

def test_dynamic_acorr(selected_acorr_dict_test, analysis, kfold_indices_val, all_models, sensors, feature, params, results_dir_acorr, verbose=False):
    """
    Test dynamic classifiers on acorr pulses

    :param _type_ selected_acorr_dict_test: selected acorr pulses
    :param _type_ analysis: class instance of Analysis
    :param _type_ kfold_indices_val: indices validation
    :param _type_ all_models: all models
    :param _type_ sensors: list of sensors
    :param str feature: feature to use
    :param _type_ params: parameter dictionary
    :param _type_ results_dir_acorr: results directory
    :param bool verbose: verbose, defaults to False
    :return _type_: frequencies, accuracies, detection rates
    """
    freqs, all_acc_gas1, all_acc_gas2, all_detection = [], [], [], []

    # Test on acorr pulses over time (i.e. on 1Hz, 2Hz, 5Hz, 10Hz, 20Hz, 40Hz, 60Hz)
    for freq, mixture_acorr in selected_acorr_dict_test.items():
        if verbose:
            print(freq)
        # get trial data
        trial_data_acorr = analysis.get_trial_data(mixture_acorr, sensors=sensors)
        y1_test_all, y2_test_all, y_pred_test_all, phases_test_all = [], [], [], []
        # test using all models from k-fold crossvalidation
        for k, _ in kfold_indices_val.items():
            model = all_models[k]
            # Get X, y, phase
            X_test, y1_test, y2_test, phases_test, _ = analysis.get_X_y_ph(trial_data_all=trial_data_acorr, feature=feature, params=params, ignore_labels=True)
            # Iterate over trials, and predict
            y_pred_test = [] 
            for X in X_test:
                y_pred_test.append(model.predict(X))
            y1_test_all.append(y1_test)
            y2_test_all.append(y2_test)
            y_pred_test_all.append(y_pred_test)
            phases_test_all.append(phases_test)
        # majority vote of all models
        y_pred_test_winner = utils.vote_winner(y_pred_test_all)

        # Collect results for shorter pulses
        detection_rates, accs_gas1, accs_gas2 = [], [], []
        for y1_test, y2_test, y_pred_test, phases_test in zip(y1_test_all, y2_test_all, y_pred_test_winner, phases_test_all):
            detection_rate, acc_gas1, acc_gas2 = utils.score_temporalclassification_multiclass(y1_test, y2_test, y_pred_test, phases_test)
            detection_rates.append(detection_rate)
            accs_gas1.append(acc_gas1)
            accs_gas2.append(acc_gas2)
        # get means for each
        detection_rate = np.mean(detection_rates)
        acc_gas1 = np.mean(accs_gas1)
        acc_gas2 = np.mean(accs_gas2)

        freqs.append(int(freq[:-2]))
        all_detection.append(detection_rate)
        all_acc_gas1.append(acc_gas1)
        all_acc_gas2.append(acc_gas2)
        if verbose:
            print(detection_rate, acc_gas1, acc_gas2)

        # Save into numpy arrays
        np.save(results_dir_acorr.joinpath(f'y1_test_all_{freq}.npy'), y1_test_all)
        np.save(results_dir_acorr.joinpath(f'y2_test_all_{freq}.npy'), y2_test_all)
        np.save(results_dir_acorr.joinpath(f'y_pred_test_all_{freq}.npy'), y_pred_test_winner)
        np.save(results_dir_acorr.joinpath(f'phases_test_all_{freq}.npy'), phases_test_all)

    return freqs, all_acc_gas1, all_acc_gas2, all_detection