import sys, os
from pathlib import Path

home_path = "/Users/nd21aad/Phd/Projects/enose-analysis-crick"  # Define project root directory
os.chdir(home_path)         # Change working directory to project root directory
sys.path.append(home_path)  # Add project root directory to python path
home_path = Path(home_path)

from tqdm import tqdm
import json
from joblib import dump, load
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import src.temporal_structure.helpers as helpers

def trainval_pid(gases_allcombos, kinds, freqs, freqpairs, pid_index, experiment_dir, result_dir_parent, run_params):
    """ 
    Train and validate classifiers to distinguish between modulation frequencies and patterns (corr vs acorr)

    :param gases_allcombos: list of gas combinations to train on
    :param kinds: list of kinds to train on
    :param freqs: list of modulation frequencies
    :param freqpairs: dictionary of frequency pairs
    :param pid_index: index of pid trials
    :param experiment_dir: directory of experiment data
    :param result_dir_parent: directory to save results
    :param run_params: dictionary of run parameters
    :return: None
    """
    n_seeds = run_params['n_seeds']
    n_splits = run_params['n_splits']
    ms_start = run_params['ms_start']
    ms_end = run_params['ms_end']
    stimulus_start = run_params['stimulus_start']
    stimulus_duration = run_params['stimulus_duration']
    buffer = run_params['buffer']
    modality = 'pid'
    print("Training")
    for gases in gases_allcombos:
        gas_str = '_'.join(gases)
        print(gas_str)
        # Make subdirectory 
        result_dir = result_dir_parent.joinpath(gas_str)
        result_dir.mkdir(exist_ok=True, parents=True)
        # Select index
        index_train_selected = pid_index.query(f"(kind == '{kinds[0]}' | kind == '{kinds[1]}') & ((gas1 == '{gases[0]}' & gas2 == '{gases[1]}') | (gas1 == '{gases[1]}' & gas2 == '{gases[0]}'))")
        index_train_selected = index_train_selected.query(f"shape != '1Hz'")
        index_train_selected_idx = index_train_selected.index.to_numpy()
        # Train / Val classifiers
        print("Train & Validate")

        # Initialise accuracy and model lists
        accs_freqs_allfeatures_all_val, accs_freqs_magnitude_all_val, accs_freqs_phase_all_val, accs_freqs_frequency_all_val = [], [], [], []
        accs_pattern_allfeatures_all_val, accs_pattern_magnitude_all_val, accs_pattern_phase_all_val, accs_pattern_frequency_all_val = [], [], [], []
        accs_freqpairs_allfeatures = {freqpair: [] for freqpair in freqpairs.keys()}
        accs_freqpairs_magnitude = {freqpair: [] for freqpair in freqpairs.keys()}
        accs_freqpairs_phase = {freqpair: [] for freqpair in freqpairs.keys()}
        accs_freqpairs_freq = {freqpair: [] for freqpair in freqpairs.keys()}
        model_ensembles_freq = []
        model_ensembles_pattern = []
        model_ensembles_freqpair = []
        # Iterate over different random seeds
        for random_seed in tqdm(range(n_seeds)):
            print("seed", random_seed)
            # Unshuffled k-fold CV splitting (unshuffled = block-wise continuous)
            kf = KFold(n_splits=n_splits, shuffle=False)
            # Initialise score collections
            all_scores_freq, all_cm_freq = [], []
            all_scores_pattern, all_cm_pattern = [], []
            all_freqs_true = {freq: [] for freq in freqs}
            all_freqs_freq_pred_magnitude = {freq: [] for freq in freqs}
            all_freqs_freq_pred_phase = {freq: [] for freq in freqs}
            all_freqs_freq_pred_freq = {freq: [] for freq in freqs}
            all_freqs_freq_pred_allfeatures = {freq: [] for freq in freqs}
            all_patterns_freq_true = {freq: [] for freq in freqs}
            all_patterns_freq_pred_magnitude = {freq: [] for freq in freqs}
            all_patterns_freq_pred_phase = {freq: [] for freq in freqs}
            all_patterns_freq_pred_freq = {freq: [] for freq in freqs}
            all_patterns_freq_pred_allfeatures = {freq: [] for freq in freqs}
            all_freqpairs_true = {freqpair: [] for freqpair in freqpairs.keys()}
            all_freqpairs_pred_magnitude = {freqpair: [] for freqpair in freqpairs.keys()}
            all_freqpairs_pred_phase = {freqpair: [] for freqpair in freqpairs.keys()}
            all_freqpairs_pred_freq = {freqpair: [] for freqpair in freqpairs.keys()}
            all_freqpairs_pred_allfeatures = {freqpair: [] for freqpair in freqpairs.keys()}
            model_ensemble_freq = []
            model_ensemble_pattern = []
            model_ensemble_freqpair = []
            # Cross validation over splits
            for i, (train_index, val_index) in enumerate(kf.split(index_train_selected_idx)):
                # Separate trials in train / val
                trials_train, trials_val = index_train_selected.loc[index_train_selected_idx[train_index]], index_train_selected.loc[index_train_selected_idx[val_index]]
                # TRAIN
                # Get features
                y_train_freqs, y_train_patterns, X_train_freqs, X_train_magnitudes, X_train_phases, X_train_allfeatures = helpers.get_freqspace_features_pid(trials_train, experiment_dir.joinpath('pid'), modality, ms_start, ms_end, stimulus_start, stimulus_duration+buffer, set='training')
                # Instantiate and train classifier to distinguish modulation frequency, for different features
                # Phase + Magnitude + Freq
                model_freq_allfeatures = RandomForestClassifier(random_state=random_seed, class_weight='balanced')
                model_freq_allfeatures.fit(X_train_allfeatures, y_train_freqs)
                # Magnitude
                model_freq_magnitude = RandomForestClassifier(random_state=random_seed, class_weight='balanced')
                model_freq_magnitude.fit(X_train_magnitudes, y_train_freqs)
                # Phase
                model_freq_phase = RandomForestClassifier(random_state=random_seed, class_weight='balanced')
                model_freq_phase.fit(X_train_phases, y_train_freqs)
                # Freq
                model_freq_freq = RandomForestClassifier(random_state=random_seed, class_weight='balanced')
                model_freq_freq.fit(X_train_freqs, y_train_freqs)
                # Instantiate and train classifier to distinguish pattern (corr vs acorr), for different features
                # Phase + Magnitude + Freq
                model_pattern_allfeatures = RandomForestClassifier(random_state=random_seed, n_estimators=100, class_weight='balanced_subsample')
                model_pattern_allfeatures.fit(X_train_allfeatures, y_train_patterns)
                # Magnitude
                model_pattern_magnitude = RandomForestClassifier(random_state=random_seed, n_estimators=100, class_weight='balanced_subsample')
                model_pattern_magnitude.fit(X_train_magnitudes, y_train_patterns)
                # Phase
                model_pattern_phase = RandomForestClassifier(random_state=random_seed, n_estimators=100, class_weight='balanced_subsample')
                model_pattern_phase.fit(X_train_phases, y_train_patterns)
                # Freq
                model_pattern_freq = RandomForestClassifier(random_state=random_seed, n_estimators=100, class_weight='balanced_subsample')
                model_pattern_freq.fit(X_train_freqs, y_train_patterns)
                # For all features, Train classifier to distinguish between frequency pairs
                models_freqpair_allfeatures, models_freqpair_magnitudes, models_freqpair_phase, models_freqpair_freq = {}, {}, {}, {}
                for pair in freqpairs.keys():
                    y_train_pair = []
                    X_train_pair = []
                    for y_train, X_train in zip(y_train_freqs, X_train_allfeatures):
                        if int(y_train) not in freqpairs[pair]:
                            continue
                        else:
                            y_train_pair.append(y_train)
                            X_train_pair.append(X_train)
                    model_freqpair = RandomForestClassifier(random_state=random_seed, class_weight='balanced')
                    model_freqpair.fit(X_train_pair, y_train_pair)
                    models_freqpair_allfeatures[pair] = model_freqpair
                    y_train_pair = []
                    X_train_pair = []
                    for y_train, X_train in zip(y_train_freqs, X_train_magnitudes):
                        if int(y_train) not in freqpairs[pair]:
                            continue
                        else:
                            y_train_pair.append(y_train)
                            X_train_pair.append(X_train)
                    model_freqpair = RandomForestClassifier(random_state=random_seed, class_weight='balanced')
                    model_freqpair.fit(X_train_pair, y_train_pair)
                    models_freqpair_magnitudes[pair] = model_freqpair
                    y_train_pair = []
                    X_train_pair = []
                    for y_train, X_train in zip(y_train_freqs, X_train_phases):
                        if int(y_train) not in freqpairs[pair]:
                            continue
                        else:
                            y_train_pair.append(y_train)
                            X_train_pair.append(X_train)
                    model_freqpair = RandomForestClassifier(random_state=random_seed, class_weight='balanced')
                    model_freqpair.fit(X_train_pair, y_train_pair)
                    models_freqpair_phase[pair] = model_freqpair
                    y_train_pair = []
                    X_train_pair = []
                    for y_train, X_train in zip(y_train_freqs, X_train_freqs):
                        if int(y_train) not in freqpairs[pair]:
                            continue
                        else:
                            y_train_pair.append(y_train)
                            X_train_pair.append(X_train)
                    model_freqpair = RandomForestClassifier(random_state=random_seed)#, class_weight='balanced')
                    model_freqpair.fit(X_train_pair, y_train_pair)
                    models_freqpair_freq[pair] = model_freqpair
                # For ensemble, take what worked best in validation split (which here is 'allfeatures')
                model_ensemble_freqpair.append(models_freqpair_allfeatures)
                # Get Validation Features
                y_val_freqs, y_val_patterns, X_val_freqs, X_val_magnitudes, X_val_phases, X_val_allfeatures = helpers.get_freqspace_features_pid(trials_val, experiment_dir.joinpath('pid'), modality, ms_start, ms_end, stimulus_start, stimulus_duration+buffer, set='validation')
                # Get frequency-wise predictions for each feature
                for f, x_allfeatures, x_mag, x_phase, x_freq, y in zip(y_val_freqs, X_val_allfeatures, X_val_magnitudes, X_val_phases, X_val_freqs, y_val_patterns):
                    all_freqs_true[f].append(f)
                    all_freqs_freq_pred_allfeatures[f].append(model_freq_allfeatures.predict(x_allfeatures.reshape(1, -1)))
                    all_freqs_freq_pred_magnitude[f].append(model_freq_magnitude.predict(x_mag.reshape(1, -1)))
                    all_freqs_freq_pred_phase[f].append(model_freq_phase.predict(x_phase.reshape(1, -1)))
                    all_freqs_freq_pred_freq[f].append(model_freq_freq.predict(x_freq.reshape(1, -1)))
                    all_patterns_freq_true[f].append(y)
                    all_patterns_freq_pred_allfeatures[f].append(model_pattern_allfeatures.predict(x_allfeatures.reshape(1, -1)))
                    all_patterns_freq_pred_magnitude[f].append(model_pattern_magnitude.predict(x_mag.reshape(1, -1)))
                    all_patterns_freq_pred_phase[f].append(model_pattern_phase.predict(x_phase.reshape(1, -1)))
                    all_patterns_freq_pred_freq[f].append(model_pattern_freq.predict(x_freq.reshape(1, -1)))
                # Validate classifier to distinguish between frequency pairs
                for pair in freqpairs.keys():
                    y_val_pair = []
                    X_val_pair_allfeatures, X_val_pair_magnitudes, X_val_pair_phases, X_val_pair_freqs = [], [], [], []
                    for y_val, _X_val_allfeatures, _X_val_magnitudes, _X_val_phases, _X_val_freqs in zip(y_val_freqs, X_val_allfeatures, X_val_magnitudes, X_val_phases, X_val_freqs):
                        if int(y_val) not in freqpairs[pair]:
                            continue
                        else:
                            y_val_pair.append(y_val)
                            X_val_pair_allfeatures.append(_X_val_allfeatures)
                            X_val_pair_magnitudes.append(_X_val_magnitudes)
                            X_val_pair_phases.append(_X_val_phases)
                            X_val_pair_freqs.append(_X_val_freqs)    
                    if len(y_val_pair)==0:
                        continue
                    all_freqpairs_true[pair] += y_val_pair
                    model_freqpair = models_freqpair_allfeatures[pair]
                    y_pred_pair = list(model_freqpair.predict(X_val_pair_allfeatures))
                    all_freqpairs_pred_allfeatures[pair] += y_pred_pair
                    model_freqpair = models_freqpair_magnitudes[pair]
                    y_pred_pair = list(model_freqpair.predict(X_val_pair_magnitudes))
                    all_freqpairs_pred_magnitude[pair] += y_pred_pair
                    model_freqpair = models_freqpair_phase[pair]
                    y_pred_pair = list(model_freqpair.predict(X_val_pair_phases))
                    all_freqpairs_pred_phase[pair] += y_pred_pair
                    model_freqpair = models_freqpair_freq[pair]
                    y_pred_pair = list(model_freqpair.predict(X_val_pair_freqs))
                    all_freqpairs_pred_freq[pair] += y_pred_pair
                model_ensemble_freq.append(model_freq_allfeatures)
                model_ensemble_pattern.append(model_pattern_allfeatures)
            # print("Get accs for freqpairs")
            for freqpair in all_freqpairs_true.keys():
                acc = accuracy_score(all_freqpairs_true[freqpair], all_freqpairs_pred_allfeatures[freqpair])
                accs_freqpairs_allfeatures[freqpair].append(acc)
                acc = accuracy_score(all_freqpairs_true[freqpair], all_freqpairs_pred_magnitude[freqpair])
                accs_freqpairs_magnitude[freqpair].append(acc)
                acc = accuracy_score(all_freqpairs_true[freqpair], all_freqpairs_pred_phase[freqpair])
                accs_freqpairs_phase[freqpair].append(acc)
                acc = accuracy_score(all_freqpairs_true[freqpair], all_freqpairs_pred_freq[freqpair])
                accs_freqpairs_freq[freqpair].append(acc)
            model_ensembles_freq.append(model_ensemble_freq)
            model_ensembles_pattern.append(model_ensemble_pattern)
            model_ensembles_freqpair.append(model_ensemble_freqpair)
            # Validation
            y_freqs, accs_freqs_allfeatures_val = helpers.get_scores(all_freqs_true, all_freqs_freq_pred_allfeatures, metric='accuracy')
            y_freqs, accs_freqs_magnitude_val = helpers.get_scores(all_freqs_true, all_freqs_freq_pred_magnitude, metric='accuracy')
            y_freqs, accs_freqs_phase_val = helpers.get_scores(all_freqs_true, all_freqs_freq_pred_phase, metric='accuracy')
            y_freqs, accs_freqs_freq_val = helpers.get_scores(all_freqs_true, all_freqs_freq_pred_freq, metric='accuracy')
            y_pattern, accs_pattern_allfeatures_val = helpers.get_scores(all_patterns_freq_true, all_patterns_freq_pred_allfeatures, metric='balanced_accuracy')
            y_pattern, accs_pattern_magnitude_val = helpers.get_scores(all_patterns_freq_true, all_patterns_freq_pred_magnitude, metric='balanced_accuracy')
            y_pattern, accs_pattern_phase_val = helpers.get_scores(all_patterns_freq_true, all_patterns_freq_pred_phase, metric='balanced_accuracy')
            y_pattern, accs_pattern_freq_val = helpers.get_scores(all_patterns_freq_true, all_patterns_freq_pred_freq, metric='balanced_accuracy') 
            accs_freqs_allfeatures_all_val.append(accs_freqs_allfeatures_val)
            accs_freqs_magnitude_all_val.append(accs_freqs_magnitude_val)
            accs_freqs_phase_all_val.append(accs_freqs_phase_val)
            accs_freqs_frequency_all_val.append(accs_freqs_freq_val)
            accs_pattern_allfeatures_all_val.append(accs_pattern_allfeatures_val)
            accs_pattern_magnitude_all_val.append(accs_pattern_magnitude_val)
            accs_pattern_phase_all_val.append(accs_pattern_phase_val)
            accs_pattern_frequency_all_val.append(accs_pattern_freq_val)
        # Save models
        dump(model_ensembles_freq, result_dir.joinpath(f"model_ensembles_freq_{modality}.joblib"))
        dump(model_ensembles_pattern, result_dir.joinpath(f"model_ensembles_pattern_{modality}.joblib"))
        dump(model_ensembles_freqpair, result_dir.joinpath(f"model_ensembles_freqpair_{modality}.joblib"))
        # Use validation results from last seed and plot
        # Get mean and std of accuracies for each feature
        accs_freqs_mean = np.mean(accs_freqs_allfeatures_all_val, axis=0)
        accs_freqs_magnitude_mean = np.mean(accs_freqs_magnitude_all_val, axis=0)
        accs_freqs_phase_mean = np.mean(accs_freqs_phase_all_val, axis=0)
        accs_freqs_freq_mean = np.mean(accs_freqs_frequency_all_val, axis=0)
        accs_freqs_std = np.std(accs_freqs_allfeatures_all_val, axis=0)
        accs_freqs_magnitude_std = np.std(accs_freqs_magnitude_all_val, axis=0)
        accs_freqs_phase_std = np.std(accs_freqs_phase_all_val, axis=0)
        accs_freqs_freq_std = np.std(accs_freqs_frequency_all_val, axis=0)
        accs_pattern_mean = np.mean(accs_pattern_allfeatures_all_val, axis=0)
        accs_pattern_magnitude_mean = np.mean(accs_pattern_magnitude_all_val, axis=0)
        accs_pattern_phase_mean = np.mean(accs_pattern_phase_all_val, axis=0)
        accs_pattern_freq_mean = np.mean(accs_pattern_frequency_all_val, axis=0)
        accs_patterns_std = np.std(accs_pattern_allfeatures_all_val, axis=0)
        accs_pattern_magnitude_std = np.std(accs_pattern_magnitude_all_val, axis=0)
        accs_pattern_phase_std = np.std(accs_pattern_phase_all_val, axis=0)
        accs_pattern_freq_std = np.std(accs_pattern_frequency_all_val, axis=0)
        print(accs_freqs_allfeatures_all_val, accs_pattern_allfeatures_all_val)
        print("Validation scores")
        np.save(result_dir.joinpath(f"accs_freqs_allfeatures_all_val_{modality}.npy"), accs_freqs_allfeatures_all_val)
        np.save(result_dir.joinpath(f"accs_freqs_magnitude_all_val_{modality}.npy"), accs_freqs_magnitude_all_val)
        np.save(result_dir.joinpath(f"accs_freqs_phase_all_val_{modality}.npy"), accs_freqs_phase_all_val)
        np.save(result_dir.joinpath(f"accs_freqs_frequency_all_val_{modality}.npy"), accs_freqs_frequency_all_val)
        np.save(result_dir.joinpath(f"accs_pattern_allfeatures_all_val_{modality}.npy"), accs_pattern_allfeatures_all_val)
        np.save(result_dir.joinpath(f"accs_pattern_magnitude_all_val_{modality}.npy"), accs_pattern_magnitude_all_val)
        np.save(result_dir.joinpath(f"accs_pattern_phase_all_val_{modality}.npy"), accs_pattern_phase_all_val)
        np.save(result_dir.joinpath(f"accs_pattern_frequency_all_val_{modality}.npy"), accs_pattern_frequency_all_val)
        np.save(result_dir.joinpath(f"all_patterns_freq_true_{modality}.npy"), all_patterns_freq_true)
        for freqpair, accs in accs_freqpairs_allfeatures.items():
            np.save(result_dir.joinpath(f"accs_freqpairs_allfeatures_val_{freqpair}_{modality}.npy"), accs_freqpairs_allfeatures[freqpair])
            print(accs_freqpairs_allfeatures[freqpair])
        with open(result_dir.joinpath(f'all_patterns_freq_true.json'), 'w', encoding='utf-8') as f:
            json.dump(all_patterns_freq_true, f, ensure_ascii=False, indent=4)

def test_pid(gases_allcombos, kinds, freqs, freqpairs, pid_index, experiment_dir, result_dir_parent, run_params):
    """
    Test classifiers to distinguish between modulation frequencies and patterns (corr vs acorr)

    :param gases_allcombos: list of gas combinations to train on
    :param kinds: list of kinds to train on
    :param freqs: list of modulation frequencies
    :param freqpairs: dictionary of frequency pairs
    :param pid_index: index of pid trials
    :param experiment_dir: directory of experiment data
    :param result_dir_parent: directory to save results
    :param run_params: dictionary of run parameters
    :return: None
    """
    # Testing
    print("Test")
    modality = 'pid'
    ms_start = run_params['ms_start']
    ms_end = run_params['ms_end']
    stimulus_start = run_params['stimulus_start']
    stimulus_duration = run_params['stimulus_duration']
    buffer = run_params['buffer']
    # Iterate over all gases
    for gases in gases_allcombos:
        gas_str = '_'.join(gases)
        print(gas_str)
        # Make subdirectory 
        result_dir = result_dir_parent.joinpath(gas_str)
        result_dir.mkdir(exist_ok=True, parents=True)
        # Select index
        index_test_selected = pid_index.query(f"(kind == '{kinds[0]}' | kind == '{kinds[1]}') & ((gas1 == '{gases[0]}' & gas2 == '{gases[1]}') | (gas1 == '{gases[1]}' & gas2 == '{gases[0]}'))")
        index_test_selected = index_test_selected.query(f"shape != '1Hz'")
        index_test_selected_idx = index_test_selected.index.to_numpy()
        accs_freqs_allfeatures_test_all = []
        accs_pattern_allfeatures_test_all = []
        accs_freqpairs_freqs_test_all = {freqpair: [] for freqpair in freqpairs.keys()}
        model_ensembles_freq = load(result_dir.joinpath(f"model_ensembles_freq_{modality}.joblib"))
        model_ensembles_pattern = load(result_dir.joinpath(f"model_ensembles_pattern_{modality}.joblib"))
        model_ensembles_freqpair = load(result_dir.joinpath(f"model_ensembles_freqpair_{modality}.joblib"))
        # Iterate over all simulations (we have trained models for each random seed)
        for simulation_idx in tqdm(range(len(model_ensembles_pattern))):
            print(simulation_idx)
            # Get features
            y_test_freqs, y_test_patterns, X_test_freqs, X_test_magnitudes, X_test_phases, X_test_allfeatures = helpers.get_freqspace_features_pid(index_test_selected, experiment_dir.joinpath('pid'), modality, ms_start, ms_end, stimulus_start, stimulus_duration+buffer, set='test')
            # Initialise score collections
            all_freqs_true = {freq: [] for freq in freqs}
            all_freqs_freq_pred_allfeatures = {freq: [] for freq in freqs}
            all_patterns_freq_true = {freq: [] for freq in freqs}
            all_patterns_freq_pred_allfeatures = {freq: [] for freq in freqs}
            all_freqpairs_true_test = {freqpair: [] for freqpair in freqpairs.keys()}
            all_freqpairs_pred_test = {freqpair: [] for freqpair in freqpairs.keys()}
            # Predict frequency from pair (test frequency pair classifier (on test data))
            models_ensemble_freqpair = model_ensembles_freqpair[simulation_idx]
            for pair in freqpairs.keys():
                y_test_pair = []
                X_test_pair = []
                for y_test, X_test in zip(y_test_freqs, X_test_allfeatures):
                    if y_test not in freqpairs[pair]:
                        continue
                    else:
                        y_test_pair.append(y_test)
                        X_test_pair.append(X_test)
                if len(y_test_pair)==0:
                    continue
                y_pred_pair_ensemble = []
                for models_freqpair in models_ensemble_freqpair:
                    model_freqpair = models_freqpair[pair]
                    y_pred_pair = list(model_freqpair.predict(X_test_pair))
                    y_pred_pair_ensemble.append(y_pred_pair)
                y_pred_pair_winner = helpers.get_winner(y_pred_pair_ensemble, axis=1)
                all_freqpairs_true_test[pair] += y_test_pair
                all_freqpairs_pred_test[pair] += y_pred_pair_winner#y_pred_pair
            # Get accuracy for each frequency pair
            for freqpair in all_freqpairs_true_test.keys():
                acc = accuracy_score(all_freqpairs_true_test[freqpair], all_freqpairs_pred_test[freqpair])
                accs_freqpairs_freqs_test_all[freqpair].append(acc)
            # Predict Freqs & Patterns
            for f, x_freqs, x_allfeatures, y in zip(y_test_freqs, X_test_freqs, X_test_allfeatures, y_test_patterns):
                # print(f)
                # Predict Freqs
                ensemble_predictions_freq = []
                for model_freq in model_ensembles_freq[simulation_idx]:
                    pred = model_freq.predict(x_allfeatures.reshape(1, -1))
                    ensemble_predictions_freq.append(pred)
                ensemble_predictions_freq = np.array(ensemble_predictions_freq)
                pred = helpers.get_winner(ensemble_predictions_freq)
                all_freqs_true[f].append(f)
                all_freqs_freq_pred_allfeatures[f].append(pred.reshape(1, -1))
                # Predict Patterns
                ensemble_predictions_pattern = []
                for model_pattern in model_ensembles_pattern[simulation_idx]:
                    pred = model_pattern.predict(x_allfeatures.reshape(1, -1))
                    ensemble_predictions_pattern.append(pred)
                ensemble_predictions_pattern = np.array(ensemble_predictions_pattern)
                pred = helpers.get_winner(ensemble_predictions_pattern)
                all_patterns_freq_true[f].append(y)
                all_patterns_freq_pred_allfeatures[f].append(pred.reshape(1, -1))
            for f in all_patterns_freq_pred_allfeatures.keys():
                all_patterns_freq_true[f] = np.array(all_patterns_freq_true[f]).flatten()
                all_freqs_freq_pred_allfeatures[f] = np.array(all_freqs_freq_pred_allfeatures[f]).flatten()
                all_patterns_freq_pred_allfeatures[f] = np.array(all_patterns_freq_pred_allfeatures[f]).flatten()
            y_freqs, accs_freqs_allfeatures_test = helpers.get_scores(all_freqs_true, all_freqs_freq_pred_allfeatures, metric='accuracy')
            y_freqs, accs_pattern_allfeatures_test = helpers.get_scores(all_patterns_freq_true, all_patterns_freq_pred_allfeatures, metric='balanced_accuracy')
            accs_freqs_allfeatures_test_all.append(accs_freqs_allfeatures_test)
            accs_pattern_allfeatures_test_all.append(accs_pattern_allfeatures_test)
        # SAVE
        np.save(result_dir.joinpath(f"accs_freqs_allfeatures_test_all_{gas_str}_{modality}.npy"), accs_freqs_allfeatures_test_all)
        np.save(result_dir.joinpath(f"accs_pattern_allfeatures_test_all_{gas_str}_{modality}.npy"), accs_pattern_allfeatures_test_all)
        for freqpair, accs in accs_freqpairs_freqs_test_all.items():
            np.save(result_dir.joinpath(f"accs_freqpairs_freqs_test_all_{gas_str}_{freqpair}_{modality}.npy"), accs_freqpairs_freqs_test_all[freqpair])
        for freq, freq_patterns in all_patterns_freq_true.items():
            all_patterns_freq_true[freq] = freq_patterns.tolist()
        with open(result_dir.joinpath(f'all_patterns_freq_true_{gas_str}_{modality}.json'), 'w', encoding='utf-8') as f:
            json.dump(all_patterns_freq_true, f, ensure_ascii=False, indent=4)
        accs_freqs_allfeatures_test_all = np.load(result_dir.joinpath(f"accs_freqs_allfeatures_test_all_{gas_str}_{modality}.npy"))
        accs_pattern_allfeatures_test_all = np.load(result_dir.joinpath(f"accs_pattern_allfeatures_test_all_{gas_str}_{modality}.npy"))