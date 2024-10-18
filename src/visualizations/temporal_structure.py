from pathlib import Path
import json
import scipy.io

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
new_rc_params = {"text.usetex": False, "svg.fonttype": "none"}
mpl.rcParams.update(new_rc_params)
cmap = mpl.cm.get_cmap('viridis')
import matplotlib.patches as patches
from scipy.fft import fftfreq
from scipy import signal
import pandas as pd
import importlib

parent_package = '.'.join(__package__.split('.')[:-1])
constants = importlib.import_module(f'{parent_package}.utils.constants')
from src.utils.constants import pattern_spelledout, sensor_names_short, sensor_colors, heater_colors
from src.temporal_structure.helpers import trial_nose, cut, scale, fft_nose, get_trial_df, ffs_to_peaks_allsensors

def plot_traintest(index_train, index_test, result_dir_parent):
    """ 
    Plot the training and testing trials

    :param index_train: pd.DataFrame
    :param index_test: pd.DataFrame
    :param result_dir_parent: Path    
    """
    # Convert to hours
    index_train['t_stimulus'] = (pd.to_timedelta(index_train['t_stimulus'])).dt.total_seconds()/60./60.
    index_test['t_stimulus'] = pd.to_timedelta(index_test['t_stimulus']).dt.total_seconds()/60./60.
    # Plot 
    fig, ax = plt.subplots(figsize=(4,2))
    min_train, max_train = min(index_train['t_stimulus']), max(index_train['t_stimulus']), 
    min_test, max_test = min(index_test['t_stimulus']), max(index_test['t_stimulus'])
    ax.set_xlim([min(min_train, min_test)-0.2, max(max_train, max_test)+0.1])
    rect_train = patches.Rectangle((min_train, 1.5+0.1), max_train-min_train, 1, edgecolor='none', facecolor='green')#, label='train / val')
    rect_test = patches.Rectangle((min_test, 1.5+0.1), max_test-min_test, 1, edgecolor='none', facecolor='red')#, label='test')
    ax.add_patch(rect_train)
    ax.add_patch(rect_test)
    rect_unused_1 = patches.Rectangle((min_train, 0.5), max_train-min_train, 1, edgecolor='none', facecolor='gray')#, label='train / val')
    rect_unused_2 = patches.Rectangle((min_test, 0.5), max_test-min_test, 1, edgecolor='none', facecolor='gray')#, label='test')
    ax.add_patch(rect_unused_1)
    ax.add_patch(rect_unused_2)
    # Add text
    ax.text(x=min_train + 0.5*(max_train-min_train), y=2+0.05, s=f"Training / Validation", ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x=min_test + 0.5*(max_test-min_test), y=2+0.05, s=f"Testing", ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x=min_train + 0.5*(max_train-min_train), y=1, s=f"unused", ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x=min_test + 0.5*(max_test-min_test), y=1, s=f"unused", ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x=min_train-0.3, y=2+0.05, s=f"Sensor 1-4", ha='right', va='center', fontsize=9)
    ax.text(x=min_train-0.3, y=1, s=f"Sensor 5-8", ha='right', va='center', fontsize=9)
    ax.set_yticks([])
    ax.set_ylim([0,3])
    ax.set_xlabel('Time (h)')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    plt.savefig(result_dir_parent.joinpath(f'trials_trainvaltest.svg'))#, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_corracorr_val_features(result_dir_parent, gases, features, modality):
    """ 
    Plot correlated vs anticorrelated for each feature (validation)

    :param result_dir_parent: Path
    :param gases: list
    :param features: list
    :param modality: str    
    """
    colors = ['#66c2a5','#fc8d62','#8da0cb', '#ff0000']
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, feature in enumerate(features):
        filename = f"accs_pattern_{feature}_all_val_{modality}.npy"
        gas_str = '_'.join(gases)
        result_dir = result_dir_parent.joinpath(gas_str)
        try:
            with open(result_dir.joinpath('all_patterns_freq_true.json'), 'r', encoding='utf-8') as f:
                all_patterns_freq_true = json.load(f)
                all_stimulus_freqs = [int(freq) for freq in all_patterns_freq_true.keys()]
        except FileNotFoundError:
            pass
        accs_pattern = np.load(result_dir.joinpath(filename))
        accs_pattern_mean = np.mean(accs_pattern, axis=0)
        accs_pattern_std = np.std(accs_pattern, axis=0)
        accs_pattern_median = np.median(accs_pattern, axis=0)
        accs_pattern_q25 = np.quantile(accs_pattern, 0.25, axis=0)
        accs_pattern_q75 = np.quantile(accs_pattern, 0.75, axis=0)
        color = colors[i]
        ax.scatter(all_stimulus_freqs, accs_pattern_mean, color=color, linewidth=2, zorder=10)
        ax.plot(all_stimulus_freqs, accs_pattern_mean, color=color, linewidth=2, zorder=8, label=feature)
        ax.fill_between(all_stimulus_freqs, np.clip(accs_pattern_mean-accs_pattern_std, 0, 1), np.clip(accs_pattern_mean+accs_pattern_std, 0, 1), color=color, alpha=0.1)
    # Formatting
    ax.axhline(0.5, c='gray', linestyle='--')
    ax.axhline(1.0, c='gray', linestyle='--')
    ax.set_xscale('log')
    ax.set_xticks(all_stimulus_freqs)
    ax.set_xticklabels(all_stimulus_freqs)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Modulation Frequency (Hz)")
    ax.set_xlim([1.8,100])
    ax.set_ylim([0,1.05])
    ax.text(2, 0.52, "Random chance", color='gray')
    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(result_dir_parent.joinpath(f"classify_corracorr_val_feature_{modality}.svg"))
    fig.savefig(result_dir_parent.joinpath(f"classify_corracorr_val_feature_{modality}.png"), dpi=300)
    plt.show()

def plot_corracorr_comparison(result_dir_parent, gases_allcombos, modality, filename):
    """ 
    Plot correlated vs anticorrelated for each feature, comparing all gases
    
    :param result_dir_parent: Path
    :param gases_allcombos: list
    :param modality: str
    :param filename: str
    """
    all_accs = []
    fig, ax = plt.subplots(figsize=(4,2.5))
    for i, gases in enumerate(gases_allcombos):
        gas_str = '_'.join(gases)
        result_dir = result_dir_parent.joinpath(gas_str)
        try:
            with open(result_dir.joinpath('all_patterns_freq_true.json'), 'r', encoding='utf-8') as f:
                all_patterns_freq_true = json.load(f)
                all_stimulus_freqs = [int(freq) for freq in all_patterns_freq_true.keys()]
        except FileNotFoundError:
            pass
        accs_pattern = np.load(result_dir.joinpath(filename))
        accs_pattern_mean = np.mean(accs_pattern, axis=0)
        accs_pattern_std = np.std(accs_pattern, axis=0)
        accs_pattern_median = np.median(accs_pattern, axis=0)
        accs_pattern_q25 = np.quantile(accs_pattern, 0.25, axis=0)
        accs_pattern_q75 = np.quantile(accs_pattern, 0.75, axis=0)
        all_accs.append(accs_pattern)
        color = cmap(i/len(gases_allcombos))
        ax.scatter(all_stimulus_freqs, accs_pattern_mean, color=color, linewidth=1.5, zorder=10, s=10)
        ax.plot(all_stimulus_freqs, accs_pattern_mean, color=color, linewidth=1.5, zorder=8, label=gas_str)
        ax.fill_between(all_stimulus_freqs, np.clip(accs_pattern_mean-accs_pattern_std, 0, 1), np.clip(accs_pattern_mean+accs_pattern_std, 0, 1), color=color, alpha=0.1)
    all_accs = np.array(all_accs)
    all_accs = all_accs.reshape(all_accs.shape[0]*all_accs.shape[1], all_accs.shape[2])
    all_accs_mean = np.mean(all_accs, axis=0)
    all_accs_std = np.std(all_accs, axis=0)
    all_accs_median = np.median(all_accs, axis=0)
    all_accs_q25 = np.quantile(all_accs, 0.25, axis=0)
    all_accs_q75 = np.quantile(all_accs, 0.75, axis=0)
    ax.scatter(all_stimulus_freqs, all_accs_mean, color='r', linewidth=2, zorder=10, s=20)
    ax.plot(all_stimulus_freqs, all_accs_mean, color='r', linewidth=2, zorder=8, label='mean')
    ax.fill_between(all_stimulus_freqs, np.clip(all_accs_mean-all_accs_std, 0, 1), np.clip(all_accs_mean+all_accs_std, 0, 1), color='r', alpha=0.2)
    print("ACORR MEAN: ", all_accs_mean)
    print("ACORR STD: ", all_accs_std)
    # Formatting
    ax.axhline(0.5, c='gray', linestyle='--')#, label='random')
    ax.axhline(1.0, c='gray', linestyle='--')#, label='random')
    ax.set_xscale('log')
    ax.set_xticks(all_stimulus_freqs)
    ax.set_xticklabels(all_stimulus_freqs)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Modulation Frequency (Hz)")
    ax.set_xlim([1.8,100])
    # ax.set_ylim([0,1.05])
    ax.text(2, 0.52, "Random chance", color='gray')
    ax.spines[['right', 'top']].set_visible(False)
    fig.legend()
    fig.savefig(result_dir_parent.joinpath(f"classify_corracorr_test_comparison_{modality}.svg"))
    fig.savefig(result_dir_parent.joinpath(f"classify_corracorr_test_comparison_{modality}.png"), dpi=300)
    plt.show()

def plot_freq_val_features(result_dir_parent, gases, features, modality):
    """ 
    Plot frequency for each feature (validation)

    :param result_dir_parent: Path
    :param gases: list
    :param features: list
    :param modality: str    
    """
    colors = ['#66c2a5','#fc8d62','#8da0cb', '#ff0000']
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, feature in enumerate(features):
        filename = f"accs_freqs_{feature}_all_val_{modality}.npy"
        gas_str = '_'.join(gases)
        result_dir = result_dir_parent.joinpath(gas_str)        
        try:
            with open(result_dir.joinpath('all_patterns_freq_true.json'), 'r', encoding='utf-8') as f:
                all_patterns_freq_true = json.load(f)
                all_stimulus_freqs = [int(freq) for freq in all_patterns_freq_true.keys()]
        except FileNotFoundError:
            pass
        accs_freqs = np.load(result_dir.joinpath(filename))
        accs_freqs_mean = np.mean(accs_freqs, axis=0)
        accs_freqs_std = np.std(accs_freqs, axis=0) 
        accs_freqs_median = np.median(accs_freqs, axis=0)
        accs_freqs_q25 = np.quantile(accs_freqs, 0.25, axis=0)
        accs_freqs_q75 = np.quantile(accs_freqs, 0.75, axis=0)  
        # E-nose
        color = colors[i]
        ax.scatter(all_stimulus_freqs, accs_freqs_mean, color=color, linewidth=2, zorder=10)
        ax.plot(all_stimulus_freqs, accs_freqs_mean, color=color, linewidth=2, zorder=8, label=gas_str)
        ax.fill_between(all_stimulus_freqs, np.clip(accs_freqs_mean-accs_freqs_std, 0, 1), np.clip(accs_freqs_mean+accs_freqs_std, 0, 1), color=color, alpha=0.1)
    # Formatting
    ax.axhline(1./6, c='gray', linestyle='--')#, label='random')
    ax.axhline(1.0, c='gray', linestyle='--')#, label='random')
    # ax.set_ylim([0.45, 1.02])
    ax.set_xscale('log')
    ax.set_xticks(all_stimulus_freqs)
    ax.set_xticklabels(all_stimulus_freqs)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Modulation Frequency (Hz)")
    ax.set_ylim([0,1.05])
    ax.text(2, 1./6 + 0.02, "Random chance", color='gray')
    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(result_dir_parent.joinpath(f"classify_freq_val_feature_{modality}.svg"))
    fig.savefig(result_dir_parent.joinpath(f"classify_freq_val_feature_{modality}.png"), dpi=300)
    fig.show()
    plt.show()

def plot_freq_comparison(result_dir_parent, gases_allcombos, modality, filename):
    """
    Plot frequency for each feature, comparing all gases

    :param result_dir_parent: Path
    :param gases_allcombos: list
    :param modality: str
    :param filename: str
    """
    fig, ax = plt.subplots(figsize=(4,2.5))
    all_accs = []
    for i, gases in enumerate(gases_allcombos):
        gas_str = '_'.join(gases)
        result_dir = result_dir_parent.joinpath(gas_str)
        try:
            with open(result_dir.joinpath('all_patterns_freq_true.json'), 'r', encoding='utf-8') as f:
                all_patterns_freq_true = json.load(f)
                all_stimulus_freqs = [int(freq) for freq in all_patterns_freq_true.keys()]
        except FileNotFoundError:
            pass
        accs_freqs = np.load(result_dir.joinpath(filename))
        accs_freqs_mean = np.mean(accs_freqs, axis=0)
        accs_freqs_std = np.std(accs_freqs, axis=0)    
        accs_freqs_q25 = np.quantile(accs_freqs, 0.25, axis=0)
        accs_freqs_q75 = np.quantile(accs_freqs, 0.75, axis=0)
        accs_freqs_median = np.median(accs_freqs, axis=0)
        all_accs.append(accs_freqs)
        # E-nose
        color = cmap(i/len(gases_allcombos))
        ax.scatter(all_stimulus_freqs, accs_freqs_mean, color=color, linewidth=1.5, zorder=10, s=10)
        ax.plot(all_stimulus_freqs, accs_freqs_mean, color=color, linewidth=1.5, zorder=8, label=gas_str)
        ax.fill_between(all_stimulus_freqs, np.clip(accs_freqs_mean-accs_freqs_std, 0, 1), np.clip(accs_freqs_mean+accs_freqs_std, 0, 1), color=color, alpha=0.1) # Clipping at 1
    # Plot summary in RED
    all_accs = np.array(all_accs)
    print(all_accs.shape)
    print(all_accs[0].shape, all_accs[1].shape, all_accs[2].shape, all_accs[3].shape, all_accs[4].shape, all_accs[5].shape)
    print(all_accs[0])
    all_accs = all_accs.reshape(all_accs.shape[0]*all_accs.shape[1], all_accs.shape[2])
    all_accs_mean = np.mean(all_accs, axis=0)
    all_accs_std = np.std(all_accs, axis=0)
    all_accs_q25 = np.quantile(all_accs, 0.25, axis=0)
    all_accs_q75 = np.quantile(all_accs, 0.75, axis=0)
    all_accs_median = np.median(all_accs, axis=0)
    ax.scatter(all_stimulus_freqs, all_accs_mean, color='r', linewidth=2, zorder=10, s=25)
    ax.plot(all_stimulus_freqs, all_accs_mean, color='r', linewidth=2, zorder=8, label='All')
    ax.fill_between(all_stimulus_freqs, np.clip(all_accs_mean-all_accs_std, 0, 1), np.clip(all_accs_mean+all_accs_std, 0, 1), color='r', alpha=0.1)
    print("FREQS MEAN: ", all_accs_mean)
    print("FREQS STD: ", all_accs_std)
    # Formatting
    ax.axhline(1./6, c='gray', linestyle='--')#, label='random')
    ax.axhline(1.0, c='gray', linestyle='--')#, label='random')
    ax.set_xscale('log')
    ax.set_xticks(all_stimulus_freqs)
    ax.set_xticklabels(all_stimulus_freqs)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Modulation Frequency (Hz)")
    ax.set_ylim([0,1.05])
    ax.text(2, 1./6 + 0.02, "Random chance", color='gray')
    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(result_dir_parent.joinpath(f"classify_freq_test_comparison_{modality}.svg"))
    fig.savefig(result_dir_parent.joinpath(f"classify_freq_test_comparison_{modality}.png"), dpi=300)
    # fig.show()
    plt.show()
    plt.close()

def plot_freqpairs_val_features(result_dir_parent, gases, features, modality):
    """ 
    Plot frequency pairs for each feature (validation)

    :param result_dir_parent: Path
    :param gases: list
    :param features: list
    :param modality: str
    """
    enose_freq_ticks = [0, 2, 5, 6, 7] # 20 Hz vs 2, 5, 10, 40, 60 Hz 
    colors = ['#66c2a5','#fc8d62','#8da0cb', '#ff0000']
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, feature in enumerate(features):
        gas_str = '_'.join(gases)
        result_dir = result_dir_parent.joinpath(gas_str)
        accs_freqpairs = {}
        freqpairs = {'2 vs 20 Hz': [2, 20,], '5 vs 20 Hz': [5, 20,], '10 vs 20 Hz': [10, 20,], '40 vs 20 Hz': [40, 20,], '60 vs 20 Hz': [60, 20]}
        for freqpair in freqpairs.keys():
            path = result_dir.joinpath(f"accs_freqpairs_{feature}_val_{freqpair}_{modality}.npy")
            accs_freqpairs[freqpair] = np.load(path)
        color = colors[i]
        freq_ticks, accs_mean, accs_std = [], [], []
        accs_median, accs_q25, accs_q75 = [], [], []
        for j, freqpair in enumerate(freqpairs.keys()):
            acc_mean = np.mean(accs_freqpairs[freqpair])
            acc_std = np.std(accs_freqpairs[freqpair])
            accs_mean.append(acc_mean)
            accs_std.append(acc_std)
            accs_median.append(np.median(accs_freqpairs[freqpair]))
            accs_q25.append(np.quantile(accs_freqpairs[freqpair], 0.25))
            accs_q75.append(np.quantile(accs_freqpairs[freqpair], 0.75))
            freq_ticks.append(enose_freq_ticks[j])
        freq_ticks, accs_mean, accs_std = np.array(freq_ticks), np.array(accs_mean), np.array(accs_std)
        accs_median, accs_q25, accs_q75 = np.array(accs_median), np.array(accs_q25), np.array(accs_q75)
        ax.scatter(freq_ticks, accs_mean, color=color, linewidth=2, zorder=8, label=gas_str)
        yerrs_clipped = [-1*(np.clip(accs_mean-accs_std, 0, 1)-accs_mean), np.clip(accs_mean+accs_std, 0, 1)-accs_mean]
        ax.errorbar(freq_ticks, accs_mean, yerr=yerrs_clipped, color=color, zorder=11, fmt='none')
    ax.axhline(0.5, c='gray', linestyle='--')
    ax.axhline(1.0, c='gray', linestyle='--')
    ax.set_xticks(enose_freq_ticks, freqpairs.keys(), rotation='90')
    ax.set_ylabel("Accuracy")
    ax.text(0, 0.52, "Random chance", color='gray')
    ax.set_ylim([0,1.05])
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(result_dir_parent.joinpath(f"classify_freqpair_val_feature_{modality}.svg"))
    fig.savefig(result_dir_parent.joinpath(f"classify_freqpair_val_feature_{modality}.png"), dpi=300)
    fig.show()
    plt.show()

def plot_freqpairs_comparison(result_dir_parent, gases_allcombos, modality, filename):
    """
    Plot frequency pairs for each feature, comparing all gases

    :param result_dir_parent: Path
    :param gases_allcombos: list
    :param modality: str
    :param filename: str
    """
    enose_freq_ticks = [0, 2, 5, 6, 7] # 20 Hz vs 2, 5, 10, 40, 60 Hz 
    all_accs = []
    fig, ax = plt.subplots(figsize=(4,2.5))
    for i, gases in enumerate(gases_allcombos):
        gas_str = '_'.join(gases)
        result_dir = result_dir_parent.joinpath(gas_str)
        all_accs_gas = []
        accs_freqpairs = {}
        freqpairs = {'2 vs 20 Hz': [2, 20,], '5 vs 20 Hz': [5, 20,], '10 vs 20 Hz': [10, 20,], '40 vs 20 Hz': [40, 20,], '60 vs 20 Hz': [60, 20]}
        for freqpair in freqpairs.keys():
            path = result_dir.joinpath(filename+f"{freqpair}_{modality}.npy")
            accs_freqpairs[freqpair] = np.load(path)
            all_accs_gas.append(accs_freqpairs[freqpair])
        color = cmap(i/len(gases_allcombos))
        freq_ticks, accs_mean, accs_std = [], [], []
        accs_median, accs_q25, accs_q75 = [], [], []
        for j, freqpair in enumerate(freqpairs.keys()):
            acc_mean = np.mean(accs_freqpairs[freqpair])
            acc_std = np.std(accs_freqpairs[freqpair])
            accs_mean.append(acc_mean)
            accs_std.append(acc_std)
            accs_median.append(np.median(accs_freqpairs[freqpair]))
            accs_q25.append(np.quantile(accs_freqpairs[freqpair], 0.25))
            accs_q75.append(np.quantile(accs_freqpairs[freqpair], 0.75))
            freq_ticks.append(enose_freq_ticks[j])
        all_accs.append(all_accs_gas)
        freq_ticks, accs_mean, accs_std = np.array(freq_ticks), np.array(accs_mean), np.array(accs_std), 
        accs_median, accs_q25, accs_q75 = np.array(accs_median), np.array(accs_q25), np.array(accs_q75)
        ax.scatter(freq_ticks, accs_mean, color=color, linewidth=2, zorder=8, label=gas_str, s=10)
        yerrs_clipped = [-1*(np.clip(accs_mean-accs_std, 0, 1)-accs_mean), np.clip(accs_mean+accs_std, 0, 1)-accs_mean]
        ax.errorbar(freq_ticks, accs_mean, yerr=yerrs_clipped, color='k', zorder=11, fmt='none')
    all_accs = np.array(all_accs)
    all_accs = np.swapaxes(all_accs, 0, 1)
    all_accs = all_accs.reshape(all_accs.shape[0], all_accs.shape[1]*all_accs.shape[2])
    all_accs_mean = np.mean(all_accs, axis=1)
    all_accs_std = np.std(all_accs, axis=1)
    all_accs_median = np.median(all_accs, axis=1)
    all_accs_q25 = np.quantile(all_accs, 0.25, axis=1)
    all_accs_q75 = np.quantile(all_accs, 0.75, axis=1)
    ax.scatter(freq_ticks, all_accs_mean, color='r', linewidth=2, zorder=8, label='All', s=20)
    yerrs_clipped = [-1*(np.clip(all_accs_mean-all_accs_std, 0, 1)-all_accs_mean), np.clip(all_accs_mean+all_accs_std, 0, 1)-all_accs_mean]
    ax.errorbar(freq_ticks, all_accs_mean, yerr=yerrs_clipped, color='r', zorder=11, fmt='none')
    print("FREQPAIRS MEAN: ", all_accs_mean)
    print("FREQPAIRS STD: ", all_accs_std)    
    ax.axhline(0.5, c='gray', linestyle='--')
    ax.axhline(1.0, c='gray', linestyle='--')
    ax.set_xticks(enose_freq_ticks, freqpairs.keys(), rotation='90')
    ax.set_ylabel("Accuracy")
    ax.text(0, 0.52, "Random chance", color='gray')
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(result_dir_parent.joinpath(f"classify_freqpair_test_comparison_{modality}.svg"))
    fig.savefig(result_dir_parent.joinpath(f"classify_freqpair_test_comparison_{modality}.png"), dpi=300)
    fig.show()
    plt.show()

def plot_raw(t_all, nose_data_all, result_dir, params):
    """ 
    Plot raw sensor data

    :param t_all: list
    :param nose_data_all: list
    :param result_dir: Path
    :param params: dict
    """
    fig, ax = plt.subplots(nrows=len(nose_data_all), figsize=(3, 2*len(nose_data_all)), sharex=True, sharey=True) 
    for i, (t, nose_data) in enumerate(zip(t_all, nose_data_all)):
        for s, nose in enumerate(nose_data):
            ax[i].plot(t, nose, label=f'{sensor_names_short[s]}', color=sensor_colors[s], zorder=1)
        ax[i].axvline(0, c='gray', ls=':')
        ax[i].axvline(1000, c='gray', ls=':')
        ax[i].set_yscale("log")
        # # Hide the right and top spines
        plt.setp(ax[i].spines.values(), linewidth=1.)
        ax[i].spines[['right', 'top']].set_visible(False)
    ax[-1].set_xlabel("Time (ms)")
    ax[0].legend(loc='upper right')
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Resistance (Ohm)")
    plt.savefig(result_dir.joinpath(f"raw_{params['modality']}.svg"))
    plt.savefig(result_dir.joinpath(f"raw_{params['modality']}.png"), dpi=300)
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_magnitudes(ffs_selected, result_dir, params):
    """
    Plot magnitudes of FFT

    :param ffs_selected: list
    :param result_dir: Path
    :param params: dict
    """
    fig, ax = plt.subplots(nrows=len(ffs_selected), figsize=(3, 2*len(ffs_selected)), sharex=True, sharey=True)
    for i, ffs in enumerate(ffs_selected):
        fhz = fftfreq(len(ffs[0]), 1e-3)
        N = len(ffs[0])  
        freqs_peaks_all, magnitudes_peaks_all, phases_peaks_all, x_all, y_all = ffs_to_peaks_allsensors(ffs)      
        for s, (freqs_peaks, magnitudes_peaks, x, y) in enumerate(zip(freqs_peaks_all, magnitudes_peaks_all, x_all, y_all)):
            ax[i].plot(x, np.abs(y), label=f'{sensor_names_short[s]}', color=sensor_colors[s], zorder=1)
            ax[i].scatter(freqs_peaks[0], magnitudes_peaks[0], color='r', zorder=2, s=25, marker='x')
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].set_xticks([1,2,5,10,20,40,60,100])
        ax[i].set_xticklabels([1,2,5,10,20,40,60,100])
        ax[i].grid(axis='x')
        # Hide the right and top spines
        plt.setp(ax[i].spines.values(), linewidth=1.)
        ax[i].spines[['right', 'top']].set_visible(False)
    ax[-1].set_xlabel("Frequency (Hz)")
    ax[0].legend(loc='upper right')
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel(r"Magnitude ($\log(\Omega)/s$)")
    plt.savefig(result_dir.joinpath(f"fft_magnitudes_{params['modality']}.svg"))
    plt.savefig(result_dir.joinpath(f"fft_magnitudes_{params['modality']}.png"), dpi=300)
    plt.show()
    plt.close()

def make_spider(features_dict, result_dir=None, modality='R_gas', n_highest = 1):
    """
    Make spider plot of features

    :param features_dict: dict
    :param result_dir: Path
    :param modality: str
    :param n_highest: int
    """
    colors = ['0', '#66c2a5','#fc8d62','#8da0cb']
    from math import pi
    # number of variable
    categories=sensor_names_short*n_highest#list(df)[1:]
    N = len(categories)
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]    # Adding first value to close circle (only works with lists!)
    feature_names = {'frequency': "Frequency (Hz)", 'magnitude': "Magnitude ($\log(\Omega)/s$)", 'phase': "Phase (rad)"}
    fig, ax = plt.subplots(len(features_dict), len(list(features_dict.values())[0]), subplot_kw={'projection': 'polar'}, figsize=(6,2*len(feature_names)), sharey='col')
    for row, (code, features) in enumerate(features_dict.items()):
        for col, (feature_name, feature) in enumerate(features.items()):
            feature_mean = list(np.mean(feature, axis=0))
            feature_mean += feature_mean[:1] # Adding first value to close circle (only works with lists!)
            ax[row, col].plot(angles, feature_mean, color='green', linewidth=2.5, linestyle='solid', zorder=5)
            for feature_mean in feature:
                feature_mean = list(feature_mean)
                feature_mean += feature_mean[:1]
                ax[row, col].plot(angles, feature_mean, color='k', linewidth=0.5, linestyle='solid', zorder=10)
                # Draw one axe per variable + add labels labels yet
                ax[row, col].set_xticks(angles[:-1], categories, color='grey', size=8)
                # If you want the first axis to be on top:
                ax[row, col].set_theta_offset(1*pi/4)
                ax[row, col].set_theta_direction(-1)
                ax[row, col].spines['polar'].set_visible(False)
                # Set appropriate tick labels
                if feature_name == 'frequency':
                    ax[row, col].set_yticks(1./np.array([5,10,20]))
                    yticks = ax[row, col].get_yticks()
                    ax[row, col].set_yticklabels((1./np.array(yticks)).astype(int))
                if feature_name == 'phase':
                    ax[row, col].set_yticks([-np.pi/2, 0, np.pi/2, np.pi])
                    ax[row, col].set_yticklabels([r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], ha='center', va='bottom')
                if row == 0:
                    ax[row, col].set_title(feature_names[feature_name])
    plt.suptitle("FFT Magnitudes", fontweight='bold')
    plt.tight_layout()
    plt.savefig(result_dir.joinpath(f"featureplots_{modality}.svg"))
    plt.savefig(result_dir.joinpath(f"featureplots_{modality}.png"), dpi=300)
    plt.show()
    plt.close()
    print(result_dir)

def plot_modulation(freqs, patterns=[None], result_dir=None):
    """  
    Plot modulation patterns valve commands

    :param freqs: list
    :param patterns: list
    :param result_dir: Path    
    """
    fs = 1000
    t = np.arange(1,1000)
    fig, ax = plt.subplots(nrows=len(freqs), figsize=(1,2*len(freqs)), sharex=True)
    for i, (freq, pattern) in enumerate(zip(freqs, patterns)):
        if pattern == 'corr':
            ax[i].plot(t, 1.2+0.5*(1+signal.square(2 * np.pi * freq * t/fs)), c='r', linewidth=0.5)
            ax[i].plot(t, +0.5*(1+signal.square(2 * np.pi * freq * t/fs)), c='b', linewidth=0.5)
        elif pattern == 'acor':
            ax[i].plot(t, 0.2+0.5*(1+signal.square(2 * np.pi * freq * t/fs)), c='r', linewidth=0.5)
            ax[i].plot(t, -0.5*(1+signal.square(2 * np.pi * freq * t/fs)), c='b', linewidth=0.5)
        # Hide the right and top spines
        plt.setp(ax[i].spines.values(), linewidth=1.)
        ax[i].spines[['left', 'right', 'top']].set_visible(False)
        ax[i].set_yticks([])
        ax[i].set_ylabel(f"{freq}Hz\n{pattern_spelledout[pattern]}", fontweight='bold')
    ax[-1].set_xlabel("Time (ms)")
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Valve Commands")
    plt.savefig(result_dir.joinpath("stimulus.svg"), bbox_inches='tight')
    plt.savefig(result_dir.joinpath("stimulus.png"), bbox_inches='tight', dpi=300)
    plt.show()

def plot_pid(pids_t, pids_data, result_dir=None):
    """ 
    Plot PID signals

    :param pids_t: list
    :param pids_data: list
    :param result_dir: Path    
    """
    fs = 1000
    t = np.arange(1,1000)
    fig, ax = plt.subplots(nrows=len(pids_t), figsize=(1,2*len(pids_t)), sharex=True, sharey=True)
    for i, (pid_t, pid_data) in enumerate(zip(pids_t, pids_data)):
        ax[i].plot(pid_t, pid_data, c='k')
        # Hide the right and top spines
        plt.setp(ax[i].spines.values(), linewidth=1.)
        ax[i].spines[['left', 'right', 'top']].set_visible(False)
        ax[i].set_yticks([])
    ax[-1].set_xlabel("Time (ms)") 
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("PID (a.u.)")
    plt.savefig(result_dir.joinpath("stimulus_pid.svg"), bbox_inches='tight')
    plt.savefig(result_dir.joinpath("stimulus_pid.png"), bbox_inches='tight', dpi=300)
    plt.show()

def plot_freq_IAEB(result_dir_parent, gases, modality, filename):
    """ 
    Plot frequency classification task

    :param result_dir_parent: Path
    :param gases: list
    :param modality: str
    :param filename: str
    """
    fig, ax = plt.subplots(figsize=(4,2.5))
    gas_str = '_'.join(gases)
    result_dir = result_dir_parent.joinpath(gas_str)
    try:
        with open(result_dir.joinpath('all_patterns_freq_true.json'), 'r', encoding='utf-8') as f:
            all_patterns_freq_true = json.load(f)
            all_stimulus_freqs = [int(freq) for freq in all_patterns_freq_true.keys()]
    except FileNotFoundError:
        pass
    accs_freqs = np.load(result_dir.joinpath(filename))
    accs_freqs_mean = np.mean(accs_freqs, axis=0)
    accs_freqs_std = np.std(accs_freqs, axis=0)    
    accs_freqs_median = np.median(accs_freqs, axis=0)
    accs_freqs_q25 = np.quantile(accs_freqs, 0.25, axis=0)
    accs_freqs_q75 = np.quantile(accs_freqs, 0.75, axis=0)
    # E-nose
    color = cmap(0) # 'r'
    ax.scatter(all_stimulus_freqs, accs_freqs_mean, color=color, linewidth=2, zorder=10, s=20)
    ax.plot(all_stimulus_freqs, accs_freqs_mean, color=color, linewidth=2, zorder=8, label=gas_str)    
    ax.fill_between(all_stimulus_freqs, np.clip(accs_freqs_mean-accs_freqs_std, 0, 1), np.clip(accs_freqs_mean+accs_freqs_std, 0, 1), color=color, alpha=0.1)
    # Formatting
    ax.axhline(1./6, c='gray', linestyle='--')#, label='random')
    ax.axhline(1.0, c='gray', linestyle='--')#, label='random')
    ax.set_xscale('log')
    ax.set_xticks(all_stimulus_freqs)
    ax.set_xticklabels(all_stimulus_freqs)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Modulation Frequency (Hz)")
    ax.set_ylim([0,1.05])
    ax.text(0, 0.5 + 0.02, "Random chance", color='gray', transform=ax.transAxes)
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend(loc='center right')
    fig.savefig(result_dir_parent.joinpath(f"classify_freq_test_IAEB_{modality}.svg"))
    fig.savefig(result_dir_parent.joinpath(f"classify_freq_test_IAEB_{modality}.png"), dpi=300)
    # fig.show()
    plt.show()

def plot_freqdiscr_IAEB_mouse_comparison(result_dir_parent, gases, modality, filename):
    """
    Plot frequency classification task and compare with mouse data

    :param result_dir_parent: Path
    :param gases: list  
    :param modality: str
    :param filename: str
    """
    # Plot frequency classification task
    enose_freq_ticks = [0, 2, 5, 6, 7] # 20 Hz vs 2, 5, 10, 40, 60 Hz 
    freqpairs_all = ['2 vs 20 Hz', '4 vs 20 Hz', '5 vs 20 Hz', '6 vs 20 Hz', '8 vs 20 Hz', '10 vs 20 Hz', '40 vs 20 Hz', '60 vs 20 Hz']
    mouse_freq_ticks = [0, 1, 3, 4, 5] # 20 Hz vs 2, 4, 6, 8, 10 Hz
    fig, ax = plt.subplots(figsize=(4,2.5))
    gas_str = '_'.join(gases)
    result_dir = result_dir_parent.joinpath(gas_str)
    # E-nose
    accs_freqpairs = {}
    freqpairs = {'2 vs 20 Hz': [2, 20,], '5 vs 20 Hz': [5, 20,], '10 vs 20 Hz': [10, 20,], '40 vs 20 Hz': [40, 20,], '60 vs 20 Hz': [60, 20]}
    for freqpair in freqpairs.keys():
        path = result_dir.joinpath(filename+f"{freqpair}_{modality}.npy")
        accs_freqpairs[freqpair] = np.load(path)
    color = cmap(0) # 'r'
    freq_ticks, accs_mean, accs_std = [], [], []
    accs_median, accs_q25, accs_q75 = [], [], []
    for j, freqpair in enumerate(freqpairs.keys()):
        acc_mean = np.mean(accs_freqpairs[freqpair])
        acc_std = np.std(accs_freqpairs[freqpair])
        accs_mean.append(acc_mean)
        accs_std.append(acc_std)
        freq_ticks.append(enose_freq_ticks[j])
        accs_median.append(np.median(accs_freqpairs[freqpair]))
        accs_q25.append(np.quantile(accs_freqpairs[freqpair], 0.25))
        accs_q75.append(np.quantile(accs_freqpairs[freqpair], 0.75))
    freq_ticks, accs_mean, accs_std = np.array(freq_ticks), np.array(accs_mean), np.array(accs_std)
    s_enose = ax.scatter(freq_ticks, accs_mean, color=color, zorder=8, label="e-nose", s=20)
    accs_mean = np.array(accs_mean)
    accs_median = np.array(accs_median)
    accs_q25 = np.array(accs_q25)
    accs_q75 = np.array(accs_q75)
    yerrs_clipped = [-1*(np.clip(accs_mean-accs_std, 0, 1)-accs_mean), np.clip(accs_mean+accs_std, 0, 1)-accs_mean]
    ax.errorbar(freq_ticks, accs_mean, yerr=yerrs_clipped, color='k', zorder=11, fmt='none')
    # Mouse
    # Load mouse frequency discrimination
    mouse_freq = scipy.io.loadmat("data/frequency_discrimination/freq_discr_results.mat")
    mouse_freqpairs = mouse_freq['colums'][0][:-2]
    freq_performances = mouse_freq['freq_performances'].T[:-2]
    for i, freqpair in enumerate(mouse_freqpairs):
        mouse_freqpair_performance = freq_performances[i]
        mouse_freqpair_performance_mean = np.mean(mouse_freqpair_performance)
        mouse_freqpair_performance_std = np.std(mouse_freqpair_performance)        
        mouse_freqpair_tick = mouse_freq_ticks[i]
        ax.scatter([mouse_freqpair_tick+0.3]*len(mouse_freqpair_performance), mouse_freqpair_performance, c='gray', s=5)
        bp_mouse = ax.boxplot(mouse_freqpair_performance, 
                   positions=[mouse_freqpair_tick], widths=0.4, showfliers=False, zorder=0, patch_artist=True, 
                   boxprops=dict(facecolor='dimgray', color='dimgray'), medianprops=dict(color='white'), whiskerprops=dict(color='k'), capprops=dict(color='k'), 
                   labels=['mouse'])
    # Formatting
    ax.axhline(0.5, c='gray', linestyle='--')
    ax.axhline(1.0, c='gray', linestyle='--')
    ax.set_xticks(range(len(freqpairs_all)), freqpairs_all, rotation='90')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Modulation Frequency (Hz)")
    ax.text(0, 0.5 + 0.02, "Random chance", color='gray')
    plt.setp(ax.spines.values(), linewidth=1.)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend([s_enose, bp_mouse['boxes'][0]], ['e-nose', 'mouse'], loc='center right')
    plt.savefig(result_dir.joinpath(f"classify_frequency_test_Rgas_vs_mouse_.svg"))
    plt.savefig(result_dir.joinpath(f"classify_frequency_test_Rgas_vs_mouse_.png"), dpi=300)
    plt.show()

def plot_corracorr_IAEB_mouse_comparison(result_dir_parent, gases, modality, filename):
    """
    Plot corr/acorr classification task and compare with mouse data

    :param result_dir_parent: Path
    :param gases: list
    :param modality: str
    :param filename: str
    """    
    fig, ax = plt.subplots(figsize=(4,2.5))
    gas_str = '_'.join(gases)
    result_dir = result_dir_parent.joinpath(gas_str)
    try:
        with open(result_dir.joinpath('all_patterns_freq_true.json'), 'r', encoding='utf-8') as f:
            all_patterns_freq_true = json.load(f)
            all_stimulus_freqs_gas = [int(freq) for freq in all_patterns_freq_true.keys()]
    except FileNotFoundError:
        pass
    accs_pattern = np.load(result_dir.joinpath(filename))
    accs_pattern_mean = np.mean(accs_pattern, axis=0)
    accs_pattern_std = np.std(accs_pattern, axis=0)
    accs_pattern_median = np.median(accs_pattern, axis=0)
    accs_pattern_q25 = np.quantile(accs_pattern, 0.25, axis=0)
    accs_pattern_q75 = np.quantile(accs_pattern, 0.75, axis=0)
    # CORR / ACORR
    # Plot corr/acorr classification task
    # Load mouse corr/acorr discrimination
    mouse_corracorr = scipy.io.loadmat("data/randomHz_performance/data_randomHz_performance.mat")
    mouse_pVals = mouse_corracorr['pVals'][0]
    mouse_testPerformance_mean = np.mean(mouse_corracorr['testPerformance'], axis=0)
    mouse_testPerformance_std = np.std(mouse_corracorr['testPerformance'], axis=0)
    mouse_testPerformance_median = np.median(mouse_corracorr['testPerformance'], axis=0)
    mouse_testPerformance_q25 = np.quantile(mouse_corracorr['testPerformance'], 0.25, axis=0)
    mouse_testPerformance_q75 = np.quantile(mouse_corracorr['testPerformance'], 0.75, axis=0)
    print("E-nose", accs_pattern_mean, accs_pattern_std)
    print("Mouse", mouse_testPerformance_mean, mouse_testPerformance_std)
    mouse_freqs_all = np.arange(1, len(mouse_testPerformance_mean)+1)
    isfinite = np.isfinite(mouse_testPerformance_mean)
    # E-nose
    color = cmap(0)
    ax.scatter(all_stimulus_freqs_gas, accs_pattern_mean, c=color if modality=='R_gas' else 'y', linewidth=2, zorder=10, s=20)
    ax.plot(all_stimulus_freqs_gas, accs_pattern_mean, c=color if modality=='R_gas' else 'y', linewidth=2, zorder=8, label='e-nose' if modality=='R_gas' else 'e-nose (T_heat)')
    ax.fill_between(all_stimulus_freqs_gas, np.clip(accs_pattern_mean-accs_pattern_std, 0, 1), np.clip(accs_pattern_mean+accs_pattern_std, 0, 1), color=color if modality=='R_gas' else 'y', alpha=0.1)
    # Mouse
    ax.scatter(mouse_freqs_all[isfinite], mouse_testPerformance_mean[isfinite], c='k', zorder=10)
    ax.plot(mouse_freqs_all[isfinite], mouse_testPerformance_mean[isfinite], c='k', label='mouse')
    ax.fill_between(mouse_freqs_all[isfinite], mouse_testPerformance_mean[isfinite]-mouse_testPerformance_std[isfinite], mouse_testPerformance_mean[isfinite]+mouse_testPerformance_std[isfinite], color='k', alpha=0.1)
    # Formatting
    ax.axhline(0.5, c='gray', linestyle='--')
    ax.axhline(1.0, c='gray', linestyle='--')
    ax.set_xscale('log')
    ax.set_xticks(all_stimulus_freqs_gas)
    ax.set_xticklabels(all_stimulus_freqs_gas)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Modulation Frequency (Hz)")
    ax.set_xlim([1.8,100])
    ax.text(2, 0.52, "Random chance", color='gray')
    plt.setp(ax.spines.values(), linewidth=1.)
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend()
    plt.savefig(result_dir.joinpath(f"classify_corracorr_test_Rgas_vs_mouse_.svg"))
    plt.savefig(result_dir.joinpath(f"classify_corracorr_test_Rgas_vs_mouse_.png"), dpi=300)
    plt.show()

def get_data_for_plots(index_selected, codes_selected, params):
    """
    Get data for plots

    :param index_selected: DataFrame
    :param codes_selected: list
    :param params: dict
    """
    nose_data_selected_allcodes, t_selected_allcodes, ffs_selected_allcodes = [], [], []
    for i, code_selected in enumerate(codes_selected):
        nose_data_selected, t_selected, ffs_selected = [], [], []
        for trial_idx, trial in index_selected.iterrows():
            # Select trial
            # check if trial is (corr or acorr) or something else
            if trial['kind'] == 'corr' or trial['kind'] == 'acorr':
                code_trial = f"{trial['kind'][:4]}_{trial['gas1']}_{trial['gas2']}_{trial['shape'][:-2]}"
            else:
                code_trial = "None"
                print(trial, "does not exist")
            if code_trial != code_selected:
                continue
            # Load trial as DataFrame
            trial_df = get_trial_df(trial, params["data_dir_enose"])
            # Extract data
            nose_data, t = trial_nose(trial_df, params["modality"], params["ms_start"], params["ms_end"])
            # Cut & Scale
            t_duration, nose_data_duration = cut(t, nose_data, t_start=params["stimulus_start"], duration=params["stimulus_duration"])
            nose_data_scaled = scale(nose_data_duration, params["modality"])
            # FFT                    
            nose_data_fft = fft_nose(nose_data_scaled)
            # Append to lists
            t_selected.append(t)
            nose_data_selected.append(nose_data)
            ffs_selected.append(nose_data_fft)                            
        nose_data_selected_allcodes.append(nose_data_selected)
        t_selected_allcodes.append(t_selected)
        ffs_selected_allcodes.append(ffs_selected)
    return nose_data_selected_allcodes, t_selected_allcodes, ffs_selected_allcodes