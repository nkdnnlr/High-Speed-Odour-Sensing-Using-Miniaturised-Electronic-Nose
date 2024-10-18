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
import pandas as pd
import importlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

parent_package = '.'.join(__package__.split('.')[:-1])
constants = importlib.import_module(f'{parent_package}.utils.constants')

cm_rainbow = plt.cm.rainbow
gases_colors = {
    "blank": (211 / 360, 211 / 360, 211 / 360, 1.0),
    "2H": cm_rainbow((2 * 3 / (2 * 4 - 1))),
    "IA": cm_rainbow((2 * 2 / (2 * 4 - 1))),
    "Eu": cm_rainbow((2 * 1 / (2 * 4 - 1))),
    "EB": cm_rainbow((2 * 0 / (2 * 4 - 1))),
} 

class ZBiasFreePlotter(object):
    """  used """
    # For randomization of z-order
    # https://stats.stackexchange.com/a/457316

    def __init__(self):
        self.plot_calls = []

    def add_plot(self, f, xs, ys, *args, **kwargs):
        self.plot_calls.append((f, xs, ys, args, kwargs))

    def draw_plots(self, chunk_size=16):
        scheduled_calls = []
        for f, xs, ys, args, kwargs in self.plot_calls:
            assert len(xs) == len(ys)
            index = np.arange(len(xs))
            np.random.shuffle(index)
            index_blocks = [
                index[i : i + chunk_size] for i in np.arange(len(index))[::chunk_size]
            ]
            for i, index_block in enumerate(index_blocks):
                # Only attach a label for one of the chunks
                if i != 0 and kwargs.get("label") is not None:
                    kwargs = kwargs.copy()
                    kwargs["label"] = None
                scheduled_calls.append(
                    (f, xs[index_block], ys[index_block], args, kwargs)
                )
        np.random.shuffle(scheduled_calls)
        for f, xs, ys, args, kwargs in scheduled_calls:
            f(xs, ys, *args, **kwargs)

def score_plot_bars_multifeatures(all_scores_concentrationpulses, results_dir_features, features=['cycle_raw', 'cycle_signature']):
    """ 
    Plot the scores for the different features and concentrations

    :param all_scores_concentrationpulses: dict with scores for each feature and concentration
    :param results_dir_features: Path to the results directory
    :param features: list of features to plot
    """
    fig, ax = plt.subplots(figsize=(3*1.2,2.3*1.2))
    # set width of bars and colors
    barWidth = 0.75
    if len(features)==2:
        colors = ['#A29582','#786E60','#514A3F', '#BB101F']
        feature_labels = {
            'cycle_raw': 'raw',
            'cycle_signature': 'normalised',
        }
    else:
        colors = ['#514A3F', '#BB101F', '#A29582','#786E60']
        feature_labels = {
            'constant_raw': 'T=400C, raw',
            'constant_subtracted': 'T=400C, baseline subtr.',
            'cycle_raw': 'T=[200, 400]C, raw',
            'cycle_signature': 'T=[200, 400]C, normalised',
        }
    all_scores_concentrationpulses = dict(sorted(all_scores_concentrationpulses.items(), reverse=True))
    for f, feature_selected in enumerate(features):
        print(feature_selected)
        jmax, xticks, xticklabels = 0, [], []
        for i, (concentration, all_scores) in enumerate(all_scores_concentrationpulses.items()):
            for j, (feature, scores_feature) in enumerate(all_scores.items()):
                if feature != feature_selected:
                    continue
                # Get mean & std
                score_mean = np.mean([score for score in scores_feature.values()])
                score_std = np.std([score for score in scores_feature.values()])
                print(f"C={concentration}%: {score_mean} pm {score_std}")
                # Barplots
                ax.bar(
                    i+f*barWidth/len(features),
                    score_mean,
                    color=colors[-f],
                    width=barWidth/len(features),
                    edgecolor="white",
                    label=feature_labels[feature] if i == 0 else "",
                )
                if jmax<j:
                    jmax=j
            xticks.append(i+0.5*barWidth/len(features))
            xticklabels.append(f"{concentration}%")
    # Deal with last column
    ax.axvline(0.7, linestyle='dashed', c='k', alpha=0.8)
    ax.text(x=0+0.5*barWidth/len(features), y=1.03, s='trained', fontsize=10, ha='center', va='bottom')
    # Set tick & axis labels 
    ax.set_ylim([0,1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Concentration level\n (duty cycle)")
    ax.set_ylabel("Accuracy")
    # remove tick marks
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)
    # Hide the right and top spines
    # change the color of the top and right spines to opaque gray
    plt.setp(ax.spines.values(), linewidth=1.)
    ax.spines[['right', 'top']].set_visible(False)
    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_style('italic')
    xlab.set_size(10)
    ylab.set_style('italic')
    ylab.set_size(10)
    plt.legend(loc='upper right')#, framealpha=0.5, edgecolor='white', fontsize=10, title='Feature')
    plt.grid(axis='y', linestyle='dashed')
    plt.tight_layout()
    plt.savefig(results_dir_features.joinpath(f"concentration_multifeatures.png"), dpi=300)
    plt.savefig(results_dir_features.joinpath(f"concentration_multifeatures.svg"))
    plt.show()

def pca_plots_new(X, y, results_dir, standard_scaler=True, feature="cycle_signature"):
    """ 
    Plot the PCA for the given feature

    :param X: Data
    :param y: Labels
    :param results_dir: Path to the results directory
    :param standard_scaler: whether to apply standard scaler
    :param feature: feature to plot
    """
    # Apply standard scaler
    if standard_scaler:
        standard_scaler = StandardScaler().fit(X)
        X = standard_scaler.transform(X)
    # Do PCA and get explained variance
    pca = PCA()
    pca.fit(X)
    if standard_scaler:
        x = pca.transform(standard_scaler.transform(X))
    else:
        x = pca.transform(X)
    expl_var = pca.explained_variance_ratio_
    expl_var_cumulative = [sum(expl_var[: i + 1]) for i in range(len(expl_var))]
    # Initiate PCA plot
    n_pcs = 3
    fig, ax = plt.subplots(ncols=n_pcs + 1, figsize=(5*0.75 * (n_pcs), 4*0.75))
    label_nrs = {"EB": 0, "Eu": 1, "IA": 2, "2H": 3, "blank": 4} 
    ax[0].plot(
        range(1, len(expl_var_cumulative) + 1),
        100 * np.array(expl_var_cumulative),
        c="k",
    )
    ax[0].scatter(
        range(1, len(expl_var_cumulative) + 1),
        100 * np.array(expl_var_cumulative),
        c="r",
    )
    ax[0].set_xlabel("Number of PCs", fontstyle='italic')#)
    ax[0].set_ylabel("Explained Variance in %", fontstyle='italic')#)
    ax[0].set_xscale("log")
    ax[0].grid()
    ax[0].set_xticks([1, 10, 100, 600])
    # Bias free plotter
    for ax_pc, (pc_x, pc_y) in enumerate(zip([1,1,2], [2,3,3])):
        bias_free_plotter = ZBiasFreePlotter()
        for l, _ in label_nrs.items(): 
            color = gases_colors[l]
            xs = x[y == l, pc_x-1]
            ys = x[y == l, pc_y-1]
            bias_free_plotter.add_plot(
                ax[ax_pc+1].scatter, xs, ys, color=color, s=5, label=l
            )
        bias_free_plotter.draw_plots()
        # Formatting
        ax[ax_pc+1].set_xlabel(f"PC{pc_x} ({np.round(100*expl_var[pc_x-1],1)}%)", fontstyle='italic')
        ax[ax_pc+1].set_ylabel(f"PC{pc_y} ({np.round(100*expl_var[pc_y-1],1)}%)", fontstyle='italic')
        ax[ax_pc+1].grid()
    handles, labels = ax[1].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles)))
    fig.legend(handles, labels, ncol=5, bbox_to_anchor=(0.65, 1.04))
    for _ax in ax.flatten():
        _ax.set_box_aspect(1)
    plt.savefig(results_dir.joinpath(f"pca_{feature}.svg"), bbox_inches="tight")
    plt.savefig(results_dir.joinpath(f"pca_{feature}.png"), bbox_inches="tight", dpi=300)
    plt.tight_layout()
    plt.show()
    plt.close()

def tsne_plot(X, y, feature, output_dir, perplexity=20):
    """ 
    Plot the t-SNE for the given feature
    # https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

    :param X: Data
    :param y: Labels
    :param feature: feature to plot
    :param output_dir: Path to the output directory
    :param perplexity: Perplexity for t-SNE   
    """
    # Compute t-distributed stochastic neighbor embedding (t-SNE)
    output_dir.mkdir(exist_ok=True, parents=True)
    np.random.seed(42)
    tsne = TSNE(
        n_components=2,
        verbose=0,
        perplexity=perplexity,
        n_iter=300,
        learning_rate="auto",
        random_state=42,
        init="random",
    )
    tsne_results = tsne.fit_transform(X)
    label_nrs = {"EB": 0, "Eu": 1, "IA": 2, "2H": 3, "blank": 4} 
    # Plotting
    fig = plt.figure(figsize=(5*0.75,4*0.75))
    ax = fig.add_subplot(111)
    bias_free_plotter = ZBiasFreePlotter()
    for l, nr in label_nrs.items():
        color = gases_colors[l]
        idx = y == l
        xs = tsne_results[idx, 0]
        ys = tsne_results[idx, 1]
        bias_free_plotter.add_plot(ax.scatter, xs, ys, color=color, s=5, label=l)
    bias_free_plotter.draw_plots()
    ax.set_xlabel("t-SNE dim1")
    ax.set_ylabel("t-SNE dim2")
    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_style('italic')
    xlab.set_size(10)
    ylab.set_style('italic')
    ylab.set_size(10)
    plt.legend()
    plt.savefig(output_dir.joinpath(f"tsne_p{perplexity}_{feature}.svg"), bbox_inches="tight")
    plt.savefig(
        output_dir.joinpath(f"tsne_p{perplexity}_{feature}.png"), bbox_inches="tight", dpi=300
    )
    plt.show()
    plt.close()

def feature_summary_plot(analysis, selected, sensors_all, features_all, phase, results_dir_features=None, params=None):
    """
    Plot the feature summary for the given phase

    :param analysis: Analysis object
    :param selected: Selected trials
    :param sensors_all: list of sensors
    :param features_all: list of features
    :param phase: phase to plot
    :param results_dir_features: Path to the results directory
    :param params: parameters
    """
    feature_labels = {
        'constant_raw': 'T const.\nraw\n(Ohm)',
        'constant_subtracted': 'T const.\nnormalised\n(Ohm)',
        'cycle_raw': 'raw\n(Ohm)',
        'cycle_signature': 'normalised\n (a.u.)',
    }
    sensor_colors = {
        0: '#1b9e77',
        1: '#d95f02',
        2: '#7570b3',
        3: '#e7298a',
    }
    gases = ['blank', 'IA', 'EB', 'Eu', '2H']
    fig, ax = plt.subplots(nrows=len(features_all), ncols=len(gases), sharex=True, sharey='row', figsize=(1.7*len(gases), 1.7*len(features_all)))
    m = 0
    for g, gas in enumerate(gases):
        selected_gas = selected.query(f"gas1 == '{gas}'")
        for j, (sensors, feature) in enumerate(zip(sensors_all, features_all)):
            # Get data
            trial_data = analysis.get_trial_data(selected_gas, sensors=sensors)
            # Get cycled features
            X_all, y_gas1_all, y_gas2_all, phases_all, phases_rejected_all = analysis.get_X_y_ph(trial_data_all=trial_data, feature=feature, params=params)
            # Extract first feature after phase 'ph'
            phase_sel = np.array([phases[phases>phase][0] for phases in phases_all])
            y_sel = np.array([y[phases>phase][0] for y, phases in zip(y_gas1_all, phases_all)])
            X_sel = np.array([X[phases>phase][0] for X, phases in zip(X_all, phases_all)])
            # Reshape to get per sensor, then calculate mean & std
            X_sel = X_sel.reshape(len(X_sel), 4, 50)
            X_sel_mean, X_sel_std = np.mean(X_sel, axis=0), np.std(X_sel, axis=0)
            t = np.arange(len(X_sel_mean[0])) + phase
            # Plot raw & mean
            for s in range(4):
                for _x in X_sel:
                    ax[j, g].plot(t, _x[s], c='k', linewidth=0.1)
            for s in range(4):
                ax[j, g].plot(t, X_sel_mean[s], c=sensor_colors[s], linewidth=2)
            # Formatting
            ax[0, g].set_title(f"{gas}", fontstyle='italic')
            if j == 0:
                # tweak the axis labels
                xlab = ax[j, g].xaxis.get_label()
                ylab = ax[j, g].yaxis.get_label()
                xlab.set_style('italic')
                xlab.set_size(10)
                ylab.set_style('italic')
                ylab.set_size(10)
            if g==0:
                ax[j, g].set_ylabel(feature_labels[feature])
    # Formatting
    for _ax in ax.flatten():
        _ax.set_box_aspect(1)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(r"Time (ms) - $\rho$", fontstyle='italic')
    plt.ylabel("Sensor resistance", fontstyle='italic')
    fig.align_ylabels(ax)
    # Save fig
    plt.savefig(results_dir_features.joinpath("features_timeseries.svg"), bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def feature_plot_trial(analysis, selected, result_dir, features_all=['constant_raw', 'cycle_raw'], sensors_all=[[1,2,3,4], [5,6,7,8]], gas_selected='2H', t_before=1, t_after=4):
    """
    Plot the feature for the given trial

    :param analysis: Analysis object
    :param selected: Selected trials
    :param result_dir: Path to the results directory
    :param features_all: list of features
    :param sensors_all: list of sensors
    :param gas_selected: gas to plot
    :param t_before: time before
    :param t_after: time after
    """
    mode = 'single'
    sensor_colors = {
        0: '#1b9e77',
        1: '#d95f02',
        2: '#7570b3',
        3: '#e7298a',
    }
    for j, (sensors, feature) in enumerate(zip(sensors_all, features_all)):
        sensor_str = analysis.sensors_list2str(sensors)
        # Get data
        trial_data = analysis.get_trial_data(selected, sensors=sensors)
        data_sensors_feature = []
        for trial_idx, trial in trial_data.items():
            if trial.trial['gas1'] == gas_selected:
                # Getting data
                nose = trial.trial_df
                # Extracting sensor data
                data_sensors = []
                for s in sensors:
                    data_sensors.append(nose[f"R_gas_{s}"])
                data_sensors = np.array(data_sensors)
                data_sensors_feature.append(data_sensors)
        data_sensors_feature = np.array(data_sensors_feature)
        # Plotting
        fig, ax = plt.subplots(figsize=(6,2))
        t_array = np.arange(data_sensors_feature.shape[2]) - t_before*1000
        ax.axvspan(0, 1000, alpha=0.2, facecolor='red')
        if mode == 'single':
            # One single trial
            trial = 0
            for s, data_sensor in enumerate(data_sensors_feature[trial]):
                ax.plot(t_array, data_sensor, c=sensor_colors[s], label=constants.sensor_names[s][:-2])
        elif mode == 'all':
            # All trials (raw and averaged)
            for data_sensors in data_sensors_feature:
                ax.plot(t_array, data_sensors.T, linewidth=0.1, c='k')
            for s, data_sensor in enumerate(np.mean(data_sensors_feature, axis=0)):
                ax.plot(t_array, data_sensor, c=sensor_colors[s], label=constants.sensor_names[s][:-2])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Sensor Resistance (Ohm)")
        ax.set_yscale('log')
        ax.set_xlim(-250, 1500)
        # tweak the axis labels
        xlab = ax.xaxis.get_label()
        ylab = ax.yaxis.get_label()
        xlab.set_style('italic')
        xlab.set_size(10)
        ylab.set_style('italic')
        ylab.set_size(10)
        ax.spines[['right', 'top']].set_visible(False)
        ax.legend(loc='upper right')
        ax.grid(axis="y")
        plt.savefig(result_dir.joinpath(f"feature_{feature}_{gas_selected}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(result_dir.joinpath(f"feature_{feature}_{gas_selected}.svg"), bbox_inches='tight')
        plt.tight_layout()
        plt.show()

def plot_labels(y, phases, phases_rejected, params, results_dir):
    """ 
    Plot the labels

    :param y: labels
    :param phases: phases
    :param phases_rejected: rejected phases
    :param params: parameters
    :param results_dir: Path to the results directory   
    """
    min_2H, max_2H = np.min(phases[y=='2H']), np.max(phases[y=='2H']+params["period"])
    min_EB, max_EB = np.min(phases[y=='EB']), np.max(phases[y=='EB']+params["period"])
    min_Eu, max_Eu = np.min(phases[y=='Eu']), np.max(phases[y=='Eu']+params["period"])
    min_IA, max_IA = np.min(phases[y=='IA']), np.max(phases[y=='IA']+params["period"])
    min_blank, max_blank = np.min(phases[y=='blank']), np.max(phases[y=='blank']+params["period"])
    min_rejected, max_rejected = np.min(phases_rejected), np.max(phases_rejected)+params["period"]

    fig, ax = plt.subplots(figsize=(8,1.5))
    # Gas pulses
    odour = '2H'
    patch_odour = patches.Rectangle((min_2H, 4), max_2H-min_2H, 0.8, facecolor=gases_colors[odour], label=odour, zorder=10)
    patch_blank = patches.Rectangle((min_blank, 4), max_blank-min_blank, 0.8, facecolor='gray', zorder=1)
    patch_rejected = patches.Rectangle((min_rejected, 4), max_rejected-min_rejected, 0.8, facecolor='lightgray', zorder=0)
    ax.add_patch(patch_odour)
    ax.add_patch(patch_blank)
    ax.add_patch(patch_rejected)

    odour = 'EB'
    patch_odour = patches.Rectangle((min_EB, 3), max_EB-min_EB, 0.8, facecolor=gases_colors[odour], label=odour, zorder=10)
    patch_blank = patches.Rectangle((min_blank, 3), max_blank-min_blank, 0.8, facecolor='gray', zorder=1)
    patch_rejected = patches.Rectangle((min_rejected, 3), max_rejected-min_rejected, 0.8, facecolor='lightgray', zorder=0)
    ax.add_patch(patch_odour)
    ax.add_patch(patch_blank)
    ax.add_patch(patch_rejected)

    odour = 'Eu'
    patch_odour = patches.Rectangle((min_Eu, 2), max_Eu-min_Eu, 0.8, facecolor=gases_colors[odour], label=odour, zorder=10)
    patch_blank = patches.Rectangle((min_blank, 2), max_blank-min_blank, 0.8, facecolor='gray', zorder=1)
    patch_rejected = patches.Rectangle((min_rejected, 2), max_rejected-min_rejected, 0.8, facecolor='lightgray', zorder=0)
    ax.add_patch(patch_odour)
    ax.add_patch(patch_blank)
    ax.add_patch(patch_rejected)

    odour = 'IA'
    patch_odour = patches.Rectangle((min_IA, 1), max_IA-min_IA, 0.8, facecolor=gases_colors[odour], label=odour, zorder=10)
    patch_blank = patches.Rectangle((min_blank, 1), max_blank-min_blank, 0.8, facecolor='gray', zorder=1)
    patch_rejected = patches.Rectangle((min_rejected, 1), max_rejected-min_rejected, 0.8, facecolor='lightgray', zorder=0)
    ax.add_patch(patch_odour)
    ax.add_patch(patch_blank)
    ax.add_patch(patch_rejected)

    # Blank pulse
    odour = 'blank'
    patch_blank = patches.Rectangle((min_blank, 0), max_blank-min_blank, 0.8, facecolor='gray', label='blank', zorder=1)
    patch_rejected = patches.Rectangle((min_rejected, 0), max_rejected-min_rejected, 0.8, facecolor='lightgray', label='rejected', zorder=0)
    ax.add_patch(patch_blank)
    ax.add_patch(patch_rejected)
    ax.set_ylim(-0.5, 5)
    ax.axvline(0, linestyle='--', c='k', zorder=100, linewidth=1)
    ax.axvline(1000, linestyle='--', c='k', zorder=100, linewidth=1)
    ax.text(x=500, y=5, s=f"Odour Pulse: 1000 ms", ha='center', va='bottom', fontsize=8)
    ax.text(x=-80, y=4+0.4, s=f"2H", ha='right', va='center', fontsize=8)
    ax.text(x=-80, y=3+0.4, s=f"EB", ha='right', va='center', fontsize=8)
    ax.text(x=-80, y=2+0.4, s=f"Eu", ha='right', va='center', fontsize=8)
    ax.text(x=-80, y=1+0.4, s=f"IA", ha='right', va='center', fontsize=8)
    ax.text(x=-80, y=0+0.4, s=f"blank", ha='right', va='center', fontsize=8)
    ax.set_xlabel("Time (ms)")
    ax.set_yticks([])
    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_style('italic')
    xlab.set_size(10)
    ylab.set_style('italic')
    ylab.set_size(10)
    ax.spines[['right', 'top', 'left']].set_visible(False)
    ax.legend(title='Assigned label', ncol=3)
    plt.savefig(results_dir.joinpath("plot_labels.png"), dpi=300, bbox_inches='tight')
    plt.savefig(results_dir.joinpath("plot_labels.svg"), bbox_inches='tight')
    plt.show()


def eventplot_all(ax, gases, splits, y_all, y_pred_all, phases_all, params):
    """
    Plot the eventplot for all gases

    :param ax: axis
    :param gases: list of gases
    :param splits: number of splits
    :param y_all: true labels
    :param y_pred_all: predicted labels
    :param phases_all: phases
    :param params: parameters
    """
    y_offsets_all, y_offsets_gas_all = [], []
    # Iterate over gases
    yoffsets = 1
    for g_idx, gas in enumerate(gases):
        y_offsets_gas = 1
        for k in range(splits):
            y, y_pred, phases = y_all[k], y_pred_all[k], phases_all[k]
            # Sort by phase
            sorted_indices = np.argsort(phases[:,0])
            y = y[sorted_indices]
            y_pred = y_pred[sorted_indices]
            phases = phases[sorted_indices]
            # Find index where ground_truth == gas
            idx = np.where(y==gas)[0]
            # Iterate over trials for each gas
            for n in idx:
                # # Sort per predicted gas
                phs_sorted = []
                colors = []
                for i, unique_pred in enumerate(np.unique(y_pred)):
                    colors.append(gases_colors[unique_pred])
                    phs_sorted.append([])
                    for _phs, _y_pred in zip(phases[n], y_pred[n]):
                        if _y_pred == unique_pred:
                            # Add period duration, as sample is taken at the end of period
                            phs_sorted[i].append(_phs+params["period"])
                # Make eventplot
                ax.eventplot(phs_sorted, lineoffsets=[yoffsets]*len(phs_sorted), colors=colors)#, label=gas if (n==0 and k==0) else "")
                yoffsets -=1
                y_offsets_gas += 1
        #  # Make line between gases
        if gas != 'blank':
            ax.axhline(yoffsets, linestyle='-', c='gray', linewidth=0.5)
        y_offsets_all.append(yoffsets)
        y_offsets_gas_all.append(y_offsets_gas)
        yoffsets -=1
    return y_offsets_all, y_offsets_gas_all

def eventplot_classification_singlepulse(y_all, y_pred_all, phases_all, params, splits=None, width=1000, output_dir=None, title=None):
    """
    Plot the eventplot for the classification

    :param y_all: true labels
    :param y_pred_all: predicted labels
    :param phases_all: phases
    :param params: parameters
    :param splits: number of splits
    :param width: width of the pulse
    :param output_dir: Path to the output directory
    :param title: title
    """
    y_all = np.array(y_all)[:,:,0]
    gases = np.unique(y_all.flatten())
    if splits is None:
        splits = 1

    fig, ax = plt.subplots(figsize=(5,3.2))
    y_offsets_all, y_offsets_gas_all = eventplot_all(ax, gases, splits, y_all, y_pred_all, phases_all, params)
    for g_idx, gas in enumerate(gases):
        ax.text(x=-60, y=y_offsets_all[g_idx]+y_offsets_gas_all[g_idx]/2+0.25, s=gas, rotation=90, color='k', ha='center', va='center', fontsize=9)#, weight='bold', style='italic')
    # Inset
    axin = ax.inset_axes([0.85, 0.83, 0.2, 0.2])
    eventplot_all(axin, gases, splits, y_all, y_pred_all, phases_all, params)
    axin.set_xlim(30, 150)
    axin.set_ylim(top=-38, bottom=-41)
    axin.set_yticks([])
    axin.set_xticks([])
    ax.indicate_inset_zoom(axin)
    # Add lines and text to denote odour pulse
    ax.axvline(0, linestyle='--', c='k', linewidth=1)
    ax.axvline(width, linestyle='--', c='k', linewidth=1)
    ax.text(x=width/2, y=3.5, s=f"Odour pulse: {width} ms", ha='center', va='bottom', fontsize=9)#, weight='bold')
    # Formatting
    ax.set_yticks([])
    ax.set_ylim(ymin=y_offsets_all[-1]-1)
    # ax.set_xticks([0,500,1000,1500,2000])
    ax.set_xlim(xmin=-25, xmax=1500)
    # ax.set_ylabel("True odour")
    ax.set_xlabel(f"Time (ms)")
    xlab = ax.xaxis.get_label()
    xlab.set_style('italic')
    xlab.set_size(10)
    ylab = ax.yaxis.get_label()
    ylab.set_style('italic')
    ylab.set_size(10)
    # # Hide the right and top spines
    plt.setp(ax.spines.values(), linewidth=1.)
    ax.spines[['right', 'top', 'left']].set_visible(False)
    # # Create legend
    # for label in ['IA', 'Eu', 'EB', '2H', 'blank']:
    for label in ['2H', 'EB', 'Eu', 'IA', 'blank']:
        color = gases_colors[label]
        ax.scatter([], [], c=color, label=label, marker="|")
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(gases_colors))
    # ax.legend(loc='center', bbox_to_anchor=(1.08, 0.5), title='Predicted odour')
    ax.legend(title='Predicted', loc='lower right')
    if output_dir is not None:
        plt.savefig(output_dir.joinpath(f"temporal_{title}.svg"), bbox_inches='tight')
        plt.savefig(output_dir.joinpath(f"temporal__{title}.png"), dpi=300, bbox_inches='tight')
    plt.show()

def scatterplot_statistics_singlepulse(widths, all_acc, all_phases_onset, all_phases_offset, results_dir, wide=True):
    """
    Plot the scatterplot for the classification statistics

    :param widths: pulse widths
    :param all_acc: accuracies
    :param all_phases_onset: phases onset
    :param all_phases_offset: phases offset
    :param results_dir: Path to the results directory
    :param wide: whether to plot wide
    """
    if wide:
        fig, ax = plt.subplots(ncols=3, figsize=(9,2), sharex=True)
    else:
        fig, ax = plt.subplots(nrows=3, figsize=(3.6,4.9), sharex=True)

    # Classification
    ax[0].axhline(0.2, linestyle='--', c='k', zorder=1, alpha=0.8)
    ax[0].scatter(widths, all_acc, c='r', zorder=5, s=15)
    ax[0].set_ylabel("Accuracy")
    ax[0].grid()
    ax[0].text(s="Random classification", x=120, y=0.275, c='k', alpha=0.8)
    ax[0].set_yticks([0, 0.2,0.5,1.0])

    ax[0].set_ylim(bottom=0)
    # Phases Onset
    onsets_flat = np.array([onset for _onset in all_phases_onset for onset in _onset])
    for i, (phase_onset, width) in enumerate(zip(all_phases_onset, widths)):
        ax[1].scatter(width, np.mean(phase_onset), c='r', zorder=6, s=15, label="mean" if i==0 else "")
        ax[1].scatter([width]*len(phase_onset), np.array(phase_onset), c='k', zorder=5, s=5, label="raw" if i==0 else "")
    ax[1].set_ylabel("Onset (ms)")
    ax[1].grid()
    ax[1].set_ylim(bottom=0)
    
    # Phases Offset
    offsets_flat = np.array([offset-width for (width, _offset) in zip(widths, all_phases_offset) for offset in _offset])
    for i, (phase_offset, width) in enumerate(zip(all_phases_offset, widths)):
        ax[2].scatter(width, np.mean(phase_offset)-width, c='r', zorder=6, s=15, label="mean" if i==0 else "")
        ax[2].scatter([width]*len(phase_offset), np.array(phase_offset)-width, c='k', zorder=5, s=5, label="raw" if i==0 else "")
    ax[2].set_xscale('log')
    ax[2].set_ylabel("Offset (ms)", fontstyle='italic')
    if wide:
        ax[0].set_xlabel("Pulse duration (ms)", fontstyle='italic')
        ax[1].set_xlabel("Pulse duration (ms)", fontstyle='italic')
        ax[2].set_xlabel("Pulse duration (ms)", fontstyle='italic')
    ax[2].set_xlabel("Pulse duration (ms)", fontstyle='italic') 
    ax[2].set_xticks([10, 20, 50, 100, 200, 500, 1000])
    ax[2].set_xticklabels([10, 20, 50, 100, 200, 500, 1000])
    for _ax in ax:
        for tick in _ax.get_xticklabels():
            tick.set_rotation(45)
    ax[2].set_yticks([0, 100, 200, 300])
    ax[2].grid()
    ax[2].set_ylim(bottom=0)
    ax[1].legend(loc='upper right', framealpha=0.5, edgecolor='k', fontsize=10, bbox_to_anchor=(1.25, 1.0))
    for _ax in ax:
        # Hide the right and top spines
        plt.setp(_ax.spines.values(), linewidth=1.)
        _ax.spines[['right', 'top']].set_visible(False)
    fig.align_ylabels(ax)
    plt.savefig(results_dir.joinpath("temporal_classification_statistics.svg"), bbox_inches='tight')
    plt.savefig(results_dir.joinpath("temporal_classification_statistics.png"), dpi=300, bbox_inches='tight')
    plt.tight_layout()

def eventplot_classification_twopulses(y1_all, y2_all, y_pred_all, phases_all, period=50, splits=None, results_dir=None, title=None, freq=None):
    """
    Plot the eventplot for the classification of two pulses

    :param y1_all: true labels for pulse 1
    :param y2_all: true labels for pulse 2
    :param y_pred_all: predicted labels
    :param phases_all: phases
    :param period: period
    :param splits: number of splits
    :param results_dir: Path to the results directory
    :param title: title
    :param freq: frequency
    """
    y1_all = np.array(y1_all)[:,:,0]
    y2_all = np.array(y2_all)[:,:,0]
    gases_1 = np.unique(y1_all.flatten())
    gases_2 = np.unique(y2_all.flatten())
    if splits is None:
        splits = 1
    start = 0
    end = 1
    fig, ax = plt.subplots(figsize=(5, 0.74*6))
    # Iterate over gases
    yoffsets = 0
    yticks = []
    yticklabels = []
    for g_idx1, gas1 in enumerate(gases_1):
        y_offsets_gas = 1
        for g_idx2, gas2, in enumerate(gases_2):
            for k in range(splits):
                y1, y2, y_pred, phases = y1_all[k], y2_all[k], y_pred_all[k], phases_all[k]
                # Find index for proper gas1-gas2 combination
                idx = np.where(np.logical_and(y1==gas1, y2==gas2))[0]       
                if len(idx)==0:
                    continue
                # Iterate over trials for each gas
                for n in idx:
                    # # Sort per predicted gas
                    phs_sorted = []
                    colors = []
                    for i, unique_pred in enumerate(np.unique(y_pred)):
                        colors.append(gases_colors[unique_pred])
                        phs_sorted.append([])
                        for _phs, _y_pred in zip(phases[n], y_pred[n]):
                            if _y_pred == unique_pred:
                                phs_sorted[i].append(_phs+period)
                    yticks.append(yoffsets)
                    yticklabels.append(gas2)
                    # Make eventplot
                    ax.eventplot(phs_sorted, lineoffsets=[yoffsets]*len(phs_sorted), colors=colors)
                    yoffsets -=1
                    y_offsets_gas += 1
        # Make line between gases
        if gas1 != 'blank':
            ax.axhline(yoffsets, linestyle='-', c='gray', linewidth=0.5)
        # Add some text to label gas
        ax.text(x=-170 if gas1!='blank' else -250, y=yoffsets+y_offsets_gas/2+0.25, s=f"{gas1} -", rotation=0, color=gases_colors[gas1], ha='center', va='center', fontsize=6)#, weight='bold', style='italic')
        yoffsets -=1
    # Set gas names as yticks
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=6)
    for ticklabel_object, ticklabel in zip(ax.get_yticklabels(), yticklabels):#:zip(plt.gca().get_xticklabels(), my_colors):
        ticklabel_object.set_color(gases_colors[ticklabel])
    if freq is not None:
        start = 0
        end =   1
        vlines = np.arange(1000*start, 1000*((end-start)+(end-start)/(2*freq)), 1000*((end-start)/(2*freq)))
        vline_old = 0
        color = 'green'
        for vline in vlines:
            if color == 'purple':
                color = 'green'
            elif color == 'green':
                color = 'purple'
            ax.axvspan(vline_old, vline, alpha=0.1, facecolor=color)
            vline_old = vline
    # Formatting
    ax.set_xticks([0,500,1000,1500,2000])
    ax.set_xlabel(f"Time (ms)")
    xlab = ax.xaxis.get_label()
    xlab.set_style('italic')
    xlab.set_size(10)
    ax.yaxis.set_tick_params(length=0)
    ax.set_xlim(xmin=-25, xmax=1500)
    # Hide the right and top spines
    plt.setp(ax.spines.values(), linewidth=1.)
    ax.spines[['right', 'top', 'left']].set_visible(False)
    # Make legend
    for label, color in gases_colors.items():
        ax.scatter([], [], c=color, label=label)
    ax.legend(loc='upper center')
    # Set title and save
    plt.suptitle(f"{title}")
    plt.savefig(results_dir.joinpath(f"temporal_classification_twopulse_{title}.svg"), bbox_inches='tight')
    plt.savefig(results_dir.joinpath(f"temporal_classification_twopulse_{title}.png"), dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def summaryplot_classification_singlepulse_comparison(widths, y_val_dict, y_pred_val_dict, phases_val_dict, params, result_dir):
    """
    Plot the summary plot for the classification of single pulses

    :param widths: pulse widths
    :param y_val_dict: true labels
    :param y_pred_val_dict: predicted labels
    :param phases_val_dict: phases
    :param params: parameters
    :param result_dir: Path to the results directory
    """
    n_rows = len(widths)
    fig, ax = plt.subplots(nrows=n_rows, figsize=(5,0.72*n_rows), sharex=True, sharey=True)
    for i, width_ms in enumerate(widths):
        if width_ms == 1000:
            splits=1
        else:
            splits=1
        y_val_all, y_pred_val_winner, phases_val_all = np.array(y_val_dict[width_ms]), np.array(y_pred_val_dict[width_ms]), np.array(phases_val_dict[width_ms])
        ax[i].axvspan(0, width_ms, alpha=0.1, facecolor='purple')
        phases_all = []
        y_pred_correctgas_all = []
        y_pred_blank_all = []
        y_pred_wronggas_all = []
        for k in range(splits):
            y_val, y_pred, phase = y_val_all[k], y_pred_val_winner[k], phases_val_all[k]
            non_blank = y_val.T[0]!='blank'
            n_nonblanks = np.sum(non_blank)
            y_val, y_pred, phase = y_val[non_blank], y_pred[non_blank], phase[non_blank]
            y_pred_correctgas = np.sum(y_pred==y_val, axis=0)
            y_pred_blank = np.sum(y_pred=='blank', axis=0)
            y_pred_wronggas = np.sum((y_pred!='blank') & (y_pred!=y_val), axis=0)
            y_pred_correctgas_all.append(y_pred_correctgas)
            y_pred_blank_all.append(y_pred_blank)
            y_pred_wronggas_all.append(y_pred_wronggas)
        ax[i].plot(np.mean(phase+50, axis=0), 100*np.mean(y_pred_correctgas_all, axis=0)/n_nonblanks, c='limegreen', label='Correct odour', zorder=10)
        ax[i].plot(np.mean(phase+50, axis=0), 100*np.mean(y_pred_wronggas_all, axis=0)/n_nonblanks, c='r', label='Wrong odour')
        ax[i].plot(np.mean(phase+50, axis=0), 100*np.mean(y_pred_blank_all, axis=0)/n_nonblanks, c='gray', label='No odour')
        if i==0:
            ax[i].text(484, 50, f"Odour Pulse: {width_ms} ms", color='dimgray', ha='center', va='center')
        else:
            ax[i].text(750, 50, f"{width_ms} ms", color='dimgray', ha='center', va='center')
        ax[i].set_xlim(xmax=1500)
        ax[i].spines[['right', 'top']].set_visible(False)
        if i == 3:
            ax[i].legend( loc = (1.05,0), framealpha=1.0, edgecolor='k')
    ax[-1].set_xlabel("Time (ms)", fontstyle='italic')
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Prediction (in %)", fontstyle='italic')
    plt.savefig(result_dir.joinpath("singlepulse_comparison.svg"))
    plt.show()

def plot_traintest_split_dynamic(index_df_trainval, index_df_test, results_dir_single_pulses, show=False):
    """ 
    Plot the train-test split for dynamic single pulses

    :param index_df_trainval: training/validation index
    :param index_df_test: testing index
    :param results_dir_single_pulses: Path to the results directory
    :param show: whether to show the plot
    """
    min_train, max_train = min(index_df_trainval['t_stimulus']), max(index_df_trainval['t_stimulus']), 
    min_test, max_test = min(index_df_test['t_stimulus']), max(index_df_test['t_stimulus'])
    rect_train = patches.Rectangle((min_train, 0.5), max_train-min_train, 1, edgecolor='none', facecolor='green')#, label='train / val')
    rect_test = patches.Rectangle((min_test, 0.5), max_test-min_test, 1, edgecolor='none', facecolor='red')#, label='test')
    fig, ax = plt.subplots(figsize=(4,2))
    ax.set_xlim([min(min_train, min_test)-0.2, max(max_train, max_test)+0.1])
    ax.add_patch(rect_train)
    ax.add_patch(rect_test)
    ax.text(x=min_train + 0.5*(max_train-min_train), y=1, s=f"Training / Validation", ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x=min_test + 0.5*(max_test-min_test), y=1, s=f"Testing", ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x=min_train-0.3, y=1, s=f"Sensor 1-8", ha='right', va='center', fontsize=9)
    ax.set_yticks([])
    ax.set_ylim([0,3])
    ax.set_xlabel('Time (h)')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    plt.savefig(results_dir_single_pulses.joinpath(f'traintest_split_dynamic.svg'))#, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
def plot_traintest_split_static(index_df_pulses_1s_c5_train, index_df_pulses_1s_c5_test, result_dir, show=False):
    """
    Plot the train-test split for static single pulses

    :param index_df_pulses_1s_c5_train: training/validation index
    :param index_df_pulses_1s_c5_test: testing index
    :param result_dir: Path to the results directory
    :param show: whether to show the plot
    """
    index_df_pulses_1s_c5_train['t_stimulus'] = (pd.to_timedelta(index_df_pulses_1s_c5_train['t_stimulus'])).dt.total_seconds()/60./60.
    index_df_pulses_1s_c5_test['t_stimulus'] = pd.to_timedelta(index_df_pulses_1s_c5_test['t_stimulus']).dt.total_seconds()/60./60.
    fig, ax = plt.subplots(figsize=(4,2))
    min_train, max_train = min(index_df_pulses_1s_c5_train['t_stimulus']), max(index_df_pulses_1s_c5_train['t_stimulus']), 
    min_test, max_test = min(index_df_pulses_1s_c5_test['t_stimulus']), max(index_df_pulses_1s_c5_test['t_stimulus'])
    ax.set_xlim([min(min_train, min_test)-0.2, max(max_train, max_test)+0.1])
    rect_unused_1 = patches.Rectangle((min_train, 1.5+0.1), max_train-min_train, 1, edgecolor='none', facecolor='gray')#, label='train / val')
    rect_unused_2 = patches.Rectangle((min_test, 1.5+0.1), max_test-min_test, 1, edgecolor='none', facecolor='gray')#, label='test')
    ax.add_patch(rect_unused_1)
    ax.add_patch(rect_unused_2)
    rect_train = patches.Rectangle((min_train, 0.5), max_train-min_train, 1, edgecolor='none', facecolor='green')#, label='train / val')
    rect_test = patches.Rectangle((min_test, 0.5), max_test-min_test, 1, edgecolor='none', facecolor='red')#, label='test')
    ax.add_patch(rect_train)
    ax.add_patch(rect_test)
    ax.text(x=min_train + 0.5*(max_train-min_train), y=2+0.05, s=f"unused", ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x=min_test + 0.5*(max_test-min_test), y=2+0.05, s=f"unused", ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x=min_train + 0.5*(max_train-min_train), y=1, s=f"Training / Validation", ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x=min_test + 0.5*(max_test-min_test), y=1, s=f"Testing", ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x=min_train-0.3, y=2+0.05, s=f"Sensor 1-4", ha='right', va='center', fontsize=9)
    ax.text(x=min_train-0.3, y=1, s=f"Sensor 5-8", ha='right', va='center', fontsize=9)
    ax.set_yticks([])
    ax.set_ylim([0,3])
    ax.set_xlabel('Time (h)')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    plt.savefig(result_dir.joinpath(f'traintest_split_static.svg'))
    if show:
        plt.show()
    else:
        plt.close()