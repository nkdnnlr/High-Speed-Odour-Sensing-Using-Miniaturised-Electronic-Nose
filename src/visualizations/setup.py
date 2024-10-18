import sys, os
from pathlib import Path
home_path = "/Users/nd21aad/Phd/Projects/enose-analysis-crick"  # Define project root directory
os.chdir(home_path)         # Change working directory to project root directory
sys.path.append(home_path)  # Add project root directory to python path
home_path = Path(home_path)

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
pd.options.mode.chained_assignment = None  # Disable warning
import matplotlib as mpl
new_rc_params = {"text.usetex": False, "svg.fonttype": "none"}
mpl.rcParams.update(new_rc_params)
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns

from src.utils.constants import *
import src.temporal_structure.helpers as helpers

def plot_protocol(index_df_enose_all, result_dir_parent):
    """ 
    Plot the experimental protocol for the enose data

    :param index_df_enose_all: DataFrame containing the index of the enose data
    :param result_dir_parent: Path to the directory where the results should be saved
    :return: None
    """
    fig, ax = plt.subplots(figsize=(4, 2.5))
    t_min, t_max = 1000, 0
    for condition in ['Lcycle25msRcycle25ms', 'LconstRcycle25ms', 'LconstRcycle100ms']:
        index_df_condition = index_df_enose_all[index_df_enose_all['condition']==condition]
        # Extract start/end time of experiment and convert to hours
        t_start = pd.to_timedelta(index_df_condition['t_stimulus'].min()).total_seconds() / 3600
        t_end = pd.to_timedelta(index_df_condition['t_stimulus'].max()).total_seconds() / 3600
        if t_start < t_min:
            t_min = t_start
        if t_end > t_max:
            t_max = t_end
        # Add rectangles and text
        rect_L = patches.Rectangle((t_start, 1.2+0.5), t_end-t_start, 1, edgecolor='none', facecolor=experiment_colors[experiment_dict_L[condition]])#, label='train / val')
        rect_R = patches.Rectangle((t_start, 0.5), t_end-t_start, 1, edgecolor='none', facecolor=experiment_colors[experiment_dict_R[condition]])#, label='test')
        ax.add_patch(rect_L)
        ax.add_patch(rect_R)
        ax.text(x=t_start + 0.5*(t_end-t_start), y=1.2+0.5+0.5, s=experiment_dict_L[condition], ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x=t_start + 0.5*(t_end-t_start), y=0.5+0.5, s=experiment_dict_R[condition], ha='center', va='center', fontsize=9, fontweight='bold')
    # Add text
    ax.text(x=t_min-0.6, y=1.2+0.5+0.5, s=f"Sensor 1-4", ha='right', va='center', fontsize=9)
    ax.text(x=t_min-0.6, y=1, s=f"Sensor 5-8", ha='right', va='center', fontsize=9)
    # Format
    ax.set_xlim([t_min-0.2, t_max+0.1])
    ax.set_yticks([])
    ax.set_ylim([0,3])
    ax.set_xlabel('Time (h)')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    plt.savefig(result_dir_parent.joinpath(f'experiments.svg'))#, bbox_inches='tight')
    plt.show()
    plt.close()

def check_chi2(index_enose_pulses_all):
    """ 
    Perform a chi-squared test to check if the distribution of classes is uniform over time intervals

    :param index_enose_pulses_all: DataFrame containing the index of the enose data
    :return: contingency_table, chi2, p, num_bins      
    """
    data = index_enose_pulses_all
    # # Define the number of time intervals (bins)
    data['Time'] = pd.to_timedelta(data['t_stimulus']).dt.total_seconds()/3600
    n_hours = np.round(data['Time'].max() - data['Time'].min()).astype(int)
    num_bins = n_hours
    # Create intervals
    bin_edges = np.linspace(data['Time'].min(), data['Time'].max(), num_bins + 1)
    # Assign data points to intervals
    data['time_binned'] = pd.cut(data['Time'], bin_edges)
    # Create the contingency table (observed counts)
    contingency_table = pd.crosstab(data['gas1'], data['time_binned'], rownames=['Gas'], colnames=['Time'])
    # Perform the chi-squared test
    chi2, p, _, _ = chi2_contingency(contingency_table)
    return contingency_table, chi2, p, num_bins

def plot_contingencytable(index_enose_pulses_all, contingency_table, p, num_bins, result_dir_parent):
    """
    Plot the contingency table as a heatmap

    :param index_enose_pulses_all: DataFrame containing the index of the enose data
    :param contingency_table: Contingency table
    :param p: p-value of the chi-squared test
    :param num_bins: Number of time intervals
    :param result_dir_parent: Path to the directory where the results should be saved
    :return: None
    """
    data = index_enose_pulses_all
    # Create a heatmap from the contingency table
    plt.figure(figsize=(2, 2.5))
    heatmap = sns.heatmap(contingency_table, annot=False, cmap='YlGnBu', fmt='d')
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Count / h', rotation=270, labelpad=15)
    plt.xlabel('Time (h)')
    plt.ylabel('Class')
    n_ticks = 4
    x_ticks = np.linspace(0, num_bins, n_ticks)
    x_ticklabels = np.round(np.linspace(data['Time'].min(), data['Time'].max(), n_ticks), 0).astype(int)
    plt.xticks(x_ticks, x_ticklabels)
    plt.title(f"$\\chi^2$ Test: $p$ = {np.round(p, 3)}")
    plt.savefig(result_dir_parent.joinpath(f'experiments_heatmap.svg'))#, bbox_inches='tight')
    plt.show()

def plot_pid_onoffset(index_df_pid, data_dir_pid, result_dir_parent):
    """ 
    Plot the onset and offset of the PID signal for different gases

    :param index_df_pid: DataFrame containing the index of the PID data
    :param data_dir_pid: Path to the directory containing the PID data
    :param result_dir_parent: Path to the directory where the results should be saved
    :return: None    
    """
    index_pulse = index_df_pid.query(f"kind == 'pulse' & gas1 != 'b1' & gas1 != 'b2' & shape == '1.0s' & concentration == 100")#.iloc[0]
    fig, ax = plt.subplots(ncols=2, figsize=(5, 3))
    make_pid_plot = True
    for gas in ['IA', 'EB', 'Eu', '2H']:
        for i, index in enumerate(index_pulse.query(f"gas1 == '{gas}'").iterrows()):
            index = index[1]
            trial_id, kind = index['trial_id'], index['kind']
            df = pd.read_csv(data_dir_pid.joinpath('pid').joinpath(kind).joinpath(f"{trial_id}.csv"))
            df['pid_V'] = -df['pid_V']
            # Calculate noise floor
            t_noise_min = -1000
            t_noise_max = -10
            df_noise = df[(df['time_ms']>t_noise_min) & (df['time_ms']<t_noise_max)]
            noise_mean, noise_std = (df_noise['pid_V']).mean(), (df_noise['pid_V']).std()
            tol = 4
            # Compute onset, i.e. time of signal exceeding four sigma of noise floor
            t_onset_min = 0
            t_onset_max = 1000
            df_onset = df[(df['time_ms']>=t_onset_min) & (df['time_ms']<t_onset_max)]
            t_onset = df_onset[df_onset['pid_V']>noise_mean+tol*noise_std]['time_ms'].iloc[0]
            # Compute offset, i.e. time of signal falling below four sigma of noise floor
            t_offset_min = 1000
            t_offset_max = 5000
            df_offset = df[(df['time_ms']>=t_offset_min) & (df['time_ms']<t_offset_max)]
            t_offset = df_offset[df_offset['pid_V']<noise_mean+tol*noise_std]['time_ms'].iloc[0]
            # Plot data with detected threshold crossing
            if make_pid_plot:
                ax[0].plot(df['time_ms'], (df['pid_V']), label='pid', c='k')
                arrow_len = 0.002
                ax[0].arrow(t_onset, noise_mean+tol*noise_std+arrow_len*1.1, 0, -arrow_len, length_includes_head=True,
                head_width=150, head_length=arrow_len/10, color='r', zorder=100, linewidth=2)
                ax[0].arrow(t_offset, noise_mean+tol*noise_std+arrow_len*1.1, 0, -arrow_len, length_includes_head=True,
                head_width=150, head_length=arrow_len/10, color='r', zorder=100, linewidth=2)
                ax[0].axhline(noise_mean, c='gray', ls='--', linewidth=0.5)
                ax[0].axhline(noise_mean+tol*noise_std, c='r', ls='--', linewidth=0.5)
                ax[0].spines[['top', 'right']].set_visible(False)
                ax[0].set_xlabel('Time (ms)')
                ax[0].set_ylabel('PID (V)')
                ylims = ax[0].get_ylim()
                ax[0].axvspan(t_onset_min, t_offset_min, alpha=0.2, facecolor='purple')
                ax[0].set_title(gas)
                make_pid_plot = False
            # Scatter plot of onset and offset
            ax[1].scatter(t_onset, t_offset-t_offset_min, color=gases_colors[gas], label=gas if i==0 else '', s=10)
            ax[1].set_xlabel('Onset (ms)')
            ax[1].set_ylabel('Offset (ms)')
            ax[1].spines[['top', 'right']].set_visible(False)
            ax[1].legend()
    plt.tight_layout()
    plt.savefig(result_dir_parent.joinpath(f"pid_onset_offset.svg"), bbox_inches='tight')
    plt.show()


def plot_pid_samples(index_df_pid, data_dir_pid, result_dir_parent, t_min=-500, t_max=1500):
    """
    Plot samples of the PID signal for different gases

    :param index_df_pid: DataFrame containing the index of the PID data
    :param data_dir_pid: Path to the directory containing the PID data
    :param result_dir_parent: Path to the directory where the results should be saved
    :param t_min: Minimum time to plot
    :param t_max: Maximum time to plot
    :return: None
    """
    # Get indices
    index_pulse_b1 = index_df_pid.query(f"kind == 'pulse' & gas1 == 'b1' & shape == '1.0s' & concentration == 100").iloc[0]
    index_pulse_IA = index_df_pid.query(f"kind == 'pulse' & gas1 == 'IA' & shape == '1.0s' & concentration == 100").iloc[0]
    index_pulse_EB = index_df_pid.query(f"kind == 'pulse' & gas1 == 'EB' & shape == '1.0s' & concentration == 100").iloc[0]
    index_acorr_IA_EB = index_df_pid.query(f"kind == 'acorr' & gas1 == 'IA' & gas2 == 'EB'  & shape == '20Hz' & concentration == 100").iloc[0]
    fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey='row', figsize=(6, 2))
    index = index_pulse_b1
    trial_id, kind = index['trial_id'], index['kind']
    df = pd.read_csv(data_dir_pid.joinpath('pid').joinpath(kind).joinpath(f"{trial_id}.csv"))
    df = df[(df['time_ms']>t_min) & (df['time_ms']<t_max)]
    ax[0,0].plot(df['time_ms'], df['b_comp'], label='b_comp', c=colors_dict['b_comp'])
    ax[0,0].plot(df['time_ms'], df['b1'], label='b1', c=colors_dict['b1'])
    ax[1,0].plot(df['time_ms'], -(df['pid_V']-df['pid_V'].iloc[0]), label='pid', c='k')

    index = index_pulse_IA
    trial_id, kind = index['trial_id'], index['kind']
    df = pd.read_csv(data_dir_pid.joinpath('pid').joinpath(kind).joinpath(f"{trial_id}.csv"))
    df = df[(df['time_ms']>t_min) & (df['time_ms']<t_max)]
    ax[0,1].plot(df['time_ms'], df['b_comp'], label='b_comp', c=colors_dict['b_comp'])
    ax[0,1].plot(df['time_ms'], df['IA'], label='IA', c=colors_dict['IA'])
    ax[1,1].plot(df['time_ms'], -(df['pid_V']-df['pid_V'].iloc[0]), label='pid', c='k')

    index = index_pulse_EB
    trial_id, kind = index['trial_id'], index['kind']
    df = pd.read_csv(data_dir_pid.joinpath('pid').joinpath(kind).joinpath(f"{trial_id}.csv"))
    df = df[(df['time_ms']>t_min) & (df['time_ms']<t_max)]
    ax[0,2].plot(df['time_ms'], df['b_comp'], label='b_comp', c=colors_dict['b_comp'])
    ax[0,2].plot(df['time_ms'], df['EB'], label='EB', c=colors_dict['EB'])
    ax[1,2].plot(df['time_ms'], -(df['pid_V']-df['pid_V'].iloc[0]), label='pid', c='k')

    index = index_acorr_IA_EB
    trial_id, kind = index['trial_id'], index['kind']
    df = pd.read_csv(data_dir_pid.joinpath('pid').joinpath(kind).joinpath(f"{trial_id}.csv"))
    df = df[(df['time_ms']>t_min) & (df['time_ms']<t_max)]
    ax[0,3].plot(df['time_ms'], df['b_comp'], label='b_comp', c=colors_dict['b_comp'])
    ax[0,3].plot(df['time_ms'], df['IA'], label='IA', c=colors_dict['IA'])
    ax[0,3].plot(df['time_ms'], df['EB'], label='EB', c=colors_dict['EB'])
    ax[1,3].plot(df['time_ms'], -(df['pid_V']-df['pid_V'].iloc[0]), label='pid', c='k')

    ax[0,0].set_ylabel('Valves')
    ax[1,0].set_ylabel('PID')
    ax[1,0].set_xlabel('Time (ms)')
    ax[1,1].set_xlabel('Time (ms)')
    ax[1,2].set_xlabel('Time (ms)')
    ax[1,3].set_xlabel('Time (ms)')
    ax[0,0].set_title('blank')
    ax[0,1].set_title('IA')
    ax[0,2].set_title('EB')
    ax[0,3].set_title('IA-EB, 20Hz')

    for _ax in ax.flatten():
        _ax.spines[['top', 'right', 'left']].set_visible(False)
        _ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(result_dir_parent.joinpath(f"pid_samples.svg"), bbox_inches='tight')
    plt.show()
    plt.close()

def plot_enose_cycle(index_df_enose, data_dir_enose, result_dir_parent, ms_start=20010, ms_end=20010+50, kind='pulse', gas='IA', shape='1.0s', concentration=100, sensor=5):
    """
    Plot the enose data for a single cycle

    :param index_df_enose: DataFrame containing the index of the enose data
    :param data_dir_enose: Path to the directory containing the enose data
    :param result_dir_parent: Path to the directory where the results should be saved
    :param ms_start: Start time of the plot
    :param ms_end: End time of the plot
    :param kind: Type of experiment
    :param gas: Gas used in the experiment
    :param shape: Shape of the stimulus
    :param concentration: Concentration of the stimulus
    :param sensor: Sensor to plot
    :return: None
    """
    color_y1 = '#FF6600'
    color_y2 = '#FF00FF'
    color_y2 = '#00FF00'
    trial = index_df_enose.query(f"kind == '{kind}' & gas1 == '{gas}' & shape == '{shape}' & concentration == {concentration}").iloc[0]
    trial_df = helpers.get_trial_df(trial, data_dir_enose, ms_start, ms_end)
    fig, ax = plt.subplots(figsize=(1.25, 1.25))
    # Plot
    ax_twin = ax.twinx()
    ax.plot(trial_df['time_ms']-ms_start, trial_df[f'T_heat_{sensor}'], label='T_heat_5', c=color_y1, zorder=10, linewidth=2)
    ax_twin.plot(trial_df['time_ms']-ms_start, 1./(trial_df[f'R_gas_{sensor}']/1e6), label='R_gas_5', c=color_y2, zorder=1, linewidth=2)
    # Format
    ax.set_zorder(ax_twin.get_zorder()+1)
    ax.patch.set_visible(False)
    ax.set_ylabel(r'$T_{heater}\ (\circ \mathrm{C})$')
    ax_twin.set_ylabel(r'$R_{sensor}\ (M\Omega$)')
    ax_twin.set_ylabel(r'$G_{sensor}\ (\mu S$)')
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([200,300,400])
    ax.set_xticks([0,25,50])
    ax_twin.set_yticks([0, 5, 10])
    # Make horizontal red bar at the top of the plot, for the first 25ms
    ax.axvspan(0, 25, alpha=0.2, facecolor='red', zorder=0)
    # Add text above the red bar
    ax.text(12.5, 425, r'$P_{heater}$', ha='center', va='center', fontsize=9, fontweight='bold', color='red', alpha=0.6)
    # set ax label fontcolors
    ax.yaxis.label.set_color(color_y1)
    ax_twin.yaxis.label.set_color(color_y2)
    # set ax tick colors
    ax.tick_params(axis='y', colors=color_y1)
    ax_twin.tick_params(axis='y', colors=color_y2)
    # set ax colors
    ax.spines['left'].set_color(color_y1)
    ax_twin.spines['right'].set_color(color_y2)
    # Make top & right axis invisible
    ax.spines[['top', 'right']].set_visible(False)
    ax_twin.spines[['top', 'left']].set_visible(False)
    # ax[1].spines[['right', 'top']].set_visible(False)
    plt.savefig(result_dir_parent.joinpath(f"set_temp_resistance_new.svg"), bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_T_R(index_df_enose, data_dir_enose, result_dir_parent, ms_start=20010, ms_end=20010+50, kind='pulse', gas='IA', shape='1.0s', concentration=100):
    """
    Plot the enose data for a single cycle (Temperature vs Resistance)

    :param index_df_enose: DataFrame containing the index of the enose data
    :param data_dir_enose: Path to the directory containing the enose data
    :param result_dir_parent: Path to the directory where the results should be saved
    :param ms_start: Start time of the plot
    :param ms_end: End time of the plot
    :param kind: Type of experiment
    :param gas: Gas used in the experiment
    :param shape: Shape of the stimulus
    :param concentration: Concentration of the stimulus
    :return: None
    """
    # Extract data
    trial = index_df_enose.query(f"kind == '{kind}' & gas1 == '{gas}' & shape == '{shape}' & concentration == {concentration}").iloc[0]
    trial_df = helpers.get_trial_df(trial, data_dir_enose, ms_start, ms_end)
    # Create plot
    fig, ax = plt.subplots(figsize=(3, 3))
    for i, sensor in enumerate([5,6,7,8]):
        c = ax.scatter(trial_df[f'T_heat_{sensor}'], trial_df[f'R_gas_{sensor}'], s=10, c=sensor_colors[i], label=sensor_names_short[i])
    # Formatting
    ax.set_yscale('log')
    ax.set_xlabel(r'$T_{heater}\ (\circ \mathrm{C})$')#, usetex=True, va='top')
    ax.set_ylabel(r'$R_{sensor}\ (\Omega$)')#, usetex=True,)
    ax.grid()
    # Set upper and right spine invisible
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    # Save and show
    plt.savefig(result_dir_parent.joinpath(f"tr_scatter.svg"), bbox_inches='tight')
    plt.tight_layout()
    plt.show()