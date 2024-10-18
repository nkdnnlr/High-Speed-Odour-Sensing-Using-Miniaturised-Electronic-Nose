import matplotlib as mpl
import matplotlib.pyplot as plt
new_rc_params = {"text.usetex": False, "svg.fonttype": "none"}
mpl.rcParams.update(new_rc_params)
import pandas as pd
import importlib

parent_package = '.'.join(__package__.split('.')[:-1])
constants = importlib.import_module(f'{parent_package}.utils.constants')

def plot_enose_trial_df(trial_df, trial_id):
    """
    Plot the trial data for the eNose sensors and gas concentrations

    :param trial_df: DataFrame with the trial data
    :param trial_id: ID of the trial
    """
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(10,6))
    for i in range(1,5):
        ax[0].plot(trial_df['time_ms'], trial_df[f'R_gas_{i}'], label=constants.sensor_names[i-1], c=constants.sensor_colors[i-1])
        ax[1].plot(trial_df['time_ms'], trial_df[f'R_gas_{i+4}'], label=constants.sensor_names[i+3], c=constants.sensor_colors[i-1])
    for gas, zorder in zip(['EB', 'IA', 'Eu', '2H', 'b1', 'b2', 'b_comp'], [10,10,10,10, 2,2,1]):
        ax[2].plot(trial_df['time_ms'], trial_df[gas], label=gas, zorder=zorder, c=constants.gases_colors[gas])
    # Formatting
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].legend(loc='center right')
    ax[1].legend(loc='center right')
    ax[2].legend(loc='center right', ncol=2)
    for _ax in ax:
        _ax.grid()
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)
    ax[0].set_ylabel(r'$R_{sensor}\ (\Omega)$')
    ax[1].set_ylabel(r'$R_{sensor}\ (\Omega)$')
    ax[2].set_ylabel('Valve control')
    ax[-1].set_xlabel('Time (ms)')
    plt.suptitle(trial_id)
    plt.show()

def plot_pid_trial_df(trial_df, trial_id):
    """
    Plot the trial data for the PID sensor and gas concentrations

    :param trial_df: DataFrame with the trial data
    :param trial_id: ID of the trial
    """
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10,4))
    ax[0].plot(trial_df['time_ms'], -1*trial_df[f'pid_V'], c='k')
    for gas, zorder in zip(['EB', 'IA', 'Eu', '2H', 'b1', 'b2', 'b_comp'], [10,10,10,10, 2,2,1]):
        ax[1].plot(trial_df['time_ms'], trial_df[gas], label=gas, zorder=zorder, c=constants.gases_colors[gas])
    # Formatting
    ax[0].grid()
    ax[1].grid()
    ax[1].legend(loc='center right', ncol=2)
    ax[0].set_ylabel('PID (V)')
    ax[1].set_ylabel('Valve control')
    ax[-1].set_xlabel('Time (ms)')
    plt.suptitle(trial_id)
    plt.show()