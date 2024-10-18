import numpy as np

gases_all = ["IA", "Eu", "EB", "2H", "blank"]
width = "1.0s"  # of pulse
pulse = 1000 if width == "1.0s" else 100  # ms, or samples
period = 50  # samples (ms) 
fs = 1000

endpoints, midpoint = False, True   
colors = ["r", "g", "b", "y", "gray"] 
colors_dict = {
    'IA': 'r',
    'EB': 'b',
    'Eu': 'g',
    '2H': 'k',
    'b1': 'dimgray',
    'b_comp': 'lightgray',
}

# Dividing stimulus in cycles
n_before = 2
n_pulse = int(np.ceil(pulse / period))-2  # number of cycles per pulse
n_between = 2  # number of cycles between pulse and blank (for classifier)
n_blank = 22  # number of cycles per blank

cycles_gas = np.arange(n_before, n_before+n_pulse).tolist()
cycles_blank = np.arange(n_before+n_pulse + n_between, n_pulse + n_between + n_blank).tolist() 
all_cycles = np.arange(0, cycles_blank[-1]).tolist() 
n_all_cycles = len(all_cycles) 

delay = 10 
on_buffer = 500
off_buffer = 500
tgas_start = 0
tgas_end = 1000
period = 50
t_end = 2000

string_noblank = "gas1 != 'blank' & gas2 != 'blank'"

gases_colors = {
    "blank": 'darkgray',
    "EB": '#e41a1c',
    "2H": '#377eb8',
    "IA": '#4daf4a',
    "Eu": '#984ea3',
} 

gases_colors = {
    "EB": '#e41a1c',            # Ethyl Butyrate
    "2H": '#377eb8',            # 2-Heptanone
    "IA": '#4daf4a',            # Isoamyl Acetate
    "Eu": '#984ea3',            # Eucalyptol
    "b1": 'dimgray',            # blank valve 1
    "b2": 'gray',               # blank valve 2
    "b_comp": 'darkgray',       # blank compensation
    "blank": 'darkgray',
    } 


sensor_names = [
    "RED_A",
    "OX_A",
    "NH3_A",
    "VOC_A",
    "RED_B",
    "OX_B",
    "NH3_B",
    "VOC_B",    
]

sensor_names_short = [
    "RED",
    "OX",
    "NH3",
    "VOC",   
]

sensor_colors = {
    0: '#1b9e77',
    1: '#d95f02',
    2: '#7570b3',
    3: '#e7298a',
}

heater_colors = {
    0: '#fef0d9',
    1: '#fdcc8a',
    2: '#fc8d59',
    3: '#d7301f',
}


pattern_spelledout = {
    'corr': 'correlated',
    'acor': 'anti-correlated',
    'acorr': 'anti-correlated',
}

experiment_dict_L = {
    'Lcycle25msRcycle25ms': '25ms cycle',
    'LconstRcycle25ms': 'constant',
    'LconstRcycle100ms': 'constant',
}

experiment_dict_R = {
    'Lcycle25msRcycle25ms': '25ms cycle',
    'LconstRcycle25ms': '25ms cycle',
    'LconstRcycle100ms': '100ms cycle',
}

experiment_colors = {
    '25ms cycle': '#636363',
    '100ms cycle': '#969696',
    'constant': '#cccccc'
}