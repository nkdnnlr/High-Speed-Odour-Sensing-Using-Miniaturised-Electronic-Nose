# import numpy as np

# # Run settings
# test_gas_pairs = False  # True
# visualize_features = False  # True
# pca_analysis = True

# # Setting parameters, such as gases, colors, sensors, normalisation mode, pulse width
# conditions = {
#     50: "cyclelow2high_25ms",
#     # 50: "cyclelow2highleft_constantright__t_high_25ms"
# }


# skip_first = 0  # How many samples to skip for each experiment
# width_acorr = "1Hz" # Correlation frequency (divide by two for 'actual' frequency)

# gases_all = ["IA", "Eu", "EB", "2H", "blank"]#, "b2"]
# width = "1.0s"  # of pulse
# pulse = 1000 if width == "1.0s" else 100  # ms, or samples
# period = 50  # samples (ms)
# fs = 1000

# condition = conditions[period]
# condition_cycles_cycles = 'cyclelow2high_25ms'
# condition_constant_cycles = 'cyclelow2highleft_constantright__t_high_25ms'

# # endpoints, midpoint = True, True
# endpoints, midpoint = False, True   # Seems to work best for short pulses!
# colors = ["r", "g", "b", "y", "gray"]

# # Dividing stimulus in cycles
# safety_margin = True
# if safety_margin:
#     # with safety margin:
#     n_before = 2
#     n_pulse = int(np.ceil(pulse / period))-2  # number of cycles per pulse
#     n_between = 2  # number of cycles between pulse and blank (for classifier)
#     n_blank = 22  # number of cycles per blank
# else:
#     # without safety margin:
#     n_before = 1
#     n_pulse = int(np.ceil(pulse / period))-1  # number of cycles per pulse
#     n_between = 1  # number of cycles between pulse and blank (for classifier)
#     n_blank = 22  # number of cycles per blank

# cycles_gas = np.arange(n_before, n_before+n_pulse).tolist()
# cycles_blank = np.arange(n_before+n_pulse + n_between, n_pulse + n_between + n_blank).tolist()  #
# print(cycles_gas, cycles_blank)

# all_cycles = np.arange(0, cycles_blank[-1]).tolist()
# n_all_cycles = len(all_cycles)

# # n_after = n_pulse + 10  # number of cycles after pulse



# # What's happening here?
# # cycle_indices = np.arange(2, n_pulse+1, n_per_col)
# # cycle_indices = [0, 1, *cycle_indices, n_pulse+1, n_pulse+2, n_pulse+3]
# # cycle_indices = np.arange(0, n_after + 2)
# # # if just_start:
# # #     cycle_indices = np.arange(0, 5)
# # ncycles = len(cycle_indices) - 1
# print("Constants:", pulse, period, n_pulse,  all_cycles, n_all_cycles)

# gases = gases_all
# print(gases)

# string_noblank = "gas1 != 'blank' & gas2 != 'blank'"


# f_test = 0.2

# t_before = 2 # seconds * np.timedelta64(1, "s"),
# t_after = 10 # seconds * np.timedelta64(1, "s")


# delay = 10
# on_buffer = 500#49
# off_buffer = 500
# tgas_start = 0
# tgas_end = 1000
# period = 50
# t_end = 2000

# # gases_colors = {
# #     "blank": "dimgray",
# #     "EB": "green",
# #     "2H": "blue",
# #     "IA": "purple",
# #     "Eu": "red",
# # }

# # gases_colors = {
# #     "blank": '#786E60',
# #     "EB": '#EEC34B',
# #     "2H": '#37D2AB',
# #     "IA": '#AFA8E4',
# #     "Eu": '#F68070',
# # }

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

# sensor_names_short = [
#     "RED",
#     "OX",
#     "NH3",
#     "VOC",   
# ]

# # sensor_colors = {
# #     0: '#a6cee3',
# #     1: '#1f78b4',
# #     2: '#b2df8a',
# #     3: '#33a02c',
# # }

sensor_colors = {
    0: '#1b9e77',
    1: '#d95f02',
    2: '#7570b3',
    3: '#e7298a',
}

# heater_colors = {
#     0: '#fef0d9',
#     1: '#fdcc8a',
#     2: '#fc8d59',
#     3: '#d7301f',
# }

# correction_factors = {
#     0: 27/33,
#     1: 6.8/10,
#     2: 27/33,
#     3: 47/56,
#     4: 27/33,
#     5: 6.8/10,
#     6: 27/33,
#     7: 47/56
# }

# actual_gasnames = {
#     "blank": "blank",
#     "EB": "Ethyl Butyrate",
#     "2H": "2-Heptanone",
#     "IA": "Isoamyl Acetate",
#     "Eu": "Cineole",
# }

# pattern_spelledout = {
#     'corr': 'correlated',
#     'acor': 'anti-correlated',
#     'acorr': 'anti-correlated',
# }

#     # colors = ['#EEC34B','#37D2AB','#AFA8E4','#F68070', ]

# #EEC34B,#37D2AB,#AFA8E4,#F68070

# # dataset_old = True
# # # dataset_old = False
# # dataloader_old = False  # Make sure to use correct virtual environment! currently either 'damien' or 'loader_old'

# # if not dataset_old:
# #     conditions = {
# #         150: "cyclelow2high_75ms",
# #         # 100:    "heatmod_t_high_50ms",
# #         50: "cyclelow2high_25ms",
# #     }
# #     width_acorr = "5Hz"  # of corr/acorr pattern

# # else:
# #     conditions = {
# #         150: "heatmod_t_high_75ms",
# #         100: "heatmod_t_high_50ms",
# #         50: "heatmod_t_high_25ms",
# #     }
# #     width_acorr = "2Hz"  # of corr/acorr pattern


# # # Setting parameters, such as gases, colors, sensors, normalisation mode, pulse width
# # # gases = ["EB", "IA", "Eu", "2H", "B1"]
# # # colors = ["r", "g", "b", "k", "y"]
# # gases = ["EB", "IA", "Eu", "B1"]
# # colors = ["r", "g", "b", "y"]
# # width = "1.0s"  # of pulse
# # pulse = 1000 if width == "1.0s" else 100  # ms
# # # period = 150
# # period = 50
# # condition = conditions[period]
# # endpoints, midpoint = True, True
# # just_start = False
# # fnorm = partial(norm01, endpoints=endpoints, midpoint=midpoint)
# # skip_first = 0  # 50

# # # Dividing stimulus in cycles
# # n_pulse = int(
# #     np.ceil(pulse / period)
# # )  # number of cycles per pulse (relevant for plotting)
# # n_blank = 5  # number of cycles per blank (relevant for plotting)
# # n_after = n_pulse + 10  # number of cycles after pulse
# # n_between = 1  # number of cycles between pulse and blank (for classifier)
# # cycles_gas = np.arange(1, n_pulse)
# # cycles_blank = np.arange(n_pulse + n_between, n_pulse + n_between + n_blank)  #
# # print(cycles_gas, cycles_blank)

# # # What's happening here?
# # # cycle_indices = np.arange(2, n_pulse+1, n_per_col)
# # # cycle_indices = [0, 1, *cycle_indices, n_pulse+1, n_pulse+2, n_pulse+3]
# # cycle_indices = np.arange(0, n_after + 2)
# # if just_start:
# #     cycle_indices = np.arange(0, 5)
# # ncols = len(cycle_indices) - 1
# # ncycles = ncols
# # print(ncycles)

# # all_sensors_train = [[1], [2], [3], [4], [5], [6], [7], [8], [1,2,3,4], [5,6,7,8], [1,2,3,4,5,6,7,8]]
# # all_sensors_test_b1 = [[1], [2], [3], [4], [5], [6], [7], [8], [1,2,3,4], [5,6,7,8], [1,2,3,4,5,6,7,8]]
# # all_sensors_train = [[1], [1,2,3,4,5,6,7,8]]
# # all_sensors_test_b1 = [[1], [1,2,3,4,5,6,7,8]]
# # all_sensors_train = [[5], [6], [7], [8], [5,6,7,8]]
# # all_sensors_test_b1 = [[5], [6], [7], [8], [5,6,7,8]]
# # all_sensors_train = [[5,6,7,8]]#, [1,2,3,4,5,6,7,8]]
# # all_sensors_test_b1 = [[5,6,7,8]]#, [1,2,3,4,5,6,7,8]]


# featurecomparison = {
#     't_start': -1000,
#     'tgas_start': 0,
#     'tgas_end': 1000,
#     't_stop': 3000,
# }