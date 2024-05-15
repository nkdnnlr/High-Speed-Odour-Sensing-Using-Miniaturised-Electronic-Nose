# High-speed odour sensing using miniaturised electronic nose

[https://doi.org/10.5061/dryad.pg4f4qrxz](https://doi.org/10.5061/dryad.pg4f4qrxz)

The data comprises of recordings of complex odour stimuli using a high-speed a miniaturised electronic nose (e-nose). The odour stimuli were presented with a custom and high-fidelity olfactometer that was most prominently used in a recent mammalian olfaction study [1]. A photo-ionisation detector (PID) was used for ground-truth recordings and performance evaluations.

Details about the experimental protocol and the specificities of the used devices can be found in the manuscript.

## Description of the data and file structure

This data repository consists of a nested directory structure (sensor modality - sensor mode - stimuli type), containing a total of 4875 individual e-nose and 2670 PID recording trials. Each set of experiments is accompanied with an `index.csv` file, which specifies the trial names and experimental conditions, thus allowing for fast access of the trials-of-interest.

#### E-nose

The e-nose recording trials are saved as comma-separated values (CSV), containing the following columns:
`index`: Index column, unique across all experiments \
`timestamp`: Timestamp from device boot\
`time_s`: Time (s) with respect to odour onset\
`time_ms`: Time (ms) with respect to odour onset\
`control_cycle_step_left`: Heater power phase for sensor 1-4 (for cycles only)\
`control_cycle_step_right`: Heater power phase for sensor 1-4 (for cycles only)\
`R_gas_1`-`R_gas_8`: Gas sensor resistance values, in Ohm\
`T_heat_1`-`T_heat_8`: Sensor heater temperature, in degree Celsius\
`p_mbar`: Ambient pressure, in millibar\
`t_celsius`: Ambient temperature, in degree Celsius\
`rh_percent`: Ambient relative humidity, in percent\
`EB`, `IA`, `Eu`, `2H`, `b1`, `b2`, `b_comp`: Flow valve control values for the different odourants\
`total`: Sum of all flow valve control values, should add up to 1.0 at any time

#### PID

The PID recording trials are saved as comma-separated values (CSV), containing the following columns:
`index`: Index column, unique across all experiments \
`time_ms`: Time (ms) with respect to odour onset\
`timestamp`: Timestamp from device boot\
`pid_V`: PID output, in Volts\
`EB`, `IA`, `Eu`, `2H`, `b1`, `b2`, `b_comp`: Flow valve control values for the different odourants\
`total`: Sum of all flow valve control values, should add up to 1 at any time

## Code/Software

For loading and displaying the data, please refer to the code in the following repository:
[https://github.com/nkdnnlr/High-Speed-Odour-Sensing-Using-Miniaturised-Electronic-Nose](https://github.com/nkdnnlr/High-Speed-Odour-Sensing-Using-Miniaturised-Electronic-Nose)
In particular, run the jupyter notebook called [00_display_data.ipynb](https://github.com/nkdnnlr/High-Speed-Odour-Sensing-Using-Miniaturised-Electronic-Nose/blob/main/00_display_data.ipynb), which will guide you through the necessary steps.
