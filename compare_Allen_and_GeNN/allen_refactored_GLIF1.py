from pathlib import Path
import json
import pickle
import os
import numpy as np
from parameters import GLIF_dict, folders, relative_path, ctc
from GLIF_Teeter_et_al_2018.libraries.data_library import get_sweep_num_by_name
from utilities import check_nan_arrays_equal, plot_results_and_diff
from allen_simulation import load_model_config_stimulus



def GLIF_1_refactored(specimen_id, model_type):
    
    # Load parameters from Allen model
    saved_model, config, stimulus = load_model_config_stimulus(specimen_id, model_type)

    num_steps = len(stimulus)
    V = np.empty(num_steps) 
    V[:] = np.nan
    V_init = config['init_voltage']
    dt = config['dt']
    G = 1/config['R_input']
    El = config['El']
    C = config['C']
    thres = config['th_inf'] * config['coeffs']['th_inf']
    spike_cut_length = config['spike_cut_length']
    i = 0
    while i< num_steps:
        
        if i == 0:
            voltage_t1 = V_init
        else:

            # Get relevant parameters
            voltage_t0 = V[i-1]
            inj = stimulus[i]
            
            # Linear euler equation
            voltage_t1 = voltage_t0 + (inj - G * (voltage_t0 - El))*dt/C

        # Reset if spike occured (with refractory period)
        if voltage_t1 > thres:

            # Reset
            voltage_t1 = V_init  # Teeter does this with voltage_t0

            # Extend this to refractory period
            refractory_beyond_end = i + spike_cut_length >= num_steps
            if refractory_beyond_end == True:
                i = num_steps
            else:
                V[i+spike_cut_length] = V_init
                i += spike_cut_length + 1

        else:
            V[i] = voltage_t1
            i+= 1

    return saved_model, V, thres

if __name__ == "__main__":

    specimen_ids = [474637203]#, 512322162]
    model_types = ['LIF_model'] #, 'LIFR_model', 'LIFASC_model', 'LIFRASC_model', 'LIFRASCAT_model']

    for specimen_id in specimen_ids:
        for model_type in model_types:
        
            saved_model, V, thres = GLIF_1_refactored(specimen_id, model_type)

            t = saved_model['time']
            mask = np.logical_and(t>18, t<18.3)
            t_mask = t[mask]

            # Plot voltages
            Allen = saved_model['voltage'][mask]*1e3
            python = V[mask].ravel()*1e3
            result = check_nan_arrays_equal(Allen, python)
            print("Are voltage results equal: {}".format(result))
            plot_results_and_diff(Allen, "Allen", python, "python", t[mask])

            # Plot thresholds
            Allen = saved_model['threshold'][mask]*1e3
            python = thres*np.ones(Allen.shape)*1e3
            result = check_nan_arrays_equal(Allen, python)
            print("Are threshold results equal: {}".format(result))
            plot_results_and_diff(Allen, "Allen", python, "python", t[mask])