from pathlib import Path
import json
import pickle
import os
from turtle import left
import numpy as np
from parameters import GLIF_dict, folders, relative_path, ctc
from GLIF_Teeter_et_al_2018.libraries.data_library import get_sweep_num_by_name
from utilities import check_nan_arrays_equal, plot_results_and_diff
from allen_simulation import load_model_config_stimulus



def GLIF_5_refactored(specimen_id, model_type):
    
    # Load parameters from Allen model
    saved_model, config, stimulus = load_model_config_stimulus(specimen_id, model_type)

    num_steps = len(stimulus)
    V = np.empty(num_steps) 
    V[:] = np.nan
    V_init = config['init_voltage']
    dt = config['dt']
    G = 1/config['R_input'] * config["coeffs"]["G"]
    El = config['El']
    C = config['C'] * config["coeffs"]["C"]
    thres = config['th_inf'] * config['coeffs']['th_inf']
    spike_cut_length = config['spike_cut_length']

    T = np.empty(num_steps)
    T[:] = np.nan
    init_threshold = config['init_threshold']
    a_spike = config['threshold_dynamics_method']["params"]["a_spike"]
    b_spike = config['threshold_dynamics_method']["params"]["b_spike"]
    a = config['voltage_reset_method']["params"]["a"]
    b = config['voltage_reset_method']["params"]["b"]
    tcs_spike_component = np.empty(num_steps)
    tcs_spike_component[:] = np.nan
    tcs_voltage_component = np.empty(num_steps)
    tcs_voltage_component[:] = np.nan
    a_voltage = config['threshold_dynamics_method']["params"]["a_voltage"] * config["coeffs"]["a"]
    b_voltage = config['threshold_dynamics_method']["params"]["b_voltage"] * config["coeffs"]["b"]

    voltage_t0 = V_init
    threshold_t0 = init_threshold
    ASCurrents_t0 = np.array(config['init_AScurrents'])
    th_spike = 0
    th_voltage = 0

    num_AScurrents = len(ASCurrents_t0)
    A = np.empty((num_steps, num_AScurrents))
    A[:] = np.nan
    asc_amp_array = np.array(config['asc_amp_array']) * np.array(config["coeffs"]['asc_amp_array'])
    k = 1 / np.array(config['asc_tau_array'])
    r = np.array(config['AScurrent_reset_method']["params"]["r"])

    i = 0
    while i< num_steps:
        
        # Voltage
        inj = stimulus[i]
        voltage_t1 = voltage_t0 + (inj + np.sum(ASCurrents_t0) - G * (voltage_t0 - El))*dt/C # Linear euler equation

        # Threshold
        if i == 0:
            th_spike = 0
            th_voltage = 0
        else:
            th_spike = tcs_spike_component[i-1]
            th_voltage = tcs_voltage_component[i-1]

        # Spike component
        b_spike_neg=-b_spike
        spike_component = th_spike*np.exp(b_spike_neg * dt)  # Exact
        tcs_spike_component[i] = spike_component
        
        # Voltage component
        I = inj + np.sum(ASCurrents_t0)
        beta = (I+G*El)/G
        phi=a_voltage/(b_voltage-G/C)
        voltage_component = phi*(voltage_t0-beta)*np.exp(-G*dt/C)+1/(np.exp(b_voltage*dt))*(th_voltage-phi*(voltage_t0-beta)-(a_voltage/b_voltage)*(beta-El)-0) +(a_voltage/b_voltage)*(beta-El) +0
        tcs_voltage_component[i] = voltage_component


        threshold_t1 = spike_component + voltage_component + thres  # TODO: This is different from white paper

        # ASCurrents
        ASCurrents_t1 = ASCurrents_t0*np.exp(-k * dt)
        
        # Reset if spike occured (with refractory period)
        if voltage_t1 > threshold_t1:

            # Voltage Reset
            voltage_t0 = El + a*(voltage_t1-El) + b  

            # Voltage component of threshold reset
            th_voltage = tcs_voltage_component[i]
            tcs_voltage_component[i+1:i+spike_cut_length+1] = np.ones(spike_cut_length)*th_voltage

            # TODO: This would error if a spike occurs when i > num_steps - spike_cut_length
            th_spike = tcs_spike_component[i]
            t = np.arange(1,spike_cut_length+1)*dt
            b_spike_neg = -b_spike
            spike_comp_decay = th_spike * np.exp(b_spike_neg * t)  # Decay spike component during refractory period
            tcs_spike_component[i+1:i+spike_cut_length+1] = spike_comp_decay
            tcs_spike_component[i+spike_cut_length] += a_spike  # a_spike = delta_theta_s; add additive component
            threshold_t0 = tcs_spike_component[i+spike_cut_length] +tcs_voltage_component[i+spike_cut_length] + thres

            # ASCurrents reset
            new_currents = asc_amp_array
            left_over_currents = ASCurrents_t1 * r * np.exp(-(k * dt * spike_cut_length))
            ASCurrents_t0 = new_currents + left_over_currents

            # Extend this to refractory period
            refractory_beyond_end = i + spike_cut_length >= num_steps
            if refractory_beyond_end == True:
                # Values remain as np.nan
                i = num_steps
            else:
                V[i+spike_cut_length] = voltage_t0
                T[i+spike_cut_length] = threshold_t0
                A[i+spike_cut_length, :] = ASCurrents_t0
                i += spike_cut_length + 1

        else:
            V[i] = voltage_t1
            T[i] = threshold_t1
            A[i, :] = ASCurrents_t1

            voltage_t0 = voltage_t1
            threshold_t0 = threshold_t1
            ASCurrents_t0 = ASCurrents_t1
            
            i+= 1
    return saved_model, V, T, A

if __name__ == "__main__":

    specimen_ids = [474637203]#, 512322162]
    model_types = ['LIFRASCAT_model']

    for specimen_id in specimen_ids:
        for model_type in model_types:
        
            saved_model, V, T, A= GLIF_5_refactored(specimen_id, model_type)

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
            python = T[mask]*1e3
            result = check_nan_arrays_equal(Allen, python)
            print("Are threshold results equal: {}".format(result))
            plot_results_and_diff(Allen, "Allen", python, "python", t[mask])

            # Plot ASCurrents
            Allen = saved_model['AScurrents'][mask]*1e3
            python = A[mask, :]*1e3
            result = check_nan_arrays_equal(Allen, python)
            print("Are threshold results equal: {}".format(result))
            plot_results_and_diff(Allen, "Allen", python, "python", t[mask])