import sys
import pickle

sys.path.append("..")  # TODO: Come up with a better way to handle the imports
from pygenn.genn_model import create_custom_current_source_class, GeNNModel
from allen_simulation import load_model_config_stimulus
from GLIF_models import GLIF1, GLIF2, GLIF3, GLIF4, GLIF5
from parameters import GLIF_dict, saved_models
import numpy as np
from pathlib import Path
from utilities import plot_results_and_diff, check_nan_arrays_equal


def get_units_dict(model_type, config):
    "Reads the config file to get the needed units. Also converts these units to the correct unit (Allen is in SI)."

    if model_type == "LIF_model":

        units_dict = {
            "C": config["C"] * 1e9,  # F -> nF
            "G": 1 / config["R_input"] * 1e6,  # S -> uS
            "El": config["El"] * 1e3,  # V -> mV
            "th_inf": config["th_inf"] * config["coeffs"]["th_inf"] * 1e3,  # V -> mV
            "dT": config["dt"] * 1e3,  # s -> ms
            "V": config["init_voltage"] * 1e3,  # V -> mV,
            "spike_cut_length": config["spike_cut_length"],
            "refractory_countdown": -1,
        }
    elif model_type == "LIFR_model":

        units_dict = {
            "C": config["C"] * 1e9,  # F -> nF
            "G": 1 / config["R_input"] * 1e6,  # S -> uS
            "El": config["El"] * 1e3,  # V -> mV
            "th_inf": config["th_inf"] * config["coeffs"]["th_inf"] * 1e3,  # V -> mV
            "dT": config["dt"] * 1e3,  # s -> ms
            "V": config["init_voltage"] * 1e3,  # V -> mV,
            "spike_cut_length": config["spike_cut_length"],
            "refractory_countdown": -1,
            "a": config["voltage_reset_method"]["params"]["a"],
            "b": config["voltage_reset_method"]["params"]["b"] * 1e3,  # V -> mV
            "a_spike": config["threshold_dynamics_method"]["params"]["a_spike"]
            * 1e3,  # V -> mV
            "b_spike": config["threshold_dynamics_method"]["params"]["b_spike"]
            * 1e-3,  # inverse of s -> ms
            "th_s": config["init_threshold"] * 1e3,  # V -> mV
        }
    elif model_type == "LIFASC_model":

        units_dict = {
            "C": config["C"] * 1e9,  # F -> nF
            "G": 1 / config["R_input"] * 1e6,  # S -> uS
            "El": config["El"] * 1e3,  # V -> mV
            "th_inf": config["th_inf"] * config["coeffs"]["th_inf"] * 1e3,  # V -> mV
            "dT": config["dt"] * 1e3,  # s -> ms
            "V": config["init_voltage"] * 1e3,  # V -> mV,
            "spike_cut_length": config["spike_cut_length"],
            "refractory_countdown": -1,
            "k": 1 / np.array(config["asc_tau_array"]) * 1e-3,  # inverse of s -> ms
            "r": np.array(config["AScurrent_reset_method"]["params"]["r"]),
            "ASC": np.array(config["init_AScurrents"]) * 1e9,  # A -> nA
            "asc_amp_array": np.array(config["asc_amp_array"])
            * np.array(config["coeffs"]["asc_amp_array"])
            * 1e9,  # A -> nA
            "ASC_length": len(config["init_AScurrents"]),
        }

    elif model_type == "LIFRASC_model":

        units_dict = {
            "El": config["El"] * 1e3,  # V -> mV
            "th_inf": config["th_inf"] * config["coeffs"]["th_inf"] * 1e3,  # V -> mV
            "dT": config["dt"] * 1e3,  # s -> ms
            "V": config["init_voltage"] * 1e3,  # V -> mV,
            "spike_cut_length": config["spike_cut_length"],
            "a": config["voltage_reset_method"]["params"]["a"],
            "b": config["voltage_reset_method"]["params"]["b"] * 1e3,  # V -> mV
            "a_spike": config["threshold_dynamics_method"]["params"]["a_spike"]
            * 1e3,  # V -> mV
            "b_spike": config["threshold_dynamics_method"]["params"]["b_spike"]
            * 1e-3,  # inverse of s -> ms
            "refractory_countdown": -1,
            "k": 1 / np.array(config["asc_tau_array"]) * 1e-3,  # inverse of s -> ms
            "r": np.array(config["AScurrent_reset_method"]["params"]["r"]),
            "ASC": np.array(config["init_AScurrents"]) * 1e9,  # A -> nA
            "asc_amp_array": np.array(config["asc_amp_array"])
            * np.array(config["coeffs"]["asc_amp_array"])
            * 1e9,  # A -> nA
            "th_s": config["init_threshold"] * 1e3,  # V -> mV
            "ASC_length": len(config["init_AScurrents"]),
        }

    elif model_type == "LIFRASCAT_model":

        units_dict = {
            "C": config["C"] * 1e9,  # F -> nF
            "G": 1 / config["R_input"] * 1e6,  # S -> uS
            "El": config["El"] * 1e3,  # V -> mV
            "th_inf": config["th_inf"] * config["coeffs"]["th_inf"] * 1e3,  # V -> mV
            "dT": config["dt"] * 1e3,  # s -> ms
            "V": config["init_voltage"] * 1e3,  # V -> mV,
            "spike_cut_length": config["spike_cut_length"],
            "a": config["voltage_reset_method"]["params"]["a"],
            "b": config["voltage_reset_method"]["params"]["b"] * 1e3,  # V -> mV
            "a_spike": config["threshold_dynamics_method"]["params"]["a_spike"]
            * 1e3,  # V -> mV
            "b_spike": config["threshold_dynamics_method"]["params"]["b_spike"]
            * 1e-3,  # inverse of s -> ms
            "refractory_countdown": -1,
            "k": 1 / np.array(config["asc_tau_array"]) * 1e-3,  # inverse of s -> ms
            "r": np.array(config["AScurrent_reset_method"]["params"]["r"]),
            "ASC": np.array(config["init_AScurrents"]) * 1e9,  # A -> nA
            "asc_amp_array": np.array(config["asc_amp_array"])
            * np.array(config["coeffs"]["asc_amp_array"])
            * 1e9,  # A -> nA
            "th_s": 0.0 * 1e3,  # V -> mV
            "th_v": 0.0 * 1e3,  # V -> mV
            "a_voltage": config["threshold_dynamics_method"]["params"]["a_voltage"]
            * config["coeffs"]["a"]
            * 1e-3,  # inverse of V -> mV
            "b_voltage": config["threshold_dynamics_method"]["params"]["b_voltage"]
            * config["coeffs"]["b"]
            * 1e-3,  # inverse of V -> mV
            "ASC_length": len(config["init_AScurrents"]),
        }
    else:
        raise NotImplementedError

    return units_dict


def get_var_list(model_type):
    """Returns the list of state variables for each model."""

    if model_type == "LIF_model":
        var_list = ["V", "refractory_countdown"]

    elif model_type == "LIFR_model":
        var_list = ["V", "refractory_countdown", "th_s"]

    elif model_type == "LIFASC_model":
        var_list = ["V", "refractory_countdown"]

    elif model_type == "LIFRASC_model":
        var_list = ["V", "refractory_countdown", "th_s"]

    elif model_type == "LIFRASCAT_model":
        var_list = ["V", "refractory_countdown", "th_s", "th_v"]

    else:
        raise NotImplementedError

    return var_list


def run_GeNN_GLIF(specimen_id, model_type, num_neurons):

    # Load Allen model parameters
    saved_model, config, stimulus = load_model_config_stimulus(specimen_id, model_type)

    # Read config parameters and convert to correct units
    units_dict = get_units_dict(model_type, config)

    # Add GLIF Class to model
    GLIF = eval(GLIF_dict[model_type])
    GLIF_params = {k: units_dict[k] for k in GLIF.get_param_names()}
    GLIF_init = {k: units_dict[k] for k in get_var_list(model_type)}
    model = GeNNModel("double", GLIF_dict[model_type], backend="SingleThreadedCPU")
    model.dT = units_dict["dT"]
    pop1 = model.add_neuron_population(
        pop_name="pop1",
        num_neurons=num_neurons,
        neuron=GLIF,
        param_space=GLIF_params,
        var_space=GLIF_init,
    )
    # Assign extra global parameter values
    for k in pop1.extra_global_params.keys():
        pop1.set_extra_global_param(k, units_dict[k])

    # Add current source to model
    external_current_source = create_custom_current_source_class(
        class_name="external_current",
        var_name_types=[("current", "double")],
        injection_code="""
        $(current)=$(Ie)[int($(t) / DT)];
        $(injectCurrent,$(current) );
        """,
        extra_global_params=[("Ie", "double*")],
    )
    cs_ini = {"current": 0.0}  # external input current from Teeter 2018
    cs = model.add_current_source(
        cs_name="external_current_source",
        current_source_model=external_current_source,
        pop=pop1,
        param_space={},
        var_space=cs_ini,
    )
    stimulus_conversion = 1e9  # amps -> nanoamps
    cs.set_extra_global_param("Ie", stimulus * stimulus_conversion)

    # Build and load model
    model.build()
    model.load()

    # Create arrays to store data of variables
    num_steps = len(stimulus)
    vars_list = pop1.vars.keys()
    vars_view_dict = {v: pop1.vars[v].view for v in vars_list}
    extra_global_params_list = pop1.extra_global_params.keys()
    extra_global_params_view_dict = {
        v: pop1.extra_global_params[v].view for v in extra_global_params_list
    }
    data_dict = {}
    full_list = list(vars_list) + list(extra_global_params_list)
    for v in full_list:
        if type(units_dict[v]) == type(np.array(0)):
            num_third_dim = len(units_dict[v])
        else:
            num_third_dim = 1
        data_shape = (num_steps, num_neurons, num_third_dim)
        data_dict[v] = np.zeros(data_shape)
        data_dict[v][:] = np.nan

    for i in range(num_steps):
        model.step_time()

        # Collect state variables
        for v in vars_list:
            pop1.pull_var_from_device(v)
            data_dict[v][model.timestep - 1, :, :] = vars_view_dict[v][:]

        # Collect extra global parameters
        for v in extra_global_params_list:
            pop1.pull_extra_global_param_from_device(v)
            data_dict[v][model.timestep - 1, :, :] = extra_global_params_view_dict[v][:]

        pass
    # Add threshold
    if "th_s" in data_dict.keys() and "th_v" in data_dict.keys():
        data_dict["T"] = data_dict["th_s"] + data_dict["th_v"] + units_dict["th_inf"]
    elif "th_s" in data_dict.keys() and "th_v" not in data_dict.keys():
        data_dict["T"] = data_dict["th_s"] + units_dict["th_inf"]
    else:
        data_dict["T"] = units_dict["th_inf"]

    # Add ASC if not already in data_dict
    if "ASC" not in data_dict.keys():
        num_third_dim = 2  #
        data_dict["ASC"] = np.zeros((num_steps, num_neurons, num_third_dim))

    return data_dict, saved_model


if __name__ == "__main__":

    # Run GeNN Simulation
    specimen_ids = [474637203]  # , 512322162]
    model_types = [
        # "LIF_model",
        # "LIFR_model",
        "LIFASC_model",
        # "LIFRASC_model",
        # "LIFRASCAT_model",
    ]

    for specimen_id in specimen_ids:
        for model_type in model_types:

            save_name = (
                "pkl_data/GeNN_" + str(specimen_id) + "_{}.pkl".format(model_type)
            )

            for model in saved_models:

                # Skip if model already run and saved
                if model.startswith("GeNN_" + str(specimen_id)) and model.endswith(
                    "_{}.pkl".format(model_type)
                ):
                    print("Already saved GeNN run for {}".format(model))
                    break
            else:
                data_dict, saved_model = run_GeNN_GLIF(
                    specimen_id, model_type, num_neurons=1
                )

                # Save results

                # with open(save_name, "wb") as f:
                #     pickle.dump((data_dict, saved_model), f)

            # # Load results
            # with open(save_name, "rb") as f:
            #     data_dict, saved_model = pickle.load(f)

            # Plot the results
            t = saved_model["time"]
            mask = np.logical_and(t > 18, t < 18.3)
            t_mask = t[mask]

            # Voltages
            var_name_dict = {"V": "voltage", "T": "threshold", "ASC": "AScurrents"}
            var_scale = {"V": 1e3, "T": 1e3, "ASC": 1e9}
            var_unit = {"V": "mV", "T": "mV", "ASC": "nA"}
            for v in var_name_dict.keys():

                if v not in data_dict.keys():
                    continue

                try:
                    Allen = saved_model[var_name_dict[v]][mask] * var_scale[v]
                except:
                    Allen = saved_model[var_name_dict[v]][mask, :] * var_scale[v]

                GeNN = np.squeeze(data_dict[v][mask, :, :])
                # result = check_nan_arrays_equal(Allen, GeNN)
                # print("Are results equal: {}".format(result))
                plot_results_and_diff(
                    Allen, "Allen", GeNN, "GeNN", t[mask], var_name_dict[v], var_unit[v]
                )
