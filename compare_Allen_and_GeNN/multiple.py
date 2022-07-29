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
            "th_s": 0.0 * 1e3,  # V -> mV
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


def run_GeNN_GLIF(all_specimens_unit_dict, units_dict, num_neurons, stimulus):

    GLIF = eval(GLIF_dict[model_type])

    # # Make units dict with single numerics
    # units_dict = {}
    # keys = [k for k in GLIF.get_param_names()]
    # print(keys)
    # keys.extend([k for k in get_var_list(model_type)])
    # print(keys)
    # for k in keys:
    #     unit = all_specimens_unit_dict[k]
    #     if len(unit) == 1:
    #         pass_none
    #     else:
    #         assert np.all(np.diff(unit) == 0.0), "Given parameters not all the same"
    #     units_dict[k] = unit[
    #         0
    #     ]  # Store only first value -- params must be numeric, not list
    # print(units_dict)

    # Add GLIF Class to model
    GLIF_params = {k: units_dict[k] for k in GLIF.get_param_names()}
    GLIF_init = {k: all_specimens_unit_dict[k] for k in get_var_list(model_type)}
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
        pop1.set_extra_global_param(k, all_specimens_unit_dict[k])

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
            data = vars_view_dict[v]
            # if data[0] != data[1]:
            #     print("mismatch")
            if data.ndim == 1:
                data_dict[v][model.timestep - 1, :, 0] = data
            elif data.ndim == 2:
                data_dict[v][model.timestep - 1, :, :] = data
            else:
                raise NotImplementedError
        # Collect extra global parameters
        for v in extra_global_params_list:
            data = extra_global_params_view_dict[v]
            data_dict[v][model.timestep - 1, :, :] = data.reshape(num_neurons, -1)

    # Add threshold
    if "th_s" in data_dict.keys() and "th_v" in data_dict.keys():
        data_dict["T"] = data_dict["th_s"] + data_dict["th_v"] + units_dict["th_inf"]
    elif "th_s" in data_dict.keys() and "th_v" not in data_dict.keys():
        data_dict["T"] = data_dict["th_s"] + units_dict["th_inf"]
    else:
        num_third_dim = 1
        data_shape = (num_steps, num_neurons, num_third_dim)
        data_dict["T"] = units_dict["th_inf"] + np.zeros(data_shape)

    # Add ASC if not already in data_dict
    if "ASC" not in data_dict.keys():
        num_third_dim = 2  #
        data_dict["ASC"] = np.zeros((num_steps, num_neurons, num_third_dim))

    return data_dict


if __name__ == "__main__":

    # Run GeNN Simulation
    specimen_ids = [474637203, 474637203, 474637203, 474637203]  # , 512322162]
    model_types = [
        "LIF_model",
        "LIFR_model",
        "LIFASC_model",
        "LIFRASC_model",
        "LIFRASCAT_model",
    ]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(model_types), len(specimen_ids))
    for row_num, model_type in enumerate(model_types):

        dummy_id = specimen_ids[0]
        save_name = "pkl_data/GeNN_" + str(dummy_id) + "_{}.pkl".format(model_type)

        # for model in saved_models:

        #     # Skip if model already run and saved
        #     if model.startswith("GeNN_" + str(dummy_id)) and model.endswith(
        #         "_{}.pkl".format(model_type)
        #     ):
        #         print("Already saved GeNN run for {}".format(model))
        #         break
        # else:

        # Load Allen model parameters

        # units_dict_list = []
        # for specimen_id in specimen_ids:
        #     saved_model, config, stimulus = load_model_config_stimulus(
        #         dummy_id, model_type
        #     )
        #     units_dict = get_units_dict(model_type, config)
        #     units_dict_list.append(units_dict)

        # Load stimulus
        id_for_stimulus = specimen_ids[0]
        saved_model, config, stimulus = load_model_config_stimulus(
            id_for_stimulus, model_type
        )

        # Group units into single dict
        all_specimens_unit_dict = {
            k: [] for k in get_units_dict(model_type, config).keys()
        }
        for specimen_id in specimen_ids:
            _, config, _ = load_model_config_stimulus(specimen_id, model_type)
            units_dict = get_units_dict(
                model_type, config
            )  # TODO: this will be the last units dict
            for k in all_specimens_unit_dict.keys():

                v = units_dict[k]

                if type(v) == np.ndarray:
                    v = v.tolist()
                    all_specimens_unit_dict[k].extend(v)
                else:
                    all_specimens_unit_dict[k].append(v)

        # Read config parameters and convert to correct units
        data_dict = run_GeNN_GLIF(
            all_specimens_unit_dict,
            units_dict,
            num_neurons=len(specimen_ids),
            stimulus=stimulus,
        )

        # Save results

        with open(save_name, "wb") as f:
            pickle.dump((data_dict, saved_model), f)

        # # Load results
        # with open(save_name, "rb") as f:
        #     data_dict, saved_model = pickle.load(f)

        t = saved_model["time"]

        # Voltages
        var_name_dict = {"V": "voltage", "T": "threshold", "ASC": "AScurrents"}
        var_scale = {"V": 1e3, "T": 1e3, "ASC": 1e9}
        var_unit = {"V": "mV", "T": "mV", "ASC": "nA"}
        for v in var_name_dict.keys():

            if v not in data_dict.keys():
                continue

            try:
                Allen = saved_model[var_name_dict[v]] * var_scale[v]
            except:
                Allen = saved_model[var_name_dict[v]][:] * var_scale[v]

            GeNN = np.squeeze(data_dict[v])
            # result = check_nan_arrays_equal(Allen, GeNN)
            # print("Are results equal: {}".format(result))
            # plot_results_and_diff(
            #     Allen, "Allen", GeNN[:, 0], "GeNN_0", t, var_name_dict[v], var_unit[v]
            # )

            # plot_results_and_diff(
            #     Allen, "Allen", GeNN[:, 1], "GeNN_1", t, var_name_dict[v], var_unit[v]
            # )

            if v == "V":

                for col_num in range(len(specimen_ids)):
                    G = GeNN[:, col_num]
                    axes[row_num, col_num].plot(t, G, label="GeNN_{}".format(col_num))
                    axes[row_num, col_num].plot(t, Allen, label="Allen")
                    axes[row_num, col_num].set_ylabel("mV")
                    axes[row_num, col_num].set_title(
                        "GeNN_{} and Allen".format(col_num)
                    )
                    axes[row_num, col_num].legend()
                    if col_num == 0:
                        axes[row_num, col_num].set_ylabel("{}\nmV".format(model_type))
                    if row_num == len(specimen_ids):
                        axes[row_num, col_num].set_xlabel("Time (s)")

plt.suptitle("GeNN with Multiple Neurons")
plt.show()
