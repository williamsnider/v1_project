import sys

sys.path.append("..")  # TODO: Come up with a better way to handle the imports
from pygenn.genn_model import create_custom_current_source_class, GeNNModel
from allen_simulation import load_model_config_stimulus
from GLIF_models import GLIF3
from parameters import GLIF_dict
import numpy as np
from pathlib import Path
from utilities import plot_results_and_diff, check_nan_arrays_equal

specimen_ids = [474637203]  # , 512322162]
model_types = ["LIFASC_model"]
saved_models = [f.name for f in Path("./pkl_data").glob("*")]

for specimen_id in specimen_ids:
    for model_type in model_types:

        # Load Allen model parameters
        saved_model, config, stimulus = load_model_config_stimulus(
            specimen_id, model_type
        )

        # Convert Allen Units (SI) to PyGenn
        units_dict = {
            "C": config["C"] * 1e9,  # F -> nF
            "G": 1 / config["R_input"] * 1e6,  # S -> uS
            "El": config["El"] * 1e3,  # V -> mV
            "th_inf": config["th_inf"] * config["coeffs"]["th_inf"] * 1e3,  # V -> mV
            "dT": config["dt"] * 1e3,  # s -> ms
            "V": config["init_voltage"] * 1e3,  # V -> mV,
            "spike_cut_length": config["spike_cut_length"],
            "refractory_count": -1,
            "k": 1 / np.array(config["asc_tau_array"]) * 1e-3,  # inverse of s -> ms
            "r": np.array(config["AScurrent_reset_method"]["params"]["r"]),
            "ASC": np.array(config["init_AScurrents"]) * 1e9,  # A -> nA
            "asc_amp_array": np.array(config["asc_amp_array"])
            * np.array(config["coeffs"]["asc_amp_array"])
            * 1e9,  # A -> nA
        }

        # Add GLIF Class to model
        num_neurons = 1
        GLIF = eval(GLIF_dict[model_type])
        GLIF_params = {k: units_dict[k] for k in GLIF.get_param_names()}
        GLIF_init = {k: units_dict[k] for k in ["V", "refractory_count"]}
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

        # pop1.set_extra_global_param("ASC", )
        # pop1.extra_global_params
        ### Add current source to model ###
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

        scale = 1e9  # amps -> nanoamps
        cs.set_extra_global_param("Ie", stimulus * scale)

        model.build()
        model.load()

        # Run a quick simulation
        num_steps = len(stimulus)
        v = np.empty((num_steps, num_neurons))
        v_view = pop1.vars["V"].view
        T = np.ones((num_steps, num_neurons)) * units_dict["th_inf"]
        # T_view = pop1.vars["th_s"].view
        for i in range(num_steps):
            model.step_time()
            pop1.pull_var_from_device("V")
            v[model.timestep - 1, :] = v_view[:]
            # pop1.pull_var_from_device("th_s")
            # T[model.timestep - 1, :] += T_view[:]

        t = saved_model["time"]
        mask = np.logical_and(t > 18, t < 18.3)
        t_mask = t[mask]

        # Plot voltages
        Allen = saved_model["voltage"][mask] * 1e3
        GeNN = v[mask, :].ravel()
        result = check_nan_arrays_equal(Allen, GeNN)
        print("Are results equal: {}".format(result))
        plot_results_and_diff(Allen, "Allen", GeNN, "GeNN", t[mask])

        # Plot thresholds
        Allen = saved_model["threshold"][mask] * 1e3
        GeNN = T[mask, :].ravel()
        result = check_nan_arrays_equal(Allen, GeNN)
        print("Are results equal: {}".format(result))
        plot_results_and_diff(Allen, "Allen", GeNN, "GeNN", t[mask])
