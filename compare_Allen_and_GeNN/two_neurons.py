# Test whether GLIF models can support multiple neurons (e.g. extra global params must be indexed intelligently)
import sys

sys.path.append("..")  # TODO: Come up with a better way to handle the imports
from pygenn.genn_model import create_custom_current_source_class, GeNNModel
import scipy as sp
from GeNN_GLIF import (
    get_units_dict,
    get_var_list,
    run_GeNN_GLIF,
    load_model_config_stimulus,
)
from GLIF_models import GLIF1, GLIF2, GLIF3, GLIF4, GLIF5
from parameters import GLIF_dict, saved_models
import numpy as np
import pickle
from utilities import plot_results_and_diff, check_nan_arrays_equal

specimen_ids = [474637203, 474637203]
model_type = "LIFASC_model"
num_neurons = len(specimen_ids)
# model_types = [
#     "LIF_model",
#     "LIFR_model",
#     "LIFASC_model",
#     "LIFRASC_model",
#     "LIFRASCAT_model",
# ]

# Get params for both specimens in single dict
dummy_specimen_id = 474637203
_, config, _ = load_model_config_stimulus(dummy_specimen_id, model_type)
all_specimens_unit_dict = {k: [] for k in get_units_dict(model_type, config).keys()}
for specimen_id in specimen_ids:
    saved_model, config, stimulus = load_model_config_stimulus(specimen_id, model_type)
    units_dict = get_units_dict(model_type, config)
    for k in all_specimens_unit_dict.keys():

        v = units_dict[k]

        if type(v) == np.ndarray:
            v = v.tolist()
            all_specimens_unit_dict[k].extend(v)
        else:
            all_specimens_unit_dict[k].append(v)


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
        if data.ndim == 1:
            data_dict[v][model.timestep - 1, :, 0] = data
        elif data.ndim == 2:
            data_dict[v][model.timestep - 1, :, :] = data
        else:
            raise NotImplementedError

    # Collect extra global parameters
    for v in extra_global_params_list:
        pop1.pull_extra_global_param_from_device(v)
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


save_name = (
    "pkl_data/GeNN_"
    + "_".join([str(s) for s in specimen_ids])
    + "_{}.pkl".format(model_type)
)

# for model in saved_models:

#     # Skip if model already run and saved
#     if model.startswith("GeNN_" + str(specimen_id)) and model.endswith(
#         "_{}.pkl".format(model_type)
#     ):
#         print("Already saved GeNN run for {}".format(model))
#         break
# else:
#     data_dict, saved_model = run_GeNN_GLIF(specimen_id, model_type, num_neurons=1)
len(specimen_ids)
# Load results
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
        Allen, "Allen", GeNN[:, 0], "GeNN 0", t[mask], var_name_dict[v], var_unit[v]
    )

    plot_results_and_diff(
        Allen, "Allen", GeNN[:, 1], "GeNN 1", t[mask], var_name_dict[v], var_unit[v]
    )