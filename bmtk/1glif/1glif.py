import pygenn.genn_model
import numpy as np
from sonata.reports.spike_trains import SpikeTrains
import json
from pathlib import Path
import pandas as pd

DYNAMICS_BASE_DIR = Path("./point_components/cell_models")
SIM_CONFIG_PATH = Path("./point_1glifs/config.simulation.json")
GLIF3_dynamics_file = Path("593618144_glif_lif_asc_psc.json")
SYN_PATH = Path("./point_1glifs/network/lgn_v1_edge_types.csv")
LGN_SPIKES_PATH = Path("./point_1glifs/inputs/lgn_spikes.h5")

### Define custom classes ###
psc_Alpha = pygenn.genn_model.create_custom_postsynaptic_class(
    class_name="Alpha",
    decay_code="""
    $(x) += exp(-DT/$(tau)) * ((DT * $(inSyn) * exp(1.0f) / $(tau)));
    $(inSyn)*=exp(-DT/$(tau));
    """,
    apply_input_code="$(Isyn) += $(x);",
    var_name_types=[("x", "scalar")],
    param_names=[("tau")],
)

GLIF3 = pygenn.genn_model.create_custom_neuron_class(
    "GLIF3",
    param_names=[
        "C",
        "G",
        "El",
        "spike_cut_length",
        "th_inf",
        "ASC_length",
    ],
    var_name_types=[("V", "double"), ("refractory_countdown", "int")],
    extra_global_params=[
        ("ASC", "scalar*"),
        ("k", "scalar*"),
        ("asc_amp_array", "scalar*"),
        ("r", "scalar*"),
    ],
    sim_code="""
    double sum_of_ASC = 0.0;
    
    // Sum after spike currents
    for (int ii=0; ii<$(ASC_length); ii++){
        int idx = $(id)*((int)$(ASC_length))+ii;
        sum_of_ASC += $(ASC)[idx];
        }
    // Voltage
    if ($(refractory_countdown) > 0) {
        $(V) += 0.0;
    }
    else {
        $(V)+=1/$(C)*($(Isyn)+sum_of_ASC-$(G)*($(V)-$(El)))*DT;
    }
    // ASCurrents
    if ($(refractory_countdown) > 0) {
        for (int ii=0; ii<$(ASC_length); ii++){
            int idx = $(id)*((int)$(ASC_length))+ii;
            $(ASC)[idx] += 0.0;
            }
    }
    else {
        for (int ii=0; ii<$(ASC_length); ii++){
            int idx = $(id)*((int)$(ASC_length))+ii;
            $(ASC)[idx] = $(ASC)[idx] * exp(-$(k)[idx]*DT);
            }
    }
    // Decrement refractory_countdown by 1; Do not decrement past -1
    if ($(refractory_countdown) > -1) {
        $(refractory_countdown) -= 1;
    }
    """,
    threshold_condition_code="$(V) > $(th_inf)",
    reset_code="""
    $(V)=0;
    for (int ii=0; ii<$(ASC_length); ii++){
        int idx = $(id)*((int)$(ASC_length))+ii;
        $(ASC)[idx] = $(asc_amp_array)[idx] + $(ASC)[idx] * $(r)[idx] * exp(-($(k)[idx] * DT * $(spike_cut_length)));
        }
    $(refractory_countdown) = $(spike_cut_length);
    """,
)


def spikes_list_to_start_end_times(spikes_list):

    spike_counts = [len(n) for n in spikes_list]

    # Get start and end indices of each spike sources section
    end_spike = np.cumsum(spike_counts)
    start_spike = np.empty_like(end_spike)
    start_spike[0] = 0
    start_spike[1:] = end_spike[0:-1]

    spike_times = np.hstack(spikes_list)
    return start_spike, end_spike, spike_times


### Create model ###
with open(SIM_CONFIG_PATH) as f:
    sim_config = json.load(f)
model = pygenn.genn_model.GeNNModel(backend="SingleThreadedCPU")
model.dT = sim_config["run"]["dt"]

### Add population of 1 LGN neuron ###
spikes = SpikeTrains.from_sonata(LGN_SPIKES_PATH)
spikes_df = spikes.to_dataframe()

num_lgn = 1
spikes_list = []
for n in range(0, num_lgn):
    spikes_list.append(spikes_df[spikes_df["node_ids"] == n]["timestamps"].to_list())

start_spike, end_spike, spike_times = spikes_list_to_start_end_times(spikes_list)

pop_dict = {}
pop_dict["LGN"] = model.add_neuron_population(
    "LGN",
    num_lgn,
    "SpikeSourceArray",
    {},
    {"startSpike": start_spike, "endSpike": end_spike},
)
pop_dict["LGN"].set_extra_global_param("spikeTimes", spike_times)

### Add population of 1 GLIF neuron ###
v1_dynamics_path = Path(DYNAMICS_BASE_DIR, GLIF3_dynamics_file)
with open(v1_dynamics_path) as f:
    dynamics_params = json.load(f)
num_GLIF = 1


dynamics_params_renamed = {
    "C": dynamics_params["C_m"] / 1000,  # pF -> nF
    "G": dynamics_params["g"] / 1000,  # nS -> uS
    "El": dynamics_params["E_L"],
    "th_inf": dynamics_params["V_th"],
    "dT": sim_config["run"]["dt"],
    "V": dynamics_params["V_m"],
    "spike_cut_length": round(dynamics_params["t_ref"] / sim_config["run"]["dt"]),
    "refractory_countdown": -1,
    "k": 1
    / np.repeat(dynamics_params["asc_decay"], num_GLIF).ravel(),  # TODO: Reciprocal?
    "r": np.repeat([1.0, 1.0], num_GLIF).ravel(),
    "ASC": np.repeat(dynamics_params["asc_init"], num_GLIF).ravel() / 1000,  # pA -> nA
    "asc_amp_array": np.repeat(dynamics_params["asc_amps"], num_GLIF).ravel()
    / 1000,  # pA -> nA
    "ASC_length": 2,
}

params = {k: dynamics_params_renamed[k] for k in GLIF3.get_param_names()}
init = {k: dynamics_params_renamed[k] for k in ["V", "refractory_countdown"]}

pop_dict["GLIF3"] = model.add_neuron_population(
    pop_name="GLIF3",
    num_neurons=num_GLIF,
    neuron=GLIF3,
    param_space=params,
    var_space=init,
)

for k in pop_dict["GLIF3"].extra_global_params.keys():
    pop_dict["GLIF3"].set_extra_global_param(k, dynamics_params_renamed[k])


### Add synapse population ###
# TODO: How to indicate excitatory/inhibitory
# TODO: How to add synaptic weight?
# TODO: Tau normalized to give 1pA?
syn_df = pd.read_csv(SYN_PATH, sep=" ")
delay_steps = round(
    syn_df[syn_df["edge_type_id"] == 100]["delay"][0] / sim_config["run"]["dt"]
)  # delay (ms) -> delay (steps)
weight = syn_df[syn_df["edge_type_id"] == 100]["syn_weight"][0] / 1e3  # nS -> uS
syn_dict = {}
psc_Alpha_params = {"tau": dynamics_params["tau_syn"][0]}  # TODO: Always port 0?
psc_Alpha_init = {"x": 0.0}  # TODO check 0
s_ini = {"g": weight}  # TODO  Unsure about this value
syn_dict["LGN_to_GLIF3"] = model.add_synapse_population(
    pop_name="LGN_to_GLIF3",
    matrix_type="SPARSE_GLOBALG_INDIVIDUAL_PSM",
    delay_steps=delay_steps,
    source="LGN",
    target="GLIF3",
    w_update_model="StaticPulse",
    wu_var_space=s_ini,
    wu_pre_var_space={},  # TODO: not sure about this weight update
    wu_post_var_space={},
    wu_param_space={},
    postsyn_model=psc_Alpha,
    ps_param_space=psc_Alpha_params,
    ps_var_space=psc_Alpha_init,
)
syn_dict["LGN_to_GLIF3"].set_sparse_connections(np.array([0]), np.array([0]))


model.build(force_rebuild=True)
model.load()

num_steps = round(3000 / model.dT)  # Nest simulation is 3000ms
var_list = ["V"]
data = {"GLIF3": {"V": np.zeros((1, num_steps))}}
view_dict = {"GLIF3": {"V": pop_dict["GLIF3"].vars["V"].view}}

for i in range(num_steps):

    model.step_time()

    for model_name in ["GLIF3"]:
        pop = pop_dict[model_name]

        v_view = pop.vars["V"].view
        for var_name in var_list:
            # print(i, model_name, var_name, sep="\t")
            # pop.pull_var_from_device("V")
            view = view_dict[model_name][var_name]
            output = view[:]
            if i % 100000 == 0:
                print(i)
                print(output)
            data[model_name][var_name][:, i] = output


# Plot voltage
A = data["GLIF3"]["V"].ravel()
t = np.arange(0, len(A)) * sim_config["run"]["dt"]
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1)

# GeNN
axs.plot(t, A, label="GeNN")
axs.set_ylabel("mV")
axs.set_xlabel("ms")

# Nest
from bmtk.utils.reports.compartment import CompartmentReport

pop_name = "Scnn1a"
REPORT_PATH = Path("./point_1glifs/output/membrane_potential.h5")
report = CompartmentReport(REPORT_PATH, population=pop_name, mode="r")
B = report.data(node_id=0).ravel()
t = (
    np.arange(0, len(B)) * sim_config["run"]["dt"]
)  # TODO: uneven numbers between A and B?
axs.plot(t, B, label="Nest")
axs.legend()

plt.show()
