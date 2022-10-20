import numpy as np
import pandas as pd
import json
from pathlib import Path
from sonata.circuit import File
from sonata.reports.spike_trains import SpikeTrains
import pygenn
import matplotlib.pyplot as plt
from helper import (
    optimize_nodes_df_memory,
    optimize_edges_df_memory,
    get_dynamics_params,
    load_df,
    save_df,
    psc_Alpha,
)

# Neuron class for GLIF3 model. Note that "extra_global_params" are used for all parameters/variables. This allows a single neuron population to be created in GeNN for all node types in V1, which greatly reduces the number of synapse groups that need to be created.
GLIF3 = pygenn.genn_model.create_custom_neuron_class(
    "GLIF3",
    param_names=[],
    var_name_types=[],
    sim_code="""

    // Sum after spike currents
    double sum_of_ASC = $(ASC_1)[$(id)]*$(asc_stable_coeff_1)[$(id)] + $(ASC_2)[$(id)]*$(asc_stable_coeff_2)[$(id)];

    // Voltage
    if ($(refractory_countdown)[$(id)] <= 0) {
        $(V)[$(id)]+=1/$(C)[$(id)]*($(Isyn)+sum_of_ASC-$(G)[$(id)]*($(V)[$(id)]-$(El)[$(id)]))*DT;
    }

    // ASCurrents
    if ($(refractory_countdown)[$(id)] <= 0) {
        $(ASC_1)[$(id)] *= $(asc_decay_rates_1)[$(id)];
        $(ASC_2)[$(id)] *= $(asc_decay_rates_2)[$(id)];
        }


    // Decrement refractory_countdown by 1; Do not decrement past -1
    if ($(refractory_countdown)[$(id)] > -1) {
        $(refractory_countdown)[$(id)] -= 1;
    }

    """,
    threshold_condition_code="$(V)[$(id)] > $(th_inf)[$(id)]",
    reset_code="""
    $(V)[$(id)]=$(V_reset)[$(id)];
    $(ASC_1)[$(id)] = $(asc_amp_array_1)[$(id)] + $(ASC_1)[$(id)] * $(asc_refractory_decay_rates_1)[$(id)];
    $(ASC_2)[$(id)] = $(asc_amp_array_2)[$(id)] + $(ASC_2)[$(id)] * $(asc_refractory_decay_rates_2)[$(id)];
    $(refractory_countdown)[$(id)] = $(spike_cut_length)[$(id)];
    """,
    extra_global_params=[
        ("V", "scalar*"),
        ("refractory_countdown", "int*"),
        ("ASC_1", "scalar*"),
        ("ASC_2", "scalar*"),
        ("C", "scalar*"),
        ("G", "scalar*"),
        ("El", "scalar*"),
        ("spike_cut_length", "int*"),
        ("th_inf", "scalar*"),
        ("V_reset", "scalar*"),
        ("asc_amp_array_1", "scalar*"),
        ("asc_amp_array_2", "scalar*"),
        ("asc_stable_coeff_1", "scalar*"),
        ("asc_stable_coeff_2", "scalar*"),
        ("asc_decay_rates_1", "scalar*"),
        ("asc_decay_rates_2", "scalar*"),
        ("asc_refractory_decay_rates_1", "scalar*"),
        ("asc_refractory_decay_rates_2", "scalar*"),
    ],
)

pulse_global = pygenn.genn_model.create_custom_weight_update_class(
    class_name="pulse_global",
    param_names=["d", "g"],
    sim_code="""
    $(addToInSynDelay, $(g), int($(d)));""",
)

v1_net = File(
    data_files=[
        "GLIF Network/network/v1_nodes.h5",
        "GLIF Network/network/v1_v1_edges.h5",
    ],
    data_type_files=[
        "GLIF Network/network/v1_node_types.csv",
        "GLIF Network/network/v1_v1_edge_types.csv",
    ],
)

lgn_net = File(
    data_files=[
        "GLIF Network/network/lgn_nodes.h5",
        "GLIF Network/network/lgn_v1_edges.h5",
    ],
    data_type_files=[
        "GLIF Network/network/lgn_node_types.csv",
        "GLIF Network/network/lgn_v1_edge_types.csv",
    ],
)

bkg_net = File(
    data_files=[
        "GLIF Network/network/bkg_nodes.h5",
        "GLIF Network/network/bkg_v1_edges.h5",
    ],
    data_type_files=[
        "GLIF Network/network/bkg_node_types.csv",
        "GLIF Network/network/bkg_v1_edge_types.csv",
    ],
)

DYNAMICS_BASE_DIR = Path("./GLIF Network/models/cell_models/nest_2.14_models")
SIM_CONFIG_PATH = Path("./GLIF Network/config.json")
LGN_V1_EDGE_CSV = Path("./GLIF Network/network/lgn_v1_edge_types.csv")
V1_EDGE_CSV = Path("./GLIF Network/network/v1_v1_edge_types.csv")
LGN_SPIKES_PATH = Path(
    "GLIF Network/inputs/full3_GScorrected_PScorrected_3.0sec_SF0.04_TF2.0_ori270.0_c100.0_gs0.5_spikes.trial_0.h5"
)
LGN_NODE_DIR = Path("./GLIF Network/network/lgn_node_types.csv")
V1_NODE_CSV = Path("./GLIF Network/network/v1_node_types.csv")
V1_ID_CONVERSION_FILENAME = Path(".", "pkl_data", "v1_edges_df.pkl")
LGN_ID_CONVERSION_FILENAME = Path(".", "pkl_data", "lgn_edges_df.pkl")
BKG_V1_EDGE_CSV = Path("./GLIF Network/network/bkg_v1_edge_types.csv")
BKG_ID_CONVERSION_FILENAME = Path(".", "pkl_data", "bkg_edges_df.pkl")
NUM_RECORDING_TIMESTEPS = 10000
num_steps = 300000
neuron_class = GLIF3
ALPHA_TAU = 5.5  # All nodes have alpha-shaped postsynaptic current with tau=5.5


### Create base model ###
with open(SIM_CONFIG_PATH) as f:
    sim_config = json.load(f)
model = pygenn.genn_model.GeNNModel(backend="CUDA")
DT = sim_config["run"]["dt"]
model.dT = DT
model._model.set_merge_postsynaptic_models(True)
model._model.set_default_narrow_sparse_ind_enabled(True)
# model.default_var_location = pygenn.genn_model.genn_wrapper.VarLocation_DEVICE

### Add Neuron Populations ###
pop_dict = {}

### V1
v1_nodes_df_path = Path("./pkl_data/v1_nodes_df.pkl")
if v1_nodes_df_path.exists():
    v1_nodes_df = load_df(v1_nodes_df_path)
else:

    # Construct df from Sonata format
    v1_nodes = v1_net.nodes["v1"]
    v1_nodes_df = v1_nodes.to_dataframe()

    # Add columns with neuron variables/paramters (to be applied as extra global params) for each node_type
    for node_type_id in v1_nodes_df["node_type_id"].unique():

        # Load dynamics params
        dynamics_file = v1_nodes_df.loc[v1_nodes_df["node_type_id"] == node_type_id][
            "dynamics_params"
        ].iloc[0]
        dynamics_file = dynamics_file.replace("config", "psc")
        dynamics_path = Path(DYNAMICS_BASE_DIR, dynamics_file)
        dynamics_params_correct_units = get_dynamics_params(dynamics_path, DT)

        # Assign to df columns for nodes that have this node_type_id
        for extra_global_param_name in [
            egp.name for egp in neuron_class.get_extra_global_params()
        ]:
            v1_nodes_df.loc[
                v1_nodes_df["node_type_id"] == node_type_id, extra_global_param_name
            ] = dynamics_params_correct_units[extra_global_param_name]

    # Reduce memory by dropping columns / downcasting variable types
    v1_nodes_df = optimize_nodes_df_memory(v1_nodes_df)

    # Save as pkl so can be reloaded faster
    save_df(v1_nodes_df, v1_nodes_df_path)

pop_dict["v1"] = model.add_neuron_population(
    pop_name="v1",
    num_neurons=len(v1_nodes_df),
    neuron=neuron_class,
    param_space={},
    var_space={},
)

# Assign extra global params to neuron population
for egc in pop_dict["v1"].extra_global_params.keys():
    pop_dict["v1"].set_extra_global_param(egc, v1_nodes_df[egc].to_list())

### Add Synapse Populations ###
syn_dict = {}

### V1 to V1
v1_edges_df_path = Path("./pkl_data/v1_edges_df.pkl")
if v1_edges_df_path.exists():
    v1_edges_df = load_df(v1_edges_df_path)
else:

    # Load df from Sonata format
    v1_edges = v1_net.edges["v1_to_v1"]
    v1_edges_df = v1_edges.groups[0].to_dataframe()

    # Reduce size of df by dropping columns / downcasting variable types
    v1_edges_df = optimize_edges_df_memory(v1_edges_df)

    # Save as pkl so can be reloaded faster
    save_df(v1_edges_df, v1_edges_df_path)


# Name synapse group
pop1 = "v1"
pop2 = "v1"
synapse_group_name = pop1 + "_to_" + pop2

# Load sub_edges_df if exists; otherwise reconstruct
# sub_edges_df_path = Path("./pkl_data/sub_edges/{}.pkl".format(synapse_group_name))
# if sub_edges_df_path.exists():
#     sub_edges_df = load_df(sub_edges_df_path)
# else:
#     # Select edges with this edge_type_id and nsyns
#     sub_edges_df = v1_edges_df.loc[
#         (v1_edges_df["edge_type_id"] == edge_type_id) & (v1_edges_df["nsyns"] == nsyns)
#     ]
#     save_df(sub_edges_df, sub_edges_df_path)
sub_edges_df = v1_edges_df

# Synaptic weight
weights = (
    sub_edges_df["syn_weight"] / 1e3 * sub_edges_df["nsyns"]
)  # pA -> nA; weight multiplied by nsyns


# Delay steps
delay_ms = sub_edges_df["delay"]
delay_steps = np.ceil(delay_ms / DT)  # delay (ms) -> delay (steps)

# Source and Target node_id for each synaptic connection
s_list = sub_edges_df["source_node_id"].tolist()
t_list = sub_edges_df["target_node_id"].tolist()

# Weight update model
# TODO: Set weights/delay_steps here
s_ini = {"g": weights, "d": delay_steps}

# Postsynaptic current model
psc_Alpha_params = {"tau": ALPHA_TAU}
psc_Alpha_init = {"x": 0.0}

# Add synapse population
syn_dict[synapse_group_name] = model.add_synapse_population(
    pop_name=synapse_group_name,
    matrix_type="SPARSE_INDIVIDUALG",
    delay_steps=pygenn.genn_model.genn_wrapper.NO_DELAY,
    source=pop1,
    target=pop2,
    w_update_model="StaticPulseDendriticDelay",
    wu_param_space={},
    wu_var_space=s_ini,
    wu_pre_var_space={},
    wu_post_var_space={},
    postsyn_model=psc_Alpha,
    ps_param_space=psc_Alpha_params,
    ps_var_space=psc_Alpha_init,
)

# Which synapse matrix is best for variable delays and weights?
# How to set individual delays?
# How to set individual weights?

# Set max_dendritic_delay_timesteps, which improves optimization
max_dendritic_delay_slots = int(np.ceil(sub_edges_df["delay"].max() / DT))
syn_dict[synapse_group_name].pop.set_max_dendritic_delay_timesteps(
    max_dendritic_delay_slots
)  # TODO: This may not be optmization correclty

# Assign synaptic connections
syn_dict[synapse_group_name].set_sparse_connections(np.array(s_list), np.array(t_list))

print("Synapse group added: {}".format(synapse_group_name))


# # extract data
# weight = v1_edges_df["syn_weight"].iloc[0] / 1e3
# delay_steps = int(
#     round(v1_edges_df["delay"].iloc[0] / DT)
# )  # delay (ms) -> delay (steps)

# s_list = v1_edges_df["source_node_id"].tolist()
# t_list = v1_edges_df["target_node_id"].tolist()


# pulse_params = {"d": delay_steps, "g": weight}
# s_ini = {"g": weight}  # , "d": delay_steps}
# psc_Alpha_params = {"tau": ALPHA_TAU}  # TODO: Always 0th port?
# psc_Alpha_init = {"x": 0.0}

# pop1 = "v1"
# pop2 = "v1"
# synapse_group_name = pop1 + "_to_" + pop2 + "_nsyns_"
# syn_dict[synapse_group_name] = model.add_synapse_population(
#     pop_name=synapse_group_name,
#     matrix_type="SPARSE_GLOBALG_INDIVIDUAL_PSM",
#     delay_steps=pygenn.genn_model.genn_wrapper.NO_DELAY,
#     source=pop1,
#     target=pop2,
#     w_update_model=pulse_global,
#     wu_param_space=pulse_params,
#     wu_var_space={},  # s_ini,
#     wu_pre_var_space={},
#     wu_post_var_space={},
#     postsyn_model=psc_Alpha,
#     ps_param_space=psc_Alpha_params,
#     ps_var_space=psc_Alpha_init,
#     # connectivity_initialiser=pygenn.genn_model.init_connectivity(
#     #     sparse_connect_model, {}
#     # ),
# )

# max_dendritic_delay_slots = int(np.ceil(v1_edges_df["delay"].max() / DT))
# syn_dict[synapse_group_name].pop.set_max_dendritic_delay_timesteps(
#     max_dendritic_delay_slots
# )

# syn_dict[synapse_group_name].set_sparse_connections(np.array(s_list), np.array(t_list))

# Build model
import time

start = time.time()
model.build(force_rebuild=False)
stop = time.time()
print("Duration = {}s".format(stop - start))
model.load(
    num_recording_timesteps=NUM_RECORDING_TIMESTEPS
)  # TODO: How big to calculate for GPU size?

for i in range(num_steps):

    model.step_time()

    # Only collect full BUFFER
    if i % NUM_RECORDING_TIMESTEPS == 0 and i != 0:
        print(i)
