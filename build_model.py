# Replicate point_450glifs example with GeNN
from pickletools import optimize
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
    GLIF3,
    load_df,
    save_df,
    psc_Alpha,
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
num_steps = 3000000
ALPHA_TAU = 5.5  # All nodes have alpha-shaped postsynaptic current with tau=5.5

# Parameters/variables used in GLIF3 neuron class
param_names = [x for x in GLIF3.get_param_names()]
var_names = [x.name for x in GLIF3.get_vars()]

### Create base model ###
with open(SIM_CONFIG_PATH) as f:
    sim_config = json.load(f)
model = pygenn.genn_model.GeNNModel(backend="CUDA")
DT = sim_config["run"]["dt"]
model.dT = DT
model._model.set_merge_postsynaptic_models(True)
model._model.set_default_narrow_sparse_ind_enabled(True)
# model.default_var_location = pygenn.genn_model.genn_wrapper.VarLocation_DEVICE
# model.default_sparse_connectivity_location = pygenn.genn_model.genn_wrapper.VarLocation_DEVICE

### Add Neuron Populations ###
pop_dict = {}

### V1

# Add data to dataframe
v1_nodes_df_path = Path("./pkl_data/v1_nodes_df.pkl")
if v1_nodes_df_path.exists():
    v1_nodes_df = load_df(v1_nodes_df_path)
else:

    # Construct df from Sonata format
    v1_nodes = v1_net.nodes["v1"]
    v1_nodes_df = v1_nodes.to_dataframe()
    v1_nodes_df = optimize_nodes_df_memory(v1_nodes_df)  # reduce memory; makes indexing faster

    # Add columns of new data
    v1_nodes_df["GeNN_node_id"] = 0  # Preallocate as int
    v1_nodes_df["refractory_countdown"] = 0  # Preallocate as int
    v1_nodes_df["spike_cut_length"] = 0  # Preallocate as int

    for node_type_id in v1_nodes_df["node_type_id"].unique():

        # Dynamics params
        dynamics_file = v1_nodes_df.loc[v1_nodes_df["node_type_id"] == node_type_id]["dynamics_params"].iloc[0]
        dynamics_file = dynamics_file.replace("config", "psc")
        dynamics_path = Path(DYNAMICS_BASE_DIR, dynamics_file)
        dynamics_params_correct_units = get_dynamics_params(dynamics_path, DT)
        for pv_name in param_names + var_names:
            v1_nodes_df.loc[v1_nodes_df["node_type_id"] == node_type_id, pv_name] = dynamics_params_correct_units[
                pv_name
            ]

        # Model name = pop_name + node_type_id
        pop_name = v1_nodes_df[v1_nodes_df["node_type_id"] == node_type_id]["pop_name"].iloc[0]
        model_name = "{}_{}".format(pop_name, node_type_id)
        v1_nodes_df.loc[v1_nodes_df["node_type_id"] == node_type_id, "model_name"] = model_name

        # GeNN ID; counts from 0 for each model_name
        num_nodes = v1_nodes_df.loc[v1_nodes_df["node_type_id"] == node_type_id].shape[0]
        v1_nodes_df.loc[v1_nodes_df["node_type_id"] == node_type_id, "GeNN_node_id"] = np.arange(num_nodes).astype(
            "int"
        )

    # Reduce memory by dropping columns / downcasting variable types
    v1_nodes_df = optimize_nodes_df_memory(v1_nodes_df)

    # Save as pkl so can be reloaded faster
    save_df(v1_nodes_df, v1_nodes_df_path)

# Add V1 nodes as neuron populations (111 node types / model_names)
for model_name in v1_nodes_df["model_name"].unique():

    # Get data from nodes with this model_name
    subset_df = v1_nodes_df[v1_nodes_df["model_name"] == model_name]
    params = {k: subset_df[k].to_list()[0] for k in param_names}
    init = {k: subset_df[k].to_list()[0] for k in var_names}
    num_neurons = len(subset_df)

    pop_dict[model_name] = model.add_neuron_population(
        pop_name=model_name,
        num_neurons=num_neurons,
        neuron=GLIF3,
        param_space=params,
        var_space=init,
    )

    # Enable spike recording
    pop_dict[model_name].spike_recording_enabled = True

    print("Added population: {}.".format(model_name))


### Add synapses ###
syn_dict = {}

# V1 to V1 synapses
v1_edges_df_path = Path("./pkl_data/v1_edges_df.pkl")
if v1_edges_df_path.exists():
    v1_edges_df = load_df(v1_edges_df_path)
else:

    # Load as dataframe
    v1_edges = v1_net.edges["v1_to_v1"]
    v1_edges_df = v1_edges.groups[0].to_dataframe()
    edges_df = v1_edges_df
    edges_df = optimize_edges_df_memory(edges_df)

    # Add ID's for GeNN (0-num_neurons in each population)
    edges_df["source_GeNN_id"] = v1_nodes_df["GeNN_node_id"].iloc[edges_df["source_node_id"]].astype("int32").tolist()
    edges_df["target_GeNN_id"] = (
        v1_nodes_df["GeNN_node_id"].iloc[v1_edges_df["target_node_id"]].astype("int32").tolist()
    )
    edges_df["source_model_name"] = v1_nodes_df["model_name"].iloc[edges_df["source_node_id"]].tolist()
    edges_df["target_model_name"] = v1_nodes_df["model_name"].iloc[edges_df["target_node_id"]].tolist()

    # Add product of nsyns and syn_weight
    edges_df["nsyns_x_syn_weight"] = edges_df["nsyns"] * edges_df["syn_weight"]

    # Reduce memory
    edges_df = optimize_edges_df_memory(edges_df)

    # Save as pickle for faster loading
    save_df(edges_df, v1_edges_df_path)

# List of all population pairs
source_target_pairs = (
    v1_edges_df.drop_duplicates(subset=["source_model_name", "target_model_name"])
    .loc[:, ("source_model_name", "target_model_name")]
    .to_numpy()
)

# Iterate through population pairs
num_pairs = len(source_target_pairs)
count = 0
for i, (pop1, pop2) in enumerate(source_target_pairs):

    # Progress bar
    if i % 10 == 0:
        print(
            "Adding synapse groups... {}%     ".format(np.round(100 * i / num_pairs, 2)),
            end="\r",
        )

    # Load source_target df if previously saved
    synapse_group_name = pop1 + "_to_" + pop2
    synapse_group_path = Path("./pkl_data", "source_target_df", synapse_group_name, ".pkl")
    if synapse_group_path.exists():
        source_target = load_df(synapse_group_path)
    else:
        source_target = v1_edges_df[
            (v1_edges_df["source_model_name"] == pop1) & (v1_edges_df["target_model_name"] == pop2)
        ]
        save_df(source_target, synapse_group_path)

    # GeNN weight = product of syn_weight and number of synapses
    weight = (source_target["nsyns_x_syn_weight"] / 1e3).to_list()  # pA -> nA

    # Delay
    delay_ms = source_target["delay"]
    delay_steps = round((delay_ms / DT)).astype("int").to_list()
    assert len(delay_ms.unique()) == 1
    delay_steps = delay_steps[0]

    # Get list of source and target node ids (GeNN numbering)
    s_list = source_target[source_target["source_model_name"] == pop1]["source_GeNN_id"].tolist()
    t_list = source_target[source_target["target_model_name"] == pop2]["target_GeNN_id"].tolist()

    # Weight update model
    s_ini = {"g": weight, "d": delay_steps}  # , "d": delay_steps}

    # Postsynaptic current model
    psc_Alpha_params = {"tau": ALPHA_TAU}
    psc_Alpha_init = {"x": 0.0}

    # Add synapse population
    syn_dict[synapse_group_name] = model.add_synapse_population(
        pop_name=synapse_group_name,
        matrix_type="SPARSE_INDIVIDUALG",
        delay_steps=delay_steps,
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

    # syn_dict[synapse_group_name].pop.set_max_dendritic_delay_timesteps(
    #     max_dendritic_delay_slots
    # )

    syn_dict[synapse_group_name].set_sparse_connections(np.array(s_list), np.array(t_list))

print("Added all {} synapse groups.      ".format(i))

### Run simulation

import time

start = time.time()
model.build(force_rebuild=False)
stop = time.time()
print("Duration = {}s".format(stop - start))
model.load(num_recording_timesteps=NUM_RECORDING_TIMESTEPS)  # TODO: How big to calculate for GPU size?

for i in range(num_steps):

    model.step_time()

    # Only collect full BUFFER
    if i % NUM_RECORDING_TIMESTEPS == 0 and i != 0:
        print(i)
