# Replicate point_450glifs example with GeNN
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sonata.circuit import File
from sonata.reports.spike_trains import SpikeTrains
import pygenn
import matplotlib.pyplot as plt
from utilities import (
    GLIF3,
    get_dynamics_params,
    spikes_list_to_start_end_times,
    psc_Alpha,
    construct_populations,
    construct_synapses,
    construct_id_conversion_df,
    add_model_name_to_df,
    add_GeNN_id,
)
import pickle

DYNAMICS_BASE_DIR = Path("./GLIF Network/models/cell_models/nest_2.14_models")
SIM_CONFIG_PATH = Path("./GLIF Network/config.json")
LGN_V1_EDGE_CSV = Path("./GLIF Network/network/lgn_v1_edge_types.csv")
V1_EDGE_CSV = Path("./GLIF Network/network/v1_v1_edge_types.csv")
LGN_SPIKES_PATH = Path(
    "GLIF Network/inputs/full3_GScorrected_PScorrected_3.0sec_SF0.04_TF2.0_ori270.0_c100.0_gs0.5_spikes.trial_0.h5"
)
LGN_NODE_DIR = Path("./GLIF Network/network/lgn_node_types.csv")
V1_NODE_CSV = Path("./GLIF Network/network/v1_node_types.csv")
V1_ID_CONVERSION_FILENAME = Path(".", "pkl_data", "v1_edge_df.pkl")
LGN_ID_CONVERSION_FILENAME = Path(".", "pkl_data", "lgn_edge_df.pkl")
BKG_V1_EDGE_CSV = Path("./GLIF Network/network/bkg_v1_edge_types.csv")
BKG_ID_CONVERSION_FILENAME = Path(".", "pkl_data", "bkg_edge_df.pkl")
NUM_RECORDING_TIMESTEPS = 10000
num_steps = 300000


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

print("Contains nodes: {}".format(v1_net.has_nodes))
print("Contains edges: {}".format(v1_net.has_edges))
print("Contains nodes: {}".format(lgn_net.has_nodes))
print("Contains edges: {}".format(lgn_net.has_edges))


### Create base model ###
with open(SIM_CONFIG_PATH) as f:
    sim_config = json.load(f)
model = pygenn.genn_model.GeNNModel(backend="CUDA")
model.dT = sim_config["run"]["dt"]

### Construct v1 neuron populations ###
v1_node_types_df = pd.read_csv(V1_NODE_CSV, sep=" ")
v1_nodes = v1_net.nodes["v1"]
# v1_dynamics_files = v1_node_types_df["dynamics_params"].to_list()

# # Add model names column to handle duplicate pop_names (with different dynamics params)
# model_column = ["_" for _ in range(v1_node_types_df.shape[0])]
# v1_node_types_df["model_name"] = model_column
# for i in range(v1_node_types_df.shape[0]):
#     suffix = 0
#     pop_name = v1_node_types_df.iloc[i]["pop_name"]
#     while True:
#         new_name = pop_name + "_" + str(suffix)
#         if new_name not in v1_node_types_df["model_name"].tolist():
#             break
#         suffix += 1

#     v1_node_types_df["model_name"][i] = new_name


# v1_model_names = v1_node_types_df["model_name"].to_list()
# v1_node_dict = {}
# for i, dynamics_file in enumerate(v1_dynamics_files):
#     print(i, end="\r")
#     v1_nodes_with_model_name = [
#         n["node_id"] for n in v1_nodes.filter(dynamics_params=dynamics_file)
#     ]
#     v1_node_dict[v1_model_names[i]] = v1_nodes_with_model_name

# for i, _ in enumerate(v1_nodes.filter(dynamics_params=dynamics_file)):
#     print(i)

# for i, type_id in enumerate(np.unique(v1_nodes.type_ids)):
#     v1_node_dict[type_id] = [n for n in v1_nodes.filter(node_type_id=type_id)]

v1_node_df_path = Path("./pkl_data/v1_node_df.pkl")
if v1_node_df_path.exists():
    with open(v1_node_df_path, "rb") as f:
        v1_node_df = pickle.load(f)
else:
    v1_node_df = v1_nodes.to_dataframe()

    # Add model_name column to account for duplicate pop_names (which have different dynamics_parameters)
    v1_node_df = add_model_name_to_df(v1_node_df)

    # Add GeNN id
    v1_node_df = add_GeNN_id(v1_node_df)

    # Improve memory of node_df
    v1_node_df.drop(
        columns=[
            "model_template",
            "model_type",
            "tuning_angle",
            "x",
            "y",
            "z",
            "gaba_synapse",
        ],
        inplace=True,
    )
    for k in [
        "dynamics_params",
        "pop_name",
        "model_name",
        "population",
        "location",
        "ei",
        "model_name",
    ]:
        v1_node_df[k] = v1_node_df[k].astype("category")

    # Save as pickle
    if v1_node_df_path.parent.exists() == False:
        Path.mkdir(v1_node_df_path.parent, parents=True)

    with open(v1_node_df_path, "wb") as f:
        pickle.dump(v1_node_df, f)
v1_model_names = v1_node_df["model_name"].unique()


# Add populations
pop_dict = {}
pop_dict = construct_populations(
    model,
    pop_dict,
    all_model_names=v1_model_names,
    dynamics_base_dir=DYNAMICS_BASE_DIR,
    node_types_df=v1_node_types_df,
    neuron_class=GLIF3,
    sim_config=sim_config,
    node_df=v1_node_df,
)

# Enable spike recording
for k in pop_dict.keys():
    pop_dict[k].spike_recording_enabled = True

### Construct LGN neuron populations ###
# lgn_node_types_df = pd.read_csv(LGN_NODE_DIR, sep=" ")

lgn_node_df_path = Path("./pkl_data/lgn_node_df.pkl")
if lgn_node_df_path.exists():
    with open(lgn_node_df_path, "rb") as f:
        lgn_node_df = pickle.load(f)
else:
    lgn_nodes = lgn_net.nodes["lgn"]
    lgn_node_df = lgn_nodes.to_dataframe()

    # Add model_name column to account for duplicate pop_names (which have different dynamics_parameters)
    lgn_node_df = add_model_name_to_df(lgn_node_df)
    lgn_model_names = lgn_node_df["model_name"].unique()

    # Save as pickle
    if v1_node_df_path.parent.exists() == False:
        Path.mkdir(lgn_node_df_path.parent, parents=True)

    with open(lgn_node_df_path, "wb") as f:
        pickle.dump(lgn_node_df, f)

lgn_model_names = lgn_node_df["model_name"].unique()

# lgn_model_names = lgn_node_types_df["model_type"].to_list()
# lgn_node_dict = {}
# for i, lgn_model_name in enumerate(lgn_model_names):
#     nodes_with_model_name = [
#         n["node_id"] for n in lgn_nodes.filter(model_type=lgn_model_name)
#     ]
#     lgn_node_dict[lgn_model_names[i]] = nodes_with_model_name

spikes_path = Path("./pkl_data/spikes.pkl")
if spikes_path.exists():
    with open(spikes_path, "rb") as f:
        spikes = pickle.load(f)
else:
    spikes_from_sonata = SpikeTrains.from_sonata(LGN_SPIKES_PATH)
    spikes_df = spikes_from_sonata.to_dataframe()

    lgn_spiking_nodes = spikes_df["node_ids"].unique().tolist()
    spikes_list = []
    for n in lgn_spiking_nodes:
        spikes_list.append(
            spikes_df[spikes_df["node_ids"] == n]["timestamps"].to_list()
        )
    start_spike, end_spike, spike_times = spikes_list_to_start_end_times(
        spikes_list
    )  # Convert to GeNN format

    # Group into one list
    spikes = [start_spike, end_spike, spike_times]

    # Save as pickle
    if spikes_path.parent.exists() == False:
        Path.mkdir(spikes_path.parent, parents=True)

    with open(spikes_path, "wb") as f:
        pickle.dump(spikes, f)

(start_spike, end_spike, spike_times) = spikes

# # Read LGN spike times
# spikes = SpikeTrains.from_sonata(LGN_SPIKES_PATH)
# spikes_df = spikes.to_dataframe()
# lgn_spiking_nodes = spikes_df["node_ids"].unique().tolist()
# spikes_list = []
# for n in lgn_spiking_nodes:
#     spikes_list.append(spikes_df[spikes_df["node_ids"] == n]["timestamps"].to_list())
# start_spike, end_spike, spike_times = spikes_list_to_start_end_times(
#     spikes_list
# )  # Convert to GeNN format

# Add population
for i, lgn_model_name in enumerate(lgn_model_names):
    num_neurons = lgn_node_df[lgn_node_df["model_name"] == lgn_model_name].shape[0]

    pop_dict[lgn_model_name] = model.add_neuron_population(
        lgn_model_name,
        num_neurons,
        "SpikeSourceArray",
        {},
        {"startSpike": start_spike, "endSpike": end_spike},
    )

    pop_dict[lgn_model_name].set_extra_global_param("spikeTimes", spike_times)
    print("Added {}".format(lgn_model_name))

### Construct BKG neuron population ###
BKG_name = "BKG"
BKG_params = {"rate": 1000}  # 1kHz
BKG_var = {"timeStepToSpike": 0}
pop_dict[BKG_name] = model.add_neuron_population(
    BKG_name,
    num_neurons=1,
    neuron="PoissonNew",
    param_space=BKG_params,
    var_space=BKG_var,
)
print("Added BKG")

### Construct v1 to v1 synapses ###
syn_dict = {}

# First create a dict that maps the NEST node_id to the GeNN node_id. NEST numbers the neurons from 0 to num_neurons, whereas GeNN numbers neurons 0 to num_neurons_per_population. This matters when assigning synapses.
# v1_node_to_pop_idx = {}
# v1_pop_counts = {}
# for n in v1_nodes:
#     model_name = n["model_name"]
#     if model_name in v1_pop_counts.keys():
#         v1_pop_counts[model_name] += 1
#     else:
#         v1_pop_counts[model_name] = 0
#     pop_idx = v1_pop_counts[model_name]
#     node_id = n["node_id"]
#     v1_node_to_pop_idx[node_id] = [model_name, pop_idx]

# # +1 so that pop_counts == num_neurons
# for k in v1_pop_counts.keys():
#     v1_pop_counts[k] += 1


v1_edge_df_path = Path("./pkl_data/v1_edge_df.pkl")
if v1_edge_df_path.exists():
    with open(v1_edge_df_path, "rb") as f:
        v1_edge_df = pickle.load(f)
else:

    # Add connections (synapses) between popluations
    v1_edges = v1_net.edges["v1_to_v1"]
    v1_edge_df = v1_edges.groups[0].to_dataframe()
    num_edges = len(v1_edge_df)

    # Remove unused columns only for memory efficiency
    v1_edge_df.drop(
        columns=[
            "target_query",
            "source_query",
            "weight_function",
            "weight_sigma",
            "dynamics_params",
            "model_template",
        ],
        inplace=True,
    )

    v1_edge_df["source_GeNN_id"] = (
        v1_node_df["GeNN_id"]
        .iloc[v1_edge_df["source_node_id"]]
        .astype("int32")
        .tolist()
    )
    v1_edge_df["target_GeNN_id"] = (
        v1_node_df["GeNN_id"]
        .iloc[v1_edge_df["target_node_id"]]
        .astype("int32")
        .tolist()
    )
    v1_edge_df["source_model_name"] = (
        v1_node_df["model_name"].iloc[v1_edge_df["source_node_id"]].tolist()
    )
    v1_edge_df["target_model_name"] = (
        v1_node_df["model_name"].iloc[v1_edge_df["target_node_id"]].tolist()
    )

    # Convert to categorical to save memory
    for k in ["source_model_name", "target_model_name", "nsyns"]:
        v1_edge_df[k] = v1_edge_df[k].astype("category")

    # Downcast to save memory
    for k in [
        "edge_type_id",
        "source_node_id",
        "target_node_id",
        "source_GeNN_id",
        "target_GeNN_id",
    ]:
        v1_edge_df[k] = pd.to_numeric(v1_edge_df[k], downcast="unsigned")

    for k in ["delay", "syn_weight"]:
        v1_edge_df[k] = pd.to_numeric(v1_edge_df[k], downcast="float")

    # Save as pickle
    if v1_edge_df_path.parent.exists() == False:
        Path.mkdir(v1_edge_df_path.parent, parents=True)

    with open(v1_edge_df_path, "wb") as f:
        pickle.dump(v1_edge_df, f)

    # for i, node_id in enumerate(node_df["node_id"].tolist()):
    #     node_id = np.array(node_id, dtype="uint64")
    #     GeNN_id = node_df.loc[node_df["node_id"] == node_id, "GeNN_id"].iloc[0]
    #     node_model_name = node_df.loc[node_df["node_id"] == node_id, "model_name"].iloc[0]
    #     # TODO: Why is this crashing
    #     edge_df.loc[
    #         edge_df["source_node_id"] == node_id, "source_model_name"
    #     ] = node_model_name
    #     edge_df.loc[edge_df["source_node_id"] == node_id, "source_GeNN_id"] = GeNN_id
    #     edge_df.loc[
    #         edge_df["target_node_id"] == node_id, "target_model_name"
    #     ] = node_model_name
    #     edge_df.loc[edge_df["target_node_id"] == node_id, "target_GeNN_id"] = GeNN_id
    #     # print progress
    #     if i % 10 == 0:
    #         print("Progress: {}%".format(np.round(i / num_edges * 100, 10)), end="\r")
    # Mapping from BMTK node_id to GeNN model name and GeNN


# v1_edge_df = construct_id_conversion_df(
#     edges=v1_edges,
#     all_model_names=v1_model_names,
#     source_node_to_pop_idx_dict=v1_node_to_pop_idx,
#     target_node_to_pop_idx_dict=v1_node_to_pop_idx,
#     filename=V1_ID_CONVERSION_FILENAME,
# )
# v1_syn_df = pd.read_csv(V1_EDGE_CSV, sep=" ")
# v1_edge_type_ids = v1_syn_df["edge_type_id"].tolist()
# v1_all_nsyns = v1_edge_df["nsyns"].unique()
# v1_all_nsyns.sort()
# def generate_source_target_df(node_df):
edge_df = v1_edge_df
source_model_names = edge_df["source_model_name"].unique().tolist()
target_model_names = edge_df["target_model_name"].unique().tolist()
all_model_names = list(set(source_model_names) | set(target_model_names))
# for pop1 in all_model_names:
#     for pop2 in all_model_names:

#         src_tgt_path = Path("./pkl_data/src_tgt/{}_{}.pkl".format(pop1, pop2))
#         if src_tgt_path.exists() == False:
#             src_tgt = edge_df.loc[
#                 (edge_df["source_model_name"] == pop1)
#                 & (edge_df["target_model_name"] == pop2)
#             ]
#         # Save as pickle
#         if src_tgt_path.parent.exists() == False:
#             Path.mkdir(src_tgt_path.parent, parents=True)

#         with open(src_tgt_path, "wb") as f:
#             pickle.dump(src_tgt, f)


def make_src_tgt_df(arg_list):
    (pop1, pop2, edge_df) = arg_list
    src_tgt = edge_df.loc[
        (edge_df["source_model_name"] == pop1) & (edge_df["target_model_name"] == pop2)
    ]
    # Save as pickle
    if src_tgt_path.parent.exists() == False:
        Path.mkdir(src_tgt_path.parent, parents=True)

    with open(src_tgt_path, "wb") as f:
        pickle.dump(src_tgt, f)

    print(src_tgt_path)


import multiprocessing

items = []
for pop1 in all_model_names:
    for pop2 in all_model_names:
        src_tgt_path = Path("./pkl_data/src_tgt/{}_{}.pkl".format(pop1, pop2))
        if src_tgt_path.exists() == False:
            items.append(pop1, pop2, edge_df)

with multiprocessing.Pool() as pool:
    pool.map(make_src_tgt_df, items)


df = edge_df.drop_duplicates(
    subset=["edge_type_id", "nsyns", "source_model_name", "target_model_name"]
)
# Add DT
df["DT"] = sim_config["run"]["dt"]
df["dynamics_path"] = "_"

# Add dynamics file
for i in range(len(df)):
    if i % 1000 == 0:
        print(i)
    target = df.iloc[0]["target_model_name"]
    dynamics_file = v1_node_df.loc[v1_node_df["model_name"] == target][
        "dynamics_params"
    ].iloc[0]
    dynamics_file = dynamics_file.replace("config", "psc")
    dynamics_path = Path(DYNAMICS_BASE_DIR, dynamics_file)
    df["dynamics_path"].iloc[0] = dynamics_path
items = df[
    [
        "source_model_name",
        "target_model_name",
        "edge_type_id",
        "nsyns",
        "DT",
        "dynamics_path",
    ]
].to_numpy()
#


from utilities import make_synapse_data

with multiprocessing.Pool() as pool:
    pool.map(make_synapse_data, items)

# all_pop_names = set(
#     edge_df["source_model_name"].unique().tolist()
#     + edge_df["target_model_name"].unique().tolist(),
# )


node_df = v1_node_df
count = -1
for pop1 in v1_model_names:
    for pop2 in v1_model_names:

        # Print progress
        count += 1
        print(
            "Progress = {}% - {} to {}".format(
                np.round(100 * count / len(v1_model_names) ** 2, 4), pop1, pop2
            ),
            end="\r",
        )

        dynamics_file = node_df.loc[node_df["model_name"] == pop2][
            "dynamics_params"
        ].unique()
        assert len(dynamics_file) == 1
        dynamics_file = dynamics_file[0]
        dynamics_file = dynamics_file.replace("config", "psc")
        dynamics_path = Path(DYNAMICS_BASE_DIR, dynamics_file)

        dynamics_params = get_dynamics_params(dynamics_path, sim_config)

        syn_dict = construct_synapses(
            model=model,
            syn_dict=syn_dict,
            pop1=pop1,
            pop2=pop2,
            edge_df=v1_edge_df,
            sim_config=sim_config,
            dynamics_params=dynamics_params,
        )

### Construct LGN to v1 synapses ###
# First create a dict that maps the NEST node_id to the GeNN node_id. NEST numbers the neurons from 0 to num_neurons, whereas GeNN numbers neurons 0 to num_neurons_per_population. This matters when assigning synapses.
lgn_node_to_pop_idx = {}
lgn_pop_counts = {}
for n in lgn_nodes:
    model_name = n["model_type"]
    if model_name in lgn_pop_counts.keys():
        lgn_pop_counts[model_name] += 1
    else:
        lgn_pop_counts[model_name] = 0
    pop_idx = lgn_pop_counts[model_name]
    node_id = n["node_id"]
    lgn_node_to_pop_idx[node_id] = [model_name, pop_idx]

# +1 so that pop_counts == num_neurons
for k in lgn_pop_counts.keys():
    lgn_pop_counts[k] += 1

# Add connections (synapses) between popluations
lgn_edges = lgn_net.edges["lgn_to_v1"].get_group(0)
lgn_edge_df = construct_id_conversion_df(
    edges=lgn_edges,
    all_model_names=v1_model_names,
    source_node_to_pop_idx_dict=lgn_node_to_pop_idx,
    target_node_to_pop_idx_dict=v1_node_to_pop_idx,
    filename=LGN_ID_CONVERSION_FILENAME,
)

lgn_syn_df = pd.read_csv(LGN_V1_EDGE_CSV, sep=" ")
lgn_edge_type_ids = lgn_syn_df["edge_type_id"].tolist()
lgn_all_nsyns = lgn_edge_df["nsyns"].unique()
lgn_all_nsyns.sort()
for pop1 in lgn_model_names:
    for pop2 in v1_model_names:

        # Dynamics for v1, since this is the target
        dynamics_params, _ = get_dynamics_params(
            node_types_df=v1_node_types_df,
            dynamics_base_dir=DYNAMICS_BASE_DIR,
            sim_config=sim_config,
            node_dict=v1_node_dict,
            model_name=pop2,  # Pop2 is target, used for dynamics_params (tau)
        )
        syn_dict = construct_synapses(
            model=model,
            syn_dict=syn_dict,
            pop1=pop1,
            pop2=pop2,
            all_edge_type_ids=lgn_edge_type_ids,
            all_nsyns=lgn_all_nsyns,
            edge_df=lgn_edge_df,
            syn_df=lgn_syn_df,
            sim_config=sim_config,
            dynamics_params=dynamics_params,
        )

### Construct BKG to v1 synapses ###

# Test BKG working with connection to all v1 with same weights
# Get delay and weight specific to the edge_type_id
delay_steps = int(1.0 / sim_config["run"]["dt"])  # delay (ms) -> delay (steps)
nsyns = 21
weight = 0.192834123607 / 1e3 * nsyns  # nS -> uS; multiply by number of synapses
s_ini = {"g": weight}
psc_Alpha_params = {"tau": dynamics_params["tau"]}  # TODO: Always 0th port?
psc_Alpha_init = {"x": 0.0}
pop1 = BKG_name
for pop2 in v1_model_names:
    synapse_group_name = pop1 + "_to_" + pop2 + "_nsyns_" + str(nsyns)
    syn_dict[synapse_group_name] = model.add_synapse_population(
        pop_name=synapse_group_name,
        matrix_type="SPARSE_GLOBALG_INDIVIDUAL_PSM",
        delay_steps=delay_steps,
        source=pop1,
        target=pop2,
        w_update_model="StaticPulse",
        wu_param_space={},
        wu_var_space=s_ini,
        wu_pre_var_space={},
        wu_post_var_space={},
        postsyn_model=psc_Alpha,
        ps_param_space=psc_Alpha_params,
        ps_var_space=psc_Alpha_init,
    )

    t_list = [i for i in range(pop_dict[pop2].size)]
    s_list = [0 for i in t_list]
    syn_dict[synapse_group_name].set_sparse_connections(
        np.array(s_list), np.array(t_list)
    )
    print("Synapses added for {} -> {} with nsyns={}".format(pop1, pop2, nsyns))

### This commented out section is hard to run on this smaller dataset of 450 neurons because there is no bkg_nodes.h5 file for this dataset. Trying to copy the one from the Billeh dataset doesn't work, as it uses populations of neurons with different names.
# bkg_node_to_pop_idx = {0: [BKG_name, 0]}
# bkg_edges = bkg_net.edges["bkg_to_v1"].get_group(0)
# bkg_edge_df = construct_id_conversion_df(
#     edges=bkg_edges,
#     all_model_names=v1_model_names,
#     source_node_to_pop_idx_dict=bkg_node_to_pop_idx,
#     target_node_to_pop_idx_dict=v1_node_to_pop_idx,
#     filename=BKG_ID_CONVERSION_FILENAME,
# )

# pop1 = BKG_name
# bkg_syn_df = pd.read_csv(BKG_V1_EDGE_CSV, sep=" ")
# bkg_edge_type_ids = bkg_syn_df["edge_type_id"].tolist()
# bkg_all_nsyns = bkg_edge_df["nsyns"].unique()
# bkg_all_nsyns.sort()

# for pop2 in v1_model_names:

#     # Dynamics for v1, since this is the target
#     dynamics_params, _ = get_dynamics_params(
#         node_types_df=v1_node_types_df,
#         dynamics_base_dir=DYNAMICS_BASE_DIR,
#         sim_config=sim_config,
#         node_dict=v1_node_dict,
#         model_name=pop2,  # Pop2 is target, used for dynamics_params (tau)
#     )
#     syn_dict = construct_synapses(
#         model=model,
#         syn_dict=syn_dict,
#         pop1=pop1,
#         pop2=pop2,
#         all_edge_type_ids=bkg_edge_type_ids,
#         all_nsyns=bkg_all_nsyns,
#         edge_df=bkg_syn_df,
#         syn_df=bkg_syn_df,
#         sim_config=sim_config,
#         dynamics_params=dynamics_params,
#     )


### Run simulation ###
model.build(force_rebuild=True)
model.load(
    num_recording_timesteps=NUM_RECORDING_TIMESTEPS
)  # TODO: How big to calculate for GPU size?
1

# Construct data for spike times
spike_data = {}
for model_name in v1_model_names:
    spike_data[model_name] = {}
    num_neurons = v1_pop_counts[model_name]
    for i in range(num_neurons):
        spike_data[model_name][i] = []  # List of spike times for each neuron


for i in range(num_steps):

    model.step_time()

    # Only collect full BUFFER
    if i % NUM_RECORDING_TIMESTEPS == 0 and i != 0:

        # Record spikes
        print(i)
        model.pull_recording_buffers_from_device()
        for model_name in v1_model_names:
            pop = pop_dict[model_name]
            spk_times, spk_ids = pop.spike_recording_data
            for j, id in enumerate(spk_ids):
                spike_data[model_name][id].append(spk_times[j])

# Convert to BMTK node_ids
spike_data_BMTK_ids = {}
for BMTK_id, (model_name, model_id) in v1_node_to_pop_idx.items():
    spike_data_BMTK_ids[BMTK_id] = spike_data[model_name][model_id]

v1_node_to_pop_idx_inv = {}
for BMTK_id, pop_id_string in v1_node_to_pop_idx.items():
    v1_node_to_pop_idx_inv[str(pop_id_string)] = BMTK_id

# Plot firing rates
fig, axs = plt.subplots(1, 1)
v1_model_names.sort()
for model_name in v1_model_names:
    firing_rates = []
    ids = []
    for id, times in spike_data[model_name].items():

        # Convert to BMTK id
        BMTK_id = v1_node_to_pop_idx_inv[str([model_name, id])]
        ids.append(BMTK_id)

        # Calculate firing rate
        num_spikes = len(times)
        period_length = num_steps / 1e6  # s
        firing_rate = num_spikes / period_length
        firing_rates.append(firing_rate)
    axs.plot(ids, firing_rates, "o", label=model_name)

axs.set_ylabel("Firing Rate (hz)")
axs.set_xlabel("node_id")
axs.legend()
plt.show()

print("Simulation complete.")
