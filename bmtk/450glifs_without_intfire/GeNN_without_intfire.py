# Replicate point_450glifs example with GeNN
import sys

sys.path.append("../../")
from GLIF_models import GLIF3
import glob
import pprint
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sonata.circuit import File
from sonata.reports.spike_trains import SpikeTrains
from pygenn.genn_model import GeNNModel

DYNAMICS_BASE_DIR = Path("./point_components/cell_models")
SIM_CONFIG_PATH = Path("point_450glifs/config.simulation.json")

v1_net = File(
    data_files=[
        "point_450glifs/network/v1_nodes.h5",
        "point_450glifs/network/v1_v1_edges.h5",
    ],
    data_type_files=[
        "point_450glifs/network/v1_node_types.csv",
        "point_450glifs/network/v1_v1_edge_types.csv",
    ],
)

lgn_net = File(
    data_files=[
        "point_450glifs/network/lgn_nodes.h5",
        "point_450glifs/network/lgn_v1_edges.h5",
    ],
    data_type_files=[
        "point_450glifs/network/lgn_node_types.csv",
        "point_450glifs/network/lgn_v1_edge_types.csv",
    ],
)
print("Contains nodes: {}".format(v1_net.has_nodes))
print("Contains edges: {}".format(v1_net.has_edges))
print("Contains nodes: {}".format(lgn_net.has_nodes))
print("Contains edges: {}".format(lgn_net.has_edges))

# Read CSV to get node types
v1_df = pd.read_csv("point_450glifs/network/v1_node_types.csv", sep=" ")

# Get all nodes of certain type
v1_nodes = v1_net.nodes["v1"]
v1_dynamics_files = v1_df["dynamics_params"].to_list()
v1_model_names = v1_df["model_name"].to_list()
v1_node_dict = {}
for i, dynamics_file in enumerate(v1_dynamics_files):
    v1_nodes_with_model_name = [
        n["node_id"] for n in v1_nodes.filter(dynamics_params=dynamics_file)
    ]
    v1_node_dict[v1_model_names[i]] = v1_nodes_with_model_name

# read dT
with open(SIM_CONFIG_PATH) as f:
    sim_config = json.load(f)

# Create base model
model = GeNNModel("double", "v1", backend="SingleThreadedCPU")
model.dT = sim_config["run"]["dt"]

# Construct V1 populations
pop_dict = {}
for i, v1_model_name in enumerate(v1_model_names):
    v1_dynamics_file = v1_dynamics_files[i]
    v1_dynamics_path = Path(DYNAMICS_BASE_DIR, v1_dynamics_file)
    with open(v1_dynamics_path) as f:
        dynamics_params = json.load(f)
    num_neurons = len(v1_node_dict[v1_model_name])

    dynamics_params_renamed = {
        "C": dynamics_params["C_m"],
        "G": dynamics_params["g"],
        "El": dynamics_params["E_L"],
        "th_inf": dynamics_params["V_th"],
        "dT": sim_config["run"]["dt"],
        "V": dynamics_params["V_m"],
        "spike_cut_length": round(dynamics_params["t_ref"] / sim_config["run"]["dt"]),
        "refractory_countdown": -1,
        "k": np.repeat(dynamics_params["asc_decay"], num_neurons).ravel(),
        "r": np.repeat([1.0, 1.0], num_neurons).ravel(),
        "ASC": np.repeat(dynamics_params["asc_init"], num_neurons).ravel(),
        "asc_amp_array": np.repeat(dynamics_params["asc_amps"], num_neurons).ravel(),
        "ASC_length": 2,
    }
    # tau_syn? used for synapses (see https://nest-simulator.readthedocs.io/en/v3.3/models/glif_psc.html#id13)
    # see citation #2

    neuron_class = GLIF3

    params = {k: dynamics_params_renamed[k] for k in neuron_class.get_param_names()}
    init = {k: dynamics_params_renamed[k] for k in ["V", "refractory_countdown"]}

    pop_dict[v1_model_name] = model.add_neuron_population(
        pop_name=v1_model_name,
        num_neurons=num_neurons,
        neuron=neuron_class,
        param_space=params,
        var_space=init,
    )

    # Add extra global params
    # Assign extra global parameter values
    for k in pop_dict[v1_model_name].extra_global_params.keys():
        pop_dict[v1_model_name].set_extra_global_param(k, dynamics_params_renamed[k])

    print("{} population added to model.".format(v1_model_name))

# Add LGN Nodes
lgn_df = pd.read_csv("point_450glifs/network/lgn_node_types.csv", sep=" ")
lgn_nodes = lgn_net.nodes["lgn"]
# dynamics_files = df["dynamics_params"].to_list()
lgn_model_names = lgn_df["model_type"].to_list()
lgn_node_dict = {}
for i, lgn_model_name in enumerate(lgn_model_names):
    nodes_with_model_name = [
        n["node_id"] for n in lgn_nodes.filter(model_type=lgn_model_name)
    ]
    lgn_node_dict[lgn_model_names[i]] = nodes_with_model_name

# Load LGN spikes
spikes = SpikeTrains.from_sonata("point_450glifs/inputs/lgn_spikes.h5")
spikes_df = spikes.to_dataframe()

num_lgn = len(lgn_nodes)
spikes_list = []
for n in range(0, num_lgn):
    spikes_list.append(spikes_df[spikes_df["node_ids"] == n]["timestamps"].to_list())


def spikes_list_to_start_end_times(spikes_list):

    spike_counts = [len(n) for n in spikes_list]

    # Get start and end indices of each spike sources section
    end_spike = np.cumsum(spike_counts)
    start_spike = np.empty_like(end_spike)
    start_spike[0] = 0
    start_spike[1:] = end_spike[0:-1]

    spike_times = np.hstack(spikes_list)
    return start_spike, end_spike, spike_times


start_spike, end_spike, spike_times = spikes_list_to_start_end_times(spikes_list)

# Construct LGN population
for i, lgn_model_name in enumerate(lgn_model_names):
    num_neurons = len(lgn_node_dict[lgn_model_name])

    # Get spike data

    pop_dict[lgn_model_name] = model.add_neuron_population(
        lgn_model_name,
        num_neurons,
        "SpikeSourceArray",
        {},
        {"startSpike": start_spike, "endSpike": end_spike},
    )

    pop_dict[lgn_model_name].set_extra_global_param("spikeTimes", spike_times)

# Dict: from node_id to population + idx (inside population)
v1_node_to_pop_idx = {}
v1_pop_counts = {}
for n in v1_nodes:
    model_name = n["model_name"]
    if model_name in v1_pop_counts.keys():
        v1_pop_counts[model_name] += 1
    else:
        v1_pop_counts[model_name] = 0
    pop_idx = v1_pop_counts[model_name]
    node_id = n["node_id"]
    v1_node_to_pop_idx[node_id] = [model_name, pop_idx]

# +1 so that pop_counts == num_neurons
for k in v1_pop_counts.keys():
    v1_pop_counts[k] += 1


# Add connections (synapses) between popluations
v1_edges = v1_net.edges["v1_to_v1"]

# df - rows are synapses, columns are populations
v1_edges_for_df = []
for e in v1_edges:
    e_dict = {}

    # Add node_ids
    e_dict["source_node_id"] = e.source_node_id
    e_dict["target_node_id"] = e.target_node_id

    # Populate empty indices for each population
    for m in v1_model_names:
        e_dict["source_" + m] = pd.NA
        e_dict["target_" + m] = pd.NA

    # Populate actual dicts
    [m_name, idx] = v1_node_to_pop_idx[e.source_node_id]
    e_dict["source_" + m_name] = idx

    [m_name, idx] = v1_node_to_pop_idx[e.target_node_id]
    e_dict["target_" + m_name] = idx

    v1_edges_for_df.append(e_dict)
v1_edge_df = pd.DataFrame(v1_edges_for_df)

synapse_dict = {}
for pop1 in v1_model_names:
    for pop2 in v1_model_names:
        src = v1_edge_df[
            ~v1_edge_df["source_" + pop1].isnull()
        ]  # Get synapses that have correct source
        src_tgt = src[~src["target_" + pop2].isnull()]

        s_list = src_tgt["source_" + pop1].tolist()
        t_list = src_tgt["target_" + pop2].tolist()
        s_ini = {"g": -0.2}
        ps_p = {
            "tau": 1.0,
            "E": -80.0,
        }  # Decay time constant [ms]  # Reversal potential [mV]

        # TODO: Add synaptic weight and delay
        # TODO: Have alpha synapses
        synapse_group_name = pop1 + "_to_" + pop2
        synapse_group = model.add_synapse_population(
            synapse_group_name,
            "SPARSE_GLOBALG",
            10,
            pop1,
            pop2,
            "StaticPulse",
            {},
            s_ini,
            {},
            {},
            "ExpCond",
            ps_p,
            {},
        )
        synapse_group.set_sparse_connections(np.array(s_list), np.array(t_list))
        synapse_dict[synapse_group_name] = synapse_group
        print("Synapses added for {} -> {}".format(pop1, pop2))


# Add Synapses for LGN --> V1
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


# df - rows are synapses, columns are populations
lgn_edges_for_df = []
for e in lgn_edges:
    e_dict = {}

    # Add node_ids
    e_dict["source_node_id"] = e.source_node_id
    e_dict["target_node_id"] = e.target_node_id

    # Populate empty indices for each population
    for m in lgn_model_names:
        e_dict["source_" + m] = pd.NA
    for m in v1_model_names:
        e_dict["target_" + m] = pd.NA

    # Populate actual dicts
    [m_name, idx] = lgn_node_to_pop_idx[e.source_node_id]
    e_dict["source_" + m_name] = idx

    [m_name, idx] = v1_node_to_pop_idx[e.target_node_id]
    e_dict["target_" + m_name] = idx

    lgn_edges_for_df.append(e_dict)
lgn_edge_df = pd.DataFrame(lgn_edges_for_df)


for pop1 in lgn_model_names:
    for pop2 in v1_model_names:
        src = lgn_edge_df[
            ~lgn_edge_df["source_" + pop1].isnull()
        ]  # Get synapses that have correct source
        src_tgt = src[~src["target_" + pop2].isnull()]

        s_list = src_tgt["source_" + pop1].tolist()
        t_list = src_tgt["target_" + pop2].tolist()
        s_ini = {"g": -0.2}
        ps_p = {
            "tau": 1.0,
            "E": -80.0,
        }  # Decay time constant [ms]  # Reversal potential [mV]

        # TODO: Add synaptic weight and delay
        # TODO: Have alpha synapses
        synapse_group_name = pop1 + "_to_" + pop2
        synapse_group = model.add_synapse_population(
            synapse_group_name,
            "SPARSE_GLOBALG",
            10,
            pop1,
            pop2,
            "StaticPulse",
            {},
            s_ini,
            {},
            {},
            "ExpCond",
            ps_p,
            {},
        )
        synapse_group.set_sparse_connections(np.array(s_list), np.array(t_list))
        synapse_dict[synapse_group_name] = synapse_group
        print("Synapses added for {} -> {}".format(pop1, pop2))


model.build(force_rebuild=True)
model.load()

num_steps = 10000
var_list = ["V"]
data = {
    model_name: {k: np.zeros((v1_pop_counts[model_name], num_steps)) for k in var_list}
    for model_name in v1_model_names
}
view_dict = {}
for model_name in v1_model_names:
    pop = pop_dict[model_name]
    for v in var_list:
        view_dict[model_name] = {v: pop.vars[v].view}


for i in range(num_steps):
    print(i)
    model.step_time()

    for model_name in v1_model_names:
        pop = pop_dict[model_name]

        v_view = pop.vars["V"].view
        for var_name in var_list:
            # print(i, model_name, var_name, sep="\t")
            # pop.pull_var_from_device("V")
            view = view_dict[model_name][var_name]
            output = view[:]
            # print(output)
            data[model_name][var_name][:, i] = output


print("Simulation complete.")
