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
df = pd.read_csv("point_450glifs/network/v1_node_types.csv", sep=" ")


# Get all nodes of certain type
net = v1_net
nodes = net.nodes["v1"]
dynamics_files = df["dynamics_params"].to_list()
model_names = df["model_name"].to_list()
node_dict = {}
for i, dynamics_file in enumerate(dynamics_files):
    nodes_with_model_name = [
        n["node_id"] for n in nodes.filter(dynamics_params=dynamics_file)
    ]
    node_dict[model_names[i]] = nodes_with_model_name

# read dT
with open(SIM_CONFIG_PATH) as f:
    sim_config = json.load(f)

# Create base model
model = GeNNModel("double", "v1", backend="SingleThreadedCPU")
model.dT = sim_config["run"]["dt"]

# Construct populations
pop_dict = {}
for i, model_name in enumerate(model_names):
    dynamics_file = dynamics_files[i]
    dynamics_path = Path(DYNAMICS_BASE_DIR, dynamics_file)
    with open(dynamics_path) as f:
        dynamics_params = json.load(f)
    num_neurons = len(node_dict[model_name])

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

    pop_dict[model_name] = model.add_neuron_population(
        pop_name=model_name,
        num_neurons=num_neurons,
        neuron=neuron_class,
        param_space=params,
        var_space=init,
    )

    # Add extra global params
    # Assign extra global parameter values
    for k in pop_dict[model_name].extra_global_params.keys():
        pop_dict[model_name].set_extra_global_param(k, dynamics_params_renamed[k])

    print("{} population added to model.".format(model_name))

# Dict: from node_id to population + idx (inside population)
node_to_pop_idx = {}
pop_counts = {}
for n in nodes:
    model_name = n["model_name"]
    if model_name in pop_counts.keys():
        pop_counts[model_name] += 1
    else:
        pop_counts[model_name] = 0
    pop_idx = pop_counts[model_name]
    node_id = n["node_id"]
    node_to_pop_idx[node_id] = [model_name, pop_idx]

# +1 so that pop_counts == num_neurons
for k in pop_counts.keys():
    pop_counts[k] += 1


# Add connections (synapses) between popluations
edges = net.edges["v1_to_v1"]
connections = edges.get_target(1)


# df - rows are synapses, columns are populations
edges_for_df = []
for e in edges:
    e_dict = {}

    # Add node_ids
    e_dict["source_node_id"] = e.source_node_id
    e_dict["target_node_id"] = e.target_node_id

    # Populate empty indices for each population
    for m in model_names:
        e_dict["source_" + m] = pd.NA
        e_dict["target_" + m] = pd.NA

    # Populate actual dicts
    [m_name, idx] = node_to_pop_idx[e.source_node_id]
    e_dict["source_" + m_name] = idx

    [m_name, idx] = node_to_pop_idx[e.target_node_id]
    e_dict["target_" + m_name] = idx

    edges_for_df.append(e_dict)
df = pd.DataFrame(edges_for_df)

synapse_dict = {}
for pop1 in model_names:
    for pop2 in model_names:
        src = df[
            ~df["source_" + pop1].isnull()
        ]  # Get synapses that have correct source
        src_tgt = src[~src["target_" + pop2].isnull()]

        s_list = src_tgt["source_" + pop1].tolist()
        t_list = src_tgt["target_" + pop2].tolist()
        s_ini = {"g": -0.2}
        ps_p = {
            "tau": 1.0,
            "E": -80.0,
        }  # Decay time constant [ms]  # Reversal potential [mV]

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

num_steps = 10
var_list = ["V"]
data = {
    model_name: {k: np.zeros((pop_counts[model_name], num_steps)) for k in var_list}
    for model_name in model_names
}
view_dict = {}
for model_name in model_names:
    pop = pop_dict[model_name]
    for v in var_list:
        view_dict[model_name] = {v: pop.vars[v].view}


for i in range(num_steps):
    model.step_time()

    for model_name in model_names:
        pop = pop_dict[model_name]

        v_view = pop.vars["V"].view
        for var_name in var_list:
            print(i, model_name, var_name, sep="\t")
            # pop.pull_var_from_device("V")
            view = view_dict[model_name][var_name]
            output = view[:]
            print(output)
            data[model_name][var_name][:, i] = output
print("Simulation complete.")
