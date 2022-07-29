# Replicate point_450glifs example with GeNN
import sys

sys.path.append("/home/williamsnider/Code/v1_project")
from GLIF_models import GLIF3
import glob
import pprint
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sonata.circuit import File
from pygenn.genn_model import (
    GeNNModel,
    create_custom_sparse_connect_init_snippet_class,
    create_cmlf_class,
    init_connectivity,
)

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
model = GeNNModel("double", "v1")
model.dT = sim_config["run"]["dt"]


# Construct populations
pop_list = []
for i, model_name in enumerate(model_names):
    dynamics_file = dynamics_files[i]
    dynamics_path = Path(DYNAMICS_BASE_DIR, dynamics_file)
    with open(dynamics_path) as f:
        dynamics_params = json.load(f)

    dynamics_params_renamed = {
        "C": dynamics_params["C_m"],
        "G": dynamics_params["g"],
        "El": dynamics_params["E_L"],
        "th_inf": dynamics_params["V_th"],
        "dT": sim_config["run"]["dt"],
        "V": dynamics_params["V_m"],
        "spike_cut_length": round(dynamics_params["t_ref"] / sim_config["run"]["dt"]),
        "refractory_countdown": -1,
        "k": dynamics_params["asc_decay"],
        "r": [1.0, 1.0],
        "ASC": dynamics_params["asc_init"],
        "asc_amp_array": dynamics_params["asc_amps"],
        "ASC_length": 2,
    }
    # tau_syn? used for synapses (see https://nest-simulator.readthedocs.io/en/v3.3/models/glif_psc.html#id13)
    # see citation #2

    neuron_class = GLIF3

    num_neurons = len(node_dict[model_name])
    params = {k: dynamics_params_renamed[k] for k in neuron_class.get_param_names()}
    init = {k: dynamics_params_renamed[k] for k in ["V", "refractory_countdown"]}

    pop_list.append(
        model.add_neuron_population(
            pop_name=model_name,
            num_neurons=num_neurons,
            neuron=neuron_class,
            param_space=params,
            var_space=init,
        )
    )

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

# Scnn1a to Scnn1a
pop1 = "Scnn1a"
pop2 = "Scnn1a"
src = df[~df["source_" + pop1].isnull()]  # Get synapses that have correct source
src_tgt = src[~src["target_" + pop2].isnull()]  # Get synapses that have correct target
pop1_pop2_synapses = {i: [] for i in range(pop_counts[pop1] + 1)}
s_list = src_tgt["source_" + pop1].tolist()
t_list = src_tgt["target_" + pop2].tolist()
for i, s in enumerate(s_list):
    pop1_pop2_synapses[s].append(t_list[i])

# Find max synapses
max_synapses = 0
for i in range(pop_counts[pop1] + 1):
    max_synapses = max(max_synapses, len(pop1_pop2_synapses[i]))

# Pad synapses
for k, v in pop1_pop2_synapses.items():

    num_pads_needed = max_synapses - len(v)
    for j in range(num_pads_needed):
        pop1_pop2_synapses[k].append(-1)

target_ids = []
for i in range(pop_counts[pop1] + 1):
    target_ids.extend(pop1_pop2_synapses[i])

# Q: What is $id_pre
connections_params = {"max_synapses": max_synapses}
connections_model = create_custom_sparse_connect_init_snippet_class(
    "connections",
    param_names=["max_synapses"],
    row_build_code="""
        int target;
        for (int jj = 0; jj<$(max_synapses); jj++){
            target = $(target_ids)[$(id_pre)*$(max_synapses)+jj];
        
            // target id of -1 signals no synaptic connection
            if (target != -1){
                $(addSynapse, target);
            }
        }
        $(endRow);
        """,
    calc_max_row_len_func=create_cmlf_class(
        lambda num_pre, num_post, pars: int(pars[0])
    )(),
    calc_max_col_len_func=create_cmlf_class(
        lambda num_pre, num_post, pars: int(pars[0])
    )(),
    extra_global_params=("target_ids", "int*"),
)

s_ini = {"g": -0.2}
ps_p = {"tau": 1.0, "E": -80.0}  # Decay time constant [ms]  # Reversal potential [mV]


synapse_group = model.add_synapse_population(
    "Pop1self",
    "SPARSE_GLOBALG",
    10,
    pop1,
    pop1,
    "StaticPulse",
    {},
    s_ini,
    {},
    {},
    "ExpCond",
    ps_p,
    {},
    init_connectivity(connections_model, connections_params),
)

# Add GLIF Class to model
# model = GeNNModel("double", GLIF_dict[model_type], backend="SingleThreadedCPU")
# model.dT = units_dict["dT"]
# pop1 = model.add_neuron_population(
#     pop_name="pop1",
#     num_neurons=num_neurons,
#     neuron=GLIF,
#     param_space=GLIF_params,
#     var_space=GLIF_init,
# )
# # Assign extra global parameter values
# for k in pop1.extra_global_params.keys():
#     pop1.set_extra_global_param(k, units_dict[k])

# # Add current source to model
# external_current_source = create_custom_current_source_class(
#     class_name="external_current",
#     var_name_types=[("current", "double")],
#     injection_code="""
#     int idx = round($(t) / DT);
#     $(current)=$(Ie)[idx];
#     $(injectCurrent,$(current) );
#     """,
#     extra_global_params=[("Ie", "double*")],
# )
# cs_ini = {"current": 0.0}  # external input current from Teeter 2018
# cs = model.add_current_source(
#     cs_name="external_current_source",
#     current_source_model=external_current_source,
#     pop=pop1,
#     param_space={},
#     var_space=cs_ini,
# )
# stimulus_conversion = 1e9  # amps -> nanoamps
# cs.set_extra_global_param("Ie", stimulus * stimulus_conversion)
