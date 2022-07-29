# Replicate point_450glifs example with GeNN
import sys

sys.path.append("..")
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
