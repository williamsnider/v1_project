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
)

DYNAMICS_BASE_DIR = Path("./point_components/cell_models")
SIM_CONFIG_PATH = Path("point_450glifs/config.simulation.json")
LGN_V1_EDGE_CSV = Path("./point_450glifs/network/lgn_v1_edge_types.csv")
V1_EDGE_CSV = Path("./point_450glifs/network/v1_v1_edge_types.csv")
LGN_SPIKES_PATH = Path("./point_450glifs/inputs/lgn_spikes.h5")
LGN_NODE_DIR = Path("./point_450glifs/network/lgn_node_types.csv")
V1_NODE_CSV = Path("./point_450glifs/network/v1_node_types.csv")


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


### Create base model ###
with open(SIM_CONFIG_PATH) as f:
    sim_config = json.load(f)
model = pygenn.genn_model.GeNNModel()
model.dT = sim_config["run"]["dt"]

### Construct v1 neuron populations ###
v1_node_types_df = pd.read_csv(V1_NODE_CSV, sep=" ")
v1_nodes = v1_net.nodes["v1"]
v1_dynamics_files = v1_node_types_df["dynamics_params"].to_list()
v1_model_names = v1_node_types_df["model_name"].to_list()
v1_node_dict = {}
for i, dynamics_file in enumerate(v1_dynamics_files):
    v1_nodes_with_model_name = [
        n["node_id"] for n in v1_nodes.filter(dynamics_params=dynamics_file)
    ]
    v1_node_dict[v1_model_names[i]] = v1_nodes_with_model_name

# Add populations
pop_dict = {}
pop_dict = construct_populations(
    model,
    pop_dict,
    all_model_names=v1_model_names,
    node_dict=v1_node_dict,
    dynamics_base_dir=DYNAMICS_BASE_DIR,
    node_types_df=v1_node_types_df,
    neuron_class=GLIF3,
    sim_config=sim_config,
)

# Enable spike recording
for k in pop_dict.keys():
    pop_dict[k].spike_recording_enabled = True

### Construct LGN neuron populations ###
lgn_node_types_df = pd.read_csv(LGN_NODE_DIR, sep=" ")
lgn_nodes = lgn_net.nodes["lgn"]
lgn_model_names = lgn_node_types_df["model_type"].to_list()
lgn_node_dict = {}
for i, lgn_model_name in enumerate(lgn_model_names):
    nodes_with_model_name = [
        n["node_id"] for n in lgn_nodes.filter(model_type=lgn_model_name)
    ]
    lgn_node_dict[lgn_model_names[i]] = nodes_with_model_name

# Read LGN spike times
spikes = SpikeTrains.from_sonata(LGN_SPIKES_PATH)
spikes_df = spikes.to_dataframe()
lgn_spiking_nodes = spikes_df["node_ids"].unique().tolist()
spikes_list = []
for n in lgn_spiking_nodes:
    spikes_list.append(spikes_df[spikes_df["node_ids"] == n]["timestamps"].to_list())
start_spike, end_spike, spike_times = spikes_list_to_start_end_times(
    spikes_list
)  # Convert to GeNN format

# Add population
for i, lgn_model_name in enumerate(lgn_model_names):
    num_neurons = len(lgn_node_dict[lgn_model_name])

    pop_dict[lgn_model_name] = model.add_neuron_population(
        lgn_model_name,
        num_neurons,
        "SpikeSourceArray",
        {},
        {"startSpike": start_spike, "endSpike": end_spike},
    )

    pop_dict[lgn_model_name].set_extra_global_param("spikeTimes", spike_times)


### Construct v1 to v1 synapses ###
syn_dict = {}

# First create a dict that maps the NEST node_id to the GeNN node_id. NEST numbers the neurons from 0 to num_neurons, whereas GeNN numbers neurons 0 to num_neurons_per_population. This matters when assigning synapses.
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
v1_edge_df = construct_id_conversion_df(
    edges=v1_edges,
    all_model_names=v1_model_names,
    source_node_to_pop_idx_dict=v1_node_to_pop_idx,
    target_node_to_pop_idx_dict=v1_node_to_pop_idx,
)
v1_syn_df = pd.read_csv(V1_EDGE_CSV, sep=" ")
v1_edge_type_ids = v1_syn_df["edge_type_id"].tolist()
v1_all_nsyns = v1_edge_df["nsyns"].unique()
v1_all_nsyns.sort()

for pop1 in v1_model_names:
    for pop2 in v1_model_names:

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
            all_edge_type_ids=v1_edge_type_ids,
            all_nsyns=v1_all_nsyns,
            edge_df=v1_edge_df,
            syn_df=v1_syn_df,
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

### Run simulation ###

NUM_RECORDING_TIMESTEPS = 1000
model.build(force_rebuild=True)
model.load(
    num_recording_timesteps=NUM_RECORDING_TIMESTEPS
)  # TODO: How big to calculate for GPU size?
1
num_steps = 3000000
# var_list = ["V"]
# data = {
#     model_name: {k: np.zeros((v1_pop_counts[model_name], num_steps)) for k in var_list}
#     for model_name in v1_model_names
# }
# view_dict = {}
# for model_name in v1_model_names:
#     pop = pop_dict[model_name]
#     for v in var_list:
#         view_dict[model_name] = {v: pop.vars[v].view}

# Construct data for spike times
spike_data = {}
for model_name in v1_model_names:
    spike_data[model_name] = {}
    num_neurons = v1_pop_counts[model_name]
    for i in range(num_neurons):
        spike_data[model_name][i] = []  # List of spike times for each neuron


for i in range(num_steps):

    model.step_time()

    # for model_name in v1_model_names:
    #     pop = pop_dict[model_name]

    #     v_view = pop.vars["V"].view
    #     for var_name in var_list:
    #         # print(i, model_name, var_name, sep="\t")
    #         pop.pull_var_from_device(var_name)
    #         view = view_dict[model_name][var_name]
    #         output = view[:]
    #         # print(output)
    #         data[model_name][var_name][:, i] = output

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

    # for model_name in v1_model_names:
    #     pop = pop_dict[model_name]

    #     pop.pull_current_spikes_from_device()
    #     spk = pop.current_spikes
    #     if len(spk) > 0:
    #         print(spk)

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


# # Plot firing rates
# fig, axs = plt.subplots(1, 1)
# for BMTK_id, times in spike_data_BMTK_ids.items():
#     num_spikes = len(times)
#     period_length = num_steps / 1e6  # s
#     firing_rate = num_spikes / period_length

#     axs.plot(BMTK_id, firing_rate, "k.")

# axs.set_ylabel("Firing Rate (hz)")
# axs.set_xlabel("node_id")
# plt.show()


# # Plot voltage
# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1, 1)

# # GeNN
# axs.set_ylabel("mV")
# axs.set_xlabel("ms")

# for k in data.keys():

#     V = data[k]["V"]
#     num_neurons, num_steps = V.shape
#     num_neurons = 1
#     t = np.arange(0, num_steps) * sim_config["run"]["dt"]
#     t = np.repeat(t.reshape(1, -1), num_neurons, axis=0)

#     for i in range(num_neurons):
#         axs.plot(t[i, :], V[i, :], label="GeNN")
# plt.show()

print("Simulation complete.")
