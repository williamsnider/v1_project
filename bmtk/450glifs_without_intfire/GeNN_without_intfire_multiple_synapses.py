# Replicate point_450glifs example with GeNN
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sonata.circuit import File
from sonata.reports.spike_trains import SpikeTrains
import pygenn
import matplotlib.pyplot as plt


DYNAMICS_BASE_DIR = Path("./point_components/cell_models")
SIM_CONFIG_PATH = Path("point_450glifs/config.simulation.json")
LGN_V1_EDGE_CSV = Path("./point_450glifs/network/lgn_v1_edge_types.csv")
V1_EDGE_CSV = Path("./point_450glifs/network/v1_v1_edge_types.csv")
LGN_SPIKES_PATH = Path("./point_450glifs/inputs/lgn_spikes.h5")
LGN_NODE_DIR = Path("./point_450glifs/network/lgn_node_types.csv")
V1_NODE_CSV = Path("./point_450glifs/network/v1_node_types.csv")

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

psc_Alpha = pygenn.genn_model.create_custom_postsynaptic_class(
    class_name="Alpha",
    decay_code="""
    $(x) = exp(-DT/$(tau)) * ((DT * $(inSyn) * exp(1.0f) / $(tau)) + $(x));
    $(inSyn)*=exp(-DT/$(tau));
    """,
    apply_input_code="""
    $(Isyn) += $(x);     
""",
    var_name_types=[("x", "scalar")],
    param_names=[("tau")],
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
v1_df = pd.read_csv(V1_NODE_CSV, sep=" ")

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
model = pygenn.genn_model.GeNNModel()
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
        "C": dynamics_params["C_m"] / 1000,  # pF -> nF
        "G": dynamics_params["g"] / 1000,  # nS -> uS
        "El": dynamics_params["E_L"],
        "th_inf": dynamics_params["V_th"],
        "dT": sim_config["run"]["dt"],
        "V": dynamics_params["V_m"],
        "spike_cut_length": round(dynamics_params["t_ref"] / sim_config["run"]["dt"]),
        "refractory_countdown": -1,
        "k": 1
        / np.repeat(
            dynamics_params["asc_decay"], num_neurons
        ).ravel(),  # TODO: Reciprocal?
        "r": np.repeat([1.0, 1.0], num_neurons).ravel(),
        "ASC": np.repeat(dynamics_params["asc_init"], num_neurons).ravel()
        / 1000,  # pA -> nA
        "asc_amp_array": np.repeat(dynamics_params["asc_amps"], num_neurons).ravel()
        / 1000,  # pA -> nA
        "ASC_length": 2,
    }

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

    # Assign extra global parameter values
    for k in pop_dict[v1_model_name].extra_global_params.keys():
        pop_dict[v1_model_name].set_extra_global_param(k, dynamics_params_renamed[k])

    print("{} population added to model.".format(v1_model_name))

# Enable spike recording
for k in pop_dict.keys():
    pop_dict[k].spike_recording_enabled = True

# Add LGN Nodes
lgn_df = pd.read_csv(LGN_NODE_DIR, sep=" ")
lgn_nodes = lgn_net.nodes["lgn"]
lgn_model_names = lgn_df["model_type"].to_list()
lgn_node_dict = {}
for i, lgn_model_name in enumerate(lgn_model_names):
    nodes_with_model_name = [
        n["node_id"] for n in lgn_nodes.filter(model_type=lgn_model_name)
    ]
    lgn_node_dict[lgn_model_names[i]] = nodes_with_model_name


# Get LGN spiking times
spikes = SpikeTrains.from_sonata(LGN_SPIKES_PATH)
spikes_df = spikes.to_dataframe()

lgn_spiking_nodes = spikes_df["node_ids"].unique().tolist()
spikes_list = []
for n in lgn_spiking_nodes:
    spikes_list.append(spikes_df[spikes_df["node_ids"] == n]["timestamps"].to_list())

start_spike, end_spike, spike_times = spikes_list_to_start_end_times(spikes_list)

# Construct LGN population
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

    # Add edge type id, which is used to get correct synaptic weight/delay
    # TODO: Is dynamics_params e.g. e2i.json used?
    e_dict["edge_type_id"] = e.edge_type_id

    # Add number of synapses
    e_dict["nsyns"] = e["nsyns"]

    v1_edges_for_df.append(e_dict)
v1_edge_df = pd.DataFrame(v1_edges_for_df)

syn_dict = {}
v1_syn_df = pd.read_csv(V1_EDGE_CSV, sep=" ")
v1_edge_type_ids = v1_syn_df["edge_type_id"].tolist()
v1_nsyn_range = v1_edge_df["nsyns"].unique()
v1_nsyn_range.sort()

for pop1 in v1_model_names:
    for pop2 in v1_model_names:
        for edge_type_id in v1_edge_type_ids:
            for nsyn in v1_nsyn_range:

                # Filter by source, target, edge_type_id, and nsyns
                src = v1_edge_df[~v1_edge_df["source_" + pop1].isnull()]
                src_tgt = src[~src["target_" + pop2].isnull()]
                src_tgt_id = src_tgt[src_tgt["edge_type_id"] == edge_type_id]
                src_tgt_id_nsyns = src_tgt_id[src_tgt_id["nsyns"] == nsyn]

                # Convert to list for GeNN
                s_list = src_tgt_id_nsyns["source_" + pop1].tolist()
                t_list = src_tgt_id_nsyns["target_" + pop2].tolist()

                # Skip if no synapses (typically only 1 edge_type_id relevant for each source)
                if len(s_list) == 0:
                    # print(
                    #     "No synapses found for {} -> {} with edge type id={}".format(
                    #         pop1,
                    #         pop2,
                    #         edge_type_id,
                    #     )
                    # )
                    continue

                # Test correct assignment
                assert np.all(~src_tgt_id_nsyns.isnull(), axis=0).sum() == 6

                # Get delay and weight specific to the edge_type_id
                delay_steps = round(
                    v1_syn_df[v1_syn_df["edge_type_id"] == edge_type_id]["delay"].iloc[
                        0
                    ]
                    / sim_config["run"]["dt"]
                )  # delay (ms) -> delay (steps)
                weight = (
                    v1_syn_df[v1_syn_df["edge_type_id"] == edge_type_id][
                        "syn_weight"
                    ].iloc[0]
                    / 1e3
                    * nsyn
                )  # nS -> uS; Multiple by number of synapses

                s_ini = {"g": weight}
                psc_Alpha_params = {
                    "tau": dynamics_params["tau_syn"][0]
                }  # TODO: Always 0th port?
                psc_Alpha_init = {"x": 0.0}

                synapse_group_name = pop1 + "_to_" + pop2 + "_nsyns_" + str(nsyn)
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
                syn_dict[synapse_group_name].set_sparse_connections(
                    np.array(s_list), np.array(t_list)
                )
                print(
                    "Synapses added for {} -> {} with edge type id={} and nsyns={}".format(
                        pop1, pop2, edge_type_id, nsyn
                    )
                )


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

    # Add edge type id, which is used to get correct synaptic weight/delay
    # TODO: Is dynamics_params e.g. e2i.json used?
    e_dict["edge_type_id"] = e.edge_type_id

    # Add number of synapses
    e_dict["nsyns"] = e["nsyns"]

    lgn_edges_for_df.append(e_dict)
lgn_edge_df = pd.DataFrame(lgn_edges_for_df)

lgn_syn_df = pd.read_csv(LGN_V1_EDGE_CSV, sep=" ")
lgn_edge_type_ids = lgn_syn_df["edge_type_id"].tolist()
lgn_nsyn_range = lgn_edge_df["nsyns"].unique()
lgn_nsyn_range.sort()
for pop1 in lgn_model_names:
    for pop2 in v1_model_names:
        for edge_type_id in lgn_edge_type_ids:
            for nsyn in lgn_nsyn_range:

                # Filter by source, target, and edge_type_id
                src = lgn_edge_df[~lgn_edge_df["source_" + pop1].isnull()]
                src_tgt = src[~src["target_" + pop2].isnull()]
                src_tgt_id = src_tgt[src_tgt["edge_type_id"] == edge_type_id]
                src_tgt_id_nsyns = src_tgt_id[src_tgt_id["nsyns"] == nsyn]

                # Convert to list for GeNN
                s_list = src_tgt_id_nsyns["source_" + pop1].tolist()
                t_list = src_tgt_id_nsyns["target_" + pop2].tolist()

                # Skip if no synapses (typically only 1 edge_type_id relevant for each source)
                if len(s_list) == 0:
                    # print(
                    #     "No synapses found for {} -> {} with edge type id={}".format(
                    #         pop1,
                    #         pop2,
                    #         edge_type_id,
                    #     )
                    # )
                    continue

                # Test correct assignment
                assert np.all(~src_tgt_id_nsyns.isnull(), axis=0).sum() == 6

                # Get delay and weight specific to the edge_type_id
                delay_steps = round(
                    lgn_syn_df[lgn_syn_df["edge_type_id"] == edge_type_id][
                        "delay"
                    ].iloc[0]
                    / sim_config["run"]["dt"]
                )  # delay (ms) -> delay (steps)
                weight = (
                    lgn_syn_df[lgn_syn_df["edge_type_id"] == edge_type_id][
                        "syn_weight"
                    ].iloc[0]
                    / 1e3
                    * nsyn
                )  # nS -> uS

                s_ini = {"g": weight}
                psc_Alpha_params = {
                    "tau": dynamics_params["tau_syn"][0]
                }  # TODO: Always 0th port?
                psc_Alpha_init = {"x": 0.0}

                synapse_group_name = pop1 + "_to_" + pop2 + "_nsyns_" + str(nsyn)
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
                syn_dict[synapse_group_name].set_sparse_connections(
                    np.array(s_list), np.array(t_list)
                )
                print(
                    "Synapses added for {} -> {} with edge type id={} and nsyns={}".format(
                        pop1, pop2, edge_type_id, nsyn
                    )
                )

NUM_RECORDING_TIMESTEPS = 1000
model.build(force_rebuild=True)
model.load(
    num_recording_timesteps=NUM_RECORDING_TIMESTEPS
)  # TODO: How big to calculate for GPU size?

num_steps = 3000000
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


# Plot firing rates
fig, axs = plt.subplots(1, 1)
for BMTK_id, times in spike_data_BMTK_ids.items():
    num_spikes = len(times)
    period_length = num_steps / 1e6  # s
    firing_rate = num_spikes / period_length

    axs.plot(BMTK_id, firing_rate, "k.")

axs.set_ylabel("Firing Rate (hz)")
axs.set_xlabel("node_id")
plt.show()


# Plot voltage
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1)

# GeNN
axs.set_ylabel("mV")
axs.set_xlabel("ms")

for k in data.keys():

    V = data[k]["V"]
    num_neurons, num_steps = V.shape
    num_neurons = 1
    t = np.arange(0, num_steps) * sim_config["run"]["dt"]
    t = np.repeat(t.reshape(1, -1), num_neurons, axis=0)

    for i in range(num_neurons):
        axs.plot(t[i, :], V[i, :], label="GeNN")
plt.show()

print("Simulation complete.")
