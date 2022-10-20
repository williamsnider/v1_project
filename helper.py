import pandas as pd
import numpy as np
import pygenn
import json
from pathlib import Path
import pickle

GLIF3 = pygenn.genn_model.create_custom_neuron_class(
    "GLIF3",
    param_names=[
        "C",
        "G",
        "El",
        "spike_cut_length",
        "th_inf",
        "V_reset",
        "asc_amp_array_1",
        "asc_amp_array_2",
        "asc_stable_coeff_1",
        "asc_stable_coeff_2",
        "asc_decay_rates_1",
        "asc_decay_rates_2",
        "asc_refractory_decay_rates_1",
        "asc_refractory_decay_rates_2",
    ],
    var_name_types=[
        ("V", "scalar"),
        ("refractory_countdown", "int"),
        ("ASC_1", "scalar"),
        ("ASC_2", "scalar"),
    ],
    sim_code="""

    // Sum after spike currents
    double sum_of_ASC = $(ASC_1)*$(asc_stable_coeff_1) + $(ASC_2)*$(asc_stable_coeff_2);

    // Voltage
    if ($(refractory_countdown) <= 0) {
        $(V)+=1/$(C)*($(Isyn)+sum_of_ASC-$(G)*($(V)-$(El)))*DT;
    }

    // ASCurrents
    if ($(refractory_countdown) <= 0) {
        $(ASC_1) *= $(asc_decay_rates_1);
        $(ASC_2) *= $(asc_decay_rates_2);
        }


    // Decrement refractory_countdown by 1; Do not decrement past -1
    if ($(refractory_countdown) > -1) {
        $(refractory_countdown) -= 1;
    }
    """,
    threshold_condition_code="$(V) > $(th_inf)",
    reset_code="""
    $(V)=$(V_reset);
    $(ASC_1) = $(asc_amp_array_1) + $(ASC_1) * $(asc_refractory_decay_rates_1);
    $(ASC_2) = $(asc_amp_array_2) + $(ASC_2) * $(asc_refractory_decay_rates_2);
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


def optimize_nodes_df_memory(nodes_df):

    columns_to_drop = [
        "tuning_angle",
        "x",
        "y",
        "z",
        "model_type",
        "model_template",
        "ei",
        "location",
        "population",
        "gaba_synapse",
    ]

    columns_category = [
        "dynamics_params",
        "pop_name",
        "node_type_id",
        "model_name",
    ]

    columns_float = [
        "C",
        "G",
        "El",
        "th_inf",
        "dT",
        "V",
        "V_reset",
        "ASC_1",
        "ASC_2",
        "asc_stable_coeff",
        "asc_decay_rates",
        "asc_refractory_decay_rates",
        "asc_amp_array_1",
        "asc_asc_amp_array_2",
        "asc_stable_coeff_1",
        "asc_stable_coeff_2",
        "asc_decay_rates_1",
        "asc_decay_rates_2",
        "asc_refractory_decay_rates_1",
        "asc_refractory_decay_rates_2",
        "tau",
        "spike_cut_length",  # Needs to be float because it's a parameter
    ]

    columns_int = [
        "node_id",
        "refractory_countdown",
        "GeNN_node_id",
    ]

    # Drop non-needed columns
    for col in columns_to_drop:

        if col not in nodes_df.columns:
            continue
        else:
            nodes_df.drop(columns=[col], inplace=True)

    # Convert to category
    for col in columns_category:

        if col not in nodes_df.columns:
            continue
        else:
            nodes_df[col] = nodes_df[col].astype("category")

    # Downcast float
    for col in columns_float:

        if col not in nodes_df.columns:
            continue
        else:
            nodes_df[col] = pd.to_numeric(nodes_df[col], downcast="float")

    # Downcast int
    for col in columns_int:

        if col not in nodes_df.columns:
            continue
        else:
            nodes_df[col] = pd.to_numeric(nodes_df[col], downcast="signed")

    return nodes_df


def optimize_edges_df_memory(edge_df):

    columns_to_drop = [
        "target_query",
        "source_query",
        "weight_function",
        "weight_sigma",
        "dynamics_params",
        "model_template",
    ]

    columns_category = ["source_model_name", "target_model_name"]

    columns_float = ["delay", "syn_weight"]

    columns_int = [
        "edge_type_id",
        "source_node_id",
        "target_node_id",
        "source_GeNN_id",
        "target_GeNN_id",
        "nsyns",
    ]

    # Drop non-needed columns
    for col in columns_to_drop:

        if col not in edge_df.columns:
            continue
        else:
            edge_df.drop(columns=[col], inplace=True)

    # Convert to category
    for col in columns_category:

        if col not in edge_df.columns:
            continue
        else:
            edge_df[col] = edge_df[col].astype("category")

    # Downcast float
    for col in columns_float:

        if col not in edge_df.columns:
            continue
        else:
            edge_df[col] = pd.to_numeric(edge_df[col], downcast="float")

    # Downcast int
    for col in columns_int:

        if col not in edge_df.columns:
            continue
        else:
            edge_df[col] = pd.to_numeric(edge_df[col], downcast="unsigned")

    return edge_df


def get_dynamics_params(dynamics_path, DT):
    with open(dynamics_path) as f:
        old_dynamics_params = json.load(f)

    asc_decay = np.array(old_dynamics_params["k"])
    r = np.array([1.0, 1.0])  # NEST default
    t_ref = old_dynamics_params["t_ref"]
    asc_decay_rates = np.exp(-asc_decay * DT)
    asc_stable_coeff = (1.0 / asc_decay / DT) * (1.0 - asc_decay_rates)
    asc_refractory_decay_rates = r * np.exp(-asc_decay * t_ref)

    dynamics_params_renamed = {
        "C": old_dynamics_params["C_m"] / 1000,  # pF -> nF
        "G": old_dynamics_params["g"] / 1000,  # nS -> uS
        "El": old_dynamics_params["E_L"],
        "th_inf": old_dynamics_params["V_th"],
        "dT": DT,
        "V": old_dynamics_params["V_reset"],
        "spike_cut_length": round(old_dynamics_params["t_ref"] / DT),
        "refractory_countdown": -1,
        "V_reset": old_dynamics_params["V_reset"],  # BMTK rounds to 3rd decimal
        "ASC_1": old_dynamics_params["asc_init"][0] / 1000,  # pA -> nA
        "ASC_2": old_dynamics_params["asc_init"][1] / 1000,  # pA -> nA
        "asc_stable_coeff": asc_stable_coeff,
        "asc_decay_rates": asc_decay_rates,
        "asc_refractory_decay_rates": asc_refractory_decay_rates,
        "asc_amp_array_1": old_dynamics_params["asc_amps"][0] / 1000,  # pA->nA
        "asc_amp_array_2": old_dynamics_params["asc_amps"][1] / 1000,  # pA->nA
        "asc_stable_coeff_1": asc_stable_coeff[0],
        "asc_stable_coeff_2": asc_stable_coeff[1],
        "asc_decay_rates_1": asc_decay_rates[0],
        "asc_decay_rates_2": asc_decay_rates[1],
        "asc_refractory_decay_rates_1": asc_refractory_decay_rates[0],
        "asc_refractory_decay_rates_2": asc_refractory_decay_rates[1],
        "tau": old_dynamics_params["tau_syn"][0],
    }

    return dynamics_params_renamed


def save_df(df, path):

    # Save as pickle
    if path.parent.exists() == False:
        Path.mkdir(path.parent, parents=True)

    with open(path, "wb") as f:
        pickle.dump(df, f)


def load_df(path):

    with open(path, "rb") as f:
        df = pickle.load(f)
    return df
