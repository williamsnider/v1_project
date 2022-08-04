### Adapted from https://github.com/AllenInstitute/GLIF_Teeter_et_al_2018 ###
from pathlib import Path
import numpy as np
import pickle
import os
from allensdk.model.glif.glif_neuron import GlifNeuron
import sys

sys.path.append(str(Path("./GLIF_Teeter_et_al_2018_python3/libraries").resolve()))
import time
from GLIF_Teeter_et_al_2018_python3.libraries.data_library import (
    get_file_path_endswith,
    get_sweep_num_by_name,
)
import json
import matplotlib.pyplot as plt
from pathlib import Path
from parameters import GLIF_dict, folders, relative_path, saved_models


def get_model(path, EW):
    """Runs the model for a specified neuron and model
    inputs:
        path: string
            folder path with files for the neuron
        EW: string
            end of file searching for:  options '_GLIF1_neuron_config.json',_GLIF2_neuron_config.json' etc.
    returns:
        run_data: dictionary
            contains data from the model run

    """
    from allensdk.core.cell_types_cache import CellTypesCache
    import allensdk.core.json_utilities as ju

    print("loading ctc")
    ctc = CellTypesCache(
    manifest_file=os.path.join(relative_path, "cell_types_manifest.json")   )
    specimen_id = int(os.path.basename(path)[:9])
    file = get_file_path_endswith(path, EW)

    # load data
    dir_name = os.path.join(relative_path, "mouse_nwb/specimen_" + str(specimen_id))
    all_sweeps = ctc.get_ephys_sweeps(
        specimen_id, os.path.join(dir_name, "ephys_sweeps.json")
    )
    sweeps = get_sweep_num_by_name(all_sweeps, "Noise 2")

    noise2_sweeps = get_sweep_num_by_name(all_sweeps, "Noise 2")
    noise2_data = ctc.get_ephys_data(
        specimen_id, os.path.join(dir_name, "ephys.nwb")
    ).get_sweep(noise2_sweeps[0])

    # run model with current
    stimulus2 = noise2_data["stimulus"]
    neuron_config = ju.read(file)
    neuron_config["dt"] = (
        1.0 / noise2_data["sampling_rate"]
    )  # reset dt to the stimulus dt not the optimization dt
    neuron = GlifNeuron.from_dict(neuron_config)
    1 / noise2_data["sampling_rate"]
    run_data = neuron.run(stimulus2)
    run_data["time"] = np.arange(0, len(run_data["voltage"])) * neuron_config["dt"]
    run_data["El_reference"] = neuron_config["El_reference"]
    run_data["stimulus"] = noise2_data["stimulus"]
    run_data["tcs"] = neuron.threshold_components

    return run_data


def make_and_save_model(specimen_id, model_type):
    """Runs models and creates resulting voltage waveforms and saves them to a pickle file
    inputs:
        specimen_id: integer
            specifies neuron to be run
        model_type: string
            specifies which type of GLIF model
    outputs:
        pickle files
    """

    start_time = time.time()  # grab start_time from outside this module

    # finding the folder associated with the desired specimen_id
    for dir in folders:
        sp_id = int(os.path.basename(dir)[:9])
        if sp_id == specimen_id:
            folder = dir
    cre = os.path.basename(folder)[10:]

    try:
        os.makedirs("pkl_data")
    except:
        pass

    print("running {}".format(model_type))
    config_name = "_{}_neuron_config.json".format(GLIF_dict[model_type])
    LIF_model = get_model(folder, config_name)
    save_name = "pkl_data/" + str(specimen_id) + cre + "_{}.pkl".format(model_type)
    with open(save_name, "wb") as f:
        pickle.dump(LIF_model, f)
    print(
        "{} done at".format(GLIF_dict[model_type]),
        (time.time() - start_time) / 60.0,
        "min",
    )



