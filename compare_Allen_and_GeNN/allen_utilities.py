import numpy as np
from parameters import saved_models, folders, GLIF_dict
import pickle
import json
import os 
from pathlib import Path

def load_model_config_stimulus(specimen_id, model_type):
    "Loads the saved Allen model for a specimen_id and model_type. Returns the model, config, and stimulus."

    # Find saved Allen model
    for model in saved_models:
        if model.startswith(str(specimen_id)) and model.endswith(
            "_{}.pkl".format(model_type)
        ):
            break
    else:
        raise ValueError(
            "Allen run data not found for specimen: {0} and model type: {1}".format(
                specimen_id, model_type
            )
        )

    # Load
    filename = Path("pkl_data", model)
    with open(filename, "rb") as f:
        saved_model = pickle.load(f)

    # Load config
    for dir in folders:
        sp_id = int(os.path.basename(dir)[:9])
        if sp_id == specimen_id:
            folder = dir
    cre = os.path.basename(folder)[10:]
    filename = Path(
        folder,
        Path(folder).parts[-1] + "_{}_neuron_config.json".format(GLIF_dict[model_type]),
    )
    with open(filename) as f:
        config = json.load(f)

    # Get stimulus dt, as config file's dt is optimization dt
    # dir_name=os.path.join(relative_path, 'mouse_nwb/specimen_'+ str(specimen_id))
    # all_sweeps=ctc.get_ephys_sweeps(specimen_id,  os.path.join(dir_name, 'ephys_sweeps.json'))
    # sweeps=get_sweep_num_by_name(all_sweeps, 'Noise 2')
    # noise2_sweeps = get_sweep_num_by_name(all_sweeps, 'Noise 2')
    # noise2_data=ctc.get_ephys_data(specimen_id, os.path.join(dir_name, 'ephys.nwb')).get_sweep(noise2_sweeps[0])
    # stimulus=noise2_data['stimulus']
    # config['dt']=1./noise2_data['sampling_rate']
    stimulus = saved_model["stimulus"]
    config["dt"] = np.diff(saved_model["time"][:2])[0]

    return saved_model, config, stimulus