# Replicates Fig 1b from Teeter 2018
import time
import os
from allen_simulation import make_and_save_model
from pathlib import Path

# Parameters
specimen_ids = [474637203, 512322162]
model_types = [
    "LIF_model",
    "LIFR_model",
    "LIFASC_model",
    "LIFRASC_model",
    "LIFRASCAT_model",
]

# Simulate GLIF neuron using Allen SDK
path = os.path.join("./GLIF_Teeter_et_al_2018", "mouse_struc_data_dir")
start_time = time.time()
saved_models = [f.name for f in Path("./pkl_data").glob("*")]

for specimen_id in specimen_ids:

    for model_type in model_types:

        for model in saved_models:

            # Skip if model already run and saved
            if model.startswith(str(specimen_id)) and model.endswith(
                "_{}.pkl".format(model_type)
            ):
                print("Already saved {}".format(model))
                break

        else:
            make_and_save_model(specimen_id, model_type)
