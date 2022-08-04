import os
# from allensdk.core.cell_types_cache import CellTypesCache
from pathlib import Path
import sys

GLIF_dict = {
    "LIF_model": "GLIF1",
    "LIFR_model": "GLIF2",
    "LIFASC_model": "GLIF3",
    "LIFRASC_model": "GLIF4",
    "LIFRASCAT_model": "GLIF5",
}

path = os.path.join("./GLIF_Teeter_et_al_2018_python3", "mouse_struc_data_dir")
folders = [os.path.join(path, f) for f in os.listdir(path)]
relative_path = os.path.dirname(os.getcwd())
# ctc = CellTypesCache(
#     manifest_file=os.path.join(relative_path, "cell_types_manifest.json")
# )
saved_models = [f.name for f in Path("./pkl_data").glob("*")]

sys.path.append(str(Path("./GLIF_Teeter_et_al_2018_python3/libraries").resolve()))


# Dictionaries for the different GLIF models - here as strings so that they can be imported flexibly
