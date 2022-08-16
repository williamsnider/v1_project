import pygenn.genn_model
import numpy as np
from sonata.reports.spike_trains import SpikeTrains
import json
from pathlib import Path

DYNAMICS_BASE_DIR = Path("../450glifs_without_intfire/point_components/cell_models")
SIM_CONFIG_PATH = Path(
    "../450glifs_without_intfire/point_450glifs/config.simulation.json"
)
GLIF3_dynamics_file = Path("478958894_glif_lif_asc_psc.json")

### Define custom classes ###
psc_Alpha = pygenn.genn_model.create_custom_postsynaptic_class(
    class_name="Alpha",
    decay_code="""
    $(x) += exp(-DT/$(tau)) * ((DT * $(inSyn) * exp(1.0f) / $(tau)));
    $(inSyn)*=exp(-DT/$(tau));
    """,
    apply_input_code="$(Isyn) += $(x);",
    var_name_types=[("x", "scalar")],
    param_names=[("tau")],
)
