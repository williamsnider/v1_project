import numpy as np
from utilities import check_nan_arrays_equal, plot_results_and_diff
from allen_refactored_GLIF import (
    GLIF1_refactored,
    GLIF2_refactored,
    GLIF3_refactored,
    GLIF4_refactored,
    GLIF5_refactored,
)
from parameters import GLIF_dict

time_range_for_plotting = [18, 18.3]
specimen_ids = [474637203]  # , 512322162]
model_types = [
    "LIF_model",
]
#     "LIFR_model",
#     "LIFASC_model",
#     "LIFRASC_model",
#     "LIFRASCAT_model",
# ]
time_range = [18, 18.3]

for specimen_id in specimen_ids:
    for model_type in model_types:

        GLIF_name = GLIF_dict[model_type]
        GLIF = eval(GLIF_name + "_refactored")
        saved_model, V, thres = GLIF(specimen_id, model_type)

        t = saved_model["time"]
        mask = np.logical_and(
            t > time_range_for_plotting[0], t < time_range_for_plotting[1]
        )
        t_mask = t[mask]

        # Plot voltages
        Allen = saved_model["voltage"][mask] * 1e3
        python = V[mask].ravel() * 1e3
        result = check_nan_arrays_equal(Allen, python)
        print("Are voltage results equal: {}".format(result))
        plot_results_and_diff(Allen, "Allen", python, "python", t[mask])

        # Plot thresholds
        Allen = saved_model["threshold"][mask] * 1e3
        python = thres * np.ones(Allen.shape) * 1e3
        result = check_nan_arrays_equal(Allen, python)
        print("Are threshold results equal: {}".format(result))
        plot_results_and_diff(Allen, "Allen", python, "python", t[mask])
