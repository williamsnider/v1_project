from allen_simulation import load_model_config_stimulus
import pickle
import matplotlib.pyplot as plt
import numpy as np
from parameters import GLIF_dict

specimen_ids = [474637203]  # , 512322162]
model_types = [
    "LIF_model",
    "LIFR_model",
    "LIFASC_model",
    "LIFRASC_model",
    "LIFRASCAT_model",
]

num_cols = 4
t_range = [18, 18.3]
var_name_dict = {"V": "voltage", "T": "threshold", "ASC": "AScurrents"}
var_scale = {"V": 1e3, "T": 1e3, "ASC": 1e9}
var_unit = {"V": "mV", "T": "mV", "ASC": "nA"}

for specimen_id in specimen_ids:

    fig, axs = plt.subplots(len(model_types), num_cols)

    for row_num, model_type in enumerate(model_types):

        # Load GeNN and Allen data
        save_name = "pkl_data/GeNN_" + str(specimen_id) + "_{}.pkl".format(model_type)
        with open(save_name, "rb") as f:
            GeNN_data_dict, Allen_model = pickle.load(f)

        t = Allen_model["time"]
        mask = np.logical_and(t > 18, t < 18.3)

        V_GeNN = np.squeeze(GeNN_data_dict["V"])
        T_GeNN = np.squeeze(GeNN_data_dict["T"])
        A_GeNN = np.squeeze(GeNN_data_dict["ASC"])

        V_Allen = Allen_model["voltage"] * 1e3
        T_Allen = Allen_model["threshold"] * 1e3
        A_Allen = Allen_model["AScurrents"] * 1e9

        for col_num in range(num_cols):

            # Overlay of Allen and GeNN Voltage
            if col_num == 0:

                axs[row_num, col_num].plot(t[mask], V_GeNN[mask], label="GeNN")
                axs[row_num, col_num].plot(t[mask], V_Allen[mask], label="Allen")
                axs[row_num, col_num].set_ylabel("{}\nmV".format(GLIF_dict[model_type]))
                axs[row_num, col_num].legend()
                if row_num == 0:
                    axs[row_num, col_num].set_title("Voltage Overlay")
                if row_num != len(model_types) - 1:
                    axs[row_num, col_num].set_xticks([])

            # Voltage diff
            if col_num == 1:

                diff = V_GeNN - V_Allen
                axs[row_num, col_num].plot(t[mask], diff[mask], label="diff")
                axs[row_num, col_num].set_ylabel("mV")
                axs[row_num, col_num].legend()
                if row_num == 0:
                    axs[row_num, col_num].set_title("Voltage Difference\nGeNN - Allen")
                if row_num != len(model_types) - 1:
                    axs[row_num, col_num].set_xticks([])

            # Threshold diff
            if col_num == 2:

                diff = T_GeNN - T_Allen
                axs[row_num, col_num].plot(t[mask], diff[mask], label="diff")
                axs[row_num, col_num].set_ylabel("mV")
                axs[row_num, col_num].legend()
                if row_num == 0:
                    axs[row_num, col_num].set_title(
                        "Threshold Difference\nGeNN - Allen"
                    )
                if row_num != len(model_types) - 1:
                    axs[row_num, col_num].set_xticks([])

            # ASC diff
            if col_num == 3:

                diff = A_GeNN - A_Allen
                axs[row_num, col_num].plot(t[mask], diff[mask], label="diff")
                axs[row_num, col_num].set_ylabel("nA")
                axs[row_num, col_num].legend()
                if row_num == 0:
                    axs[row_num, col_num].set_title(
                        "ASCurrents Difference\nGeNN - Allen"
                    )
                if row_num != len(model_types) - 1:
                    axs[row_num, col_num].set_xticks([])

plt.show()
