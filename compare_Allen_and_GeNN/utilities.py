import matplotlib.pyplot as plt
import numpy as np


def plot_results_and_diff(A, A_name, B, B_name, t):

    fig, axs = plt.subplots(2, 1)

    # Both Plots
    axs[0].plot(t, A, label=A_name)
    axs[0].plot(t, B, label=B_name)
    axs[0].set_ylabel("mV")
    axs[0].set_title("{0} and {1}".format(A_name, B_name))
    axs[0].legend()

    # Diff
    diff = A - B
    axs[1].plot(t, diff, label="diff")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("{0} - {1} (mV)".format(A_name, B_name))
    axs[1].set_title("Difference")
    axs[1].legend()
    plt.show()


def check_nan_arrays_equal(A, B):
    "Checks if two arrays (containing some np.nan value) are equal."
    return ((A == B) | (np.isnan(A) & np.isnan(B))).all()


def count_unequal_ignoring_nans(A, B):
    "Counts the number of unequal elements (ignoring nans)."
    return (~((A == B) | (np.isnan(A) & np.isnan(B)))).sum()
