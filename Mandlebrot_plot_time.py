# -*- coding: utf-8 -*-
"""
Mini project for the course Numerical Scientific Computing

Loads all the created files and plot the time plot

@author: 871
"""

# %% Imports
import matplotlib.pyplot as plt

import numpy as np

from os import path, makedirs

# %% load


def _load(directory, title, res):
    """
    Parameters
    ----------
    directory : string
        subfolder name inside the data folder. e.g. "naive"
    title : string
        title of files. e.g. "Mandlebrot_Naive"
    res : int
        the resolution

    Returns
    -------
    t : float
        time it took to run the method
    mfractal : int
        the values from the method

    """
    f = open(f"data/{directory}/{title}_{res}.npy", "rb")
    t = np.load(f)
    mfractal = np.load(f)
    f.close()

    return t, mfractal

# %% plot


def _plot_time(t, res):
    """
    Plot three different plots based on the time values and the res.

    Parameters
    ----------
    t : float
        time it took to run the method
    res : int
        Resolutions.

    Returns
    -------
    None.

    """
    if not path.isdir("data/time"):
        makedirs("data/time")

    # Colors
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # Legends
    legends = ['naive', 'numba', 'numpy', 'multiprocessing',
               'dask', 'GPU', 'cython_naive', 'cython_vector']

    # Styles
    marker = '.'
    linestyle = '-'

    # Grid styles
    grid_linewidth = 0.5
    grid_alpha = 0.3

    # Plot all
    for i in range(len(t)):
        plt.plot(res, t[i][:], color=colors[i], label=legends[i],
                 marker=marker, linestyle=linestyle)

    plt.title(f"{title}_{res}")
    plt.xlabel('resolution')
    plt.ylabel('time [s]')
    plt.legend()
    plt.title("")
    plt.grid(linewidth=grid_linewidth, alpha=grid_alpha)
    plt.savefig("data/time/all.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Plot the fast ones
    for i in range(len(t)):
        if np.max(t[i][:]) < 400:
            plt.plot(res, t[i][:], color=colors[i], label=legends[i],
                     marker=marker, linestyle=linestyle)

    plt.title(f"{title}_{res}")
    plt.xlabel('resolution')
    plt.ylabel('time [s]')
    plt.legend()
    plt.title("")
    plt.grid(linewidth=grid_linewidth, alpha=grid_alpha)
    plt.savefig("data/time/fast.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Plot log
    for i in range(len(t)):
        plt.plot(res, t[i][:], color=colors[i], label=legends[i],
                 marker=marker, linestyle=linestyle)

    plt.title(f"{title}_{res}")
    plt.xlabel('resolution')
    plt.ylabel('time [s]')
    plt.legend(loc=('upper left'))
    plt.title("")
    plt.yscale("log")
    plt.grid(linewidth=grid_linewidth, alpha=grid_alpha)
    plt.savefig("data/time/all_log.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()


# %% Main
if __name__ == '__main__':
    # Number of processes
    p = 8

    # Constants - Limits
    lim = [-2, 1, -1.5, 1.5]  # [x_min, x_max, y_min, y_max]

    # Constants - Resolution
    res = [100, 500, 1000, 2000, 5000]

    # Constants - Threshold
    T = 2

    # Constants - Number of Iterations
    iterations = 100

    # Load
    title = ["Mandlebrot_Naive", "Mandlebrot_Numba",
             "Mandlebrot_Numpy", "Mandlebrot_Multiprocessing",
             "Mandlebrot_Dask", "Mandlebrot_GPU",
             "Mandlebrot_Cython_naive", "Mandlebrot_Cython_vector"]
    folder = ["naive", "numba", "numpy", "multiprocessing",
              "dask", "GPU", "cython_naive", "cython_vector"]
    t = []
    for j in range(len(title)):
        print(title[j])
        for i in range(len(res)):
            print(i)
            # assign res
            p_re, p_im = [res[i], res[i]]

            t_output, _ = _load(folder[j], title[j], res[i])
            if i == 0:
                t.append([t_output])
            else:
                t[j].append(t_output)

    _plot_time(t, res)
