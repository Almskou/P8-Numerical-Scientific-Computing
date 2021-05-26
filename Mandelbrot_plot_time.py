# -*- coding: utf-8 -*-
"""
Mini project for the course Numerical Scientific Computing

Loads all the created files and plot the time plot

@author: 871
"""

# %% Imports
from os import path, makedirs

import matplotlib.pyplot as plt

import h5py

# %% load


def _load(directory, title, res):
    """
    Parameters
    ----------
    directory : string
        subfolder name inside the data folder. e.g. "naive"
    title : string
        title of files. e.g. "Mandelbrot_Naive"
    res : int
        the resolution

    Returns
    -------
    t : float
        time it took to run the method
    mfractal : int
        the values from the method

    """
    with h5py.File(f"data/{directory}/{title}_{res}.hdf5", 'r') as hf:
        t = hf['time'][()]
        mfractal = hf['mfractal'][:]
        z = hf['z'][:]

    return t, mfractal, z

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
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
              'limegreen', 'goldenrod', 'k', 'b', 'fuchsia', 'yellow']
    # Legends
    legends = ['naive', 'numba', 'numpy', 'MP_1', 'MP_2', 'MP_4', 'MP_8',
               'MP_16', 'dask_1', 'dask_2', 'dask_4', 'dask_8', 'dask_16',
               'GPU', 'cython_naive', 'cython_vector']

    # Index for different sets
    MP = [3, 4, 5, 6, 7]
    DA = [8, 9, 10, 11, 12]
    AL = [0, 1, 2, 6, 11, 13, 14, 15]
    FA = [1, 2, 6, 11, 13, 15]

    # Styles
    marker = '.'
    linestyle = '-'

    # Grid styles
    grid_linewidth = 0.5
    grid_alpha = 0.3

    # Plot all
    for i in AL:
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

    # Plot multiprocessing
    for i in MP:
        plt.plot(res, t[i][:], color=colors[i], label=legends[i],
                 marker=marker, linestyle=linestyle)

    plt.title(f"{title}_{res}")
    plt.xlabel('resolution')
    plt.ylabel('time [s]')
    plt.legend()
    plt.title("")
    plt.grid(linewidth=grid_linewidth, alpha=grid_alpha)
    plt.savefig("data/time/MP.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Plot dask
    for i in DA:
        plt.plot(res, t[i][:], color=colors[i], label=legends[i],
                 marker=marker, linestyle=linestyle)

    plt.title(f"{title}_{res}")
    plt.xlabel('resolution')
    plt.ylabel('time [s]')
    plt.legend()
    plt.title("")
    plt.grid(linewidth=grid_linewidth, alpha=grid_alpha)
    plt.savefig("data/time/DA.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()
    # Plot the fast ones
    for i in FA:
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
    for i in AL:
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
    # Constants - Resolution
    res = [100, 500, 1000, 2000]

    # Load
    title = ["Mandelbrot_Naive", "Mandelbrot_Numba",
             "Mandelbrot_Numpy", "Mandelbrot_Multiprocessing_1",
             "Mandelbrot_Multiprocessing_2", "Mandelbrot_Multiprocessing_4",
             "Mandelbrot_Multiprocessing_8", "Mandelbrot_Multiprocessing_16",
             "Mandelbrot_Dask_1", "Mandelbrot_Dask_2", "Mandelbrot_Dask_4",
             "Mandelbrot_Dask_8", "Mandelbrot_Dask_16", "Mandelbrot_GPU",
             "Mandelbrot_Cython_naive", "Mandelbrot_Cython_vector"]

    folder = ["naive", "numba", "numpy", "multiprocessing_1",
              "multiprocessing_2", "multiprocessing_4", "multiprocessing_8",
              "multiprocessing_16", "dask_1", "dask_2", "dask_4",
              "dask_8", "dask_16", "GPU", "cython_naive", "cython_vector"]
    t = []
    for j in range(len(title)):
        print(title[j])
        for i in range(len(res)):
            print(f"res: {res[i]}")

            t_output, _, _ = _load(folder[j], title[j], res[i])
            if i == 0:
                t.append([t_output])
            else:
                t[j].append(t_output)

   # _plot_time(t, res)
