# -*- coding: utf-8 -*-
"""
Mini project for the course Numerical Scientific Computing

Loads all the created files and plot the data.

@author: 871
"""

# %% Imports
import matplotlib.pyplot as plt

import numpy as np

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
    with h5py.File(f"data/{directory}/{title}_{res}.h5", 'r') as hf:
        t = hf['time'][()]
        mfractal = hf['mfractal'][:]
        z = hf['z'][:]

    return t, mfractal, z

# %% plot


def _plot(mfractal, lim, directory, title, res):
    """

    Plots the mandlbrot based on values from the different methods

    Parameters
    ----------
    mfractal : matrix
        matrix with values from the mandelbrot methods
    lim : array
        array with limits. [x_min, x_max, y_min, y_max].
    directory : string
        subfolder name inside the data folder. e.g. "naive"
    title : string
        title of files. e.g. "Mandelbrot_Naive"
    res : int
        the resolution

    Returns
    -------
    None.

    """
    x_min, x_max, y_min, y_max = lim

    # Make plot and save figure
    plt.imshow(np.log(mfractal), cmap=plt.cm.hot,
               extent=[x_min, x_max, y_min, y_max])
    plt.title(f"{title}_{res}")
    plt.xlabel('Re[c]')
    plt.ylabel('Im[c]')
    plt.savefig(f"data/{directory}/{title}_{res}.pdf",
                bbox_inches='tight', pad_inches=0.05, dpi=500)
    plt.close()

# %% Main


if __name__ == '__main__':
    # Constants - Resolution
    res = [100, 500, 1000, 2000, 5000]

    # Constants - Limits
    lim = [-2, 1, -1.5, 1.5]  # [x_min, x_max, y_min, y_max]

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

    mfractal = []
    t = []
    for j in range(len(title)):
        print(title[j])
        for i in range(len(res)):
            print(f"res: {res[i]}")

            _, mfractal_output, _ = _load(folder[j], title[j], res[i])
            if i == 0:
                mfractal.append([mfractal_output])
            else:
                mfractal[j].append(mfractal_output)

            # Plot
            _plot(mfractal[j][i], lim, folder[j], title[j], res[i])
