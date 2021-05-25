# -*- coding: utf-8 -*-
"""
Mini project for the course Numerical Scientific Computing

Run all the different mandelbrots method and saves the values in a data folder

@author: 871
"""

# %% Imports
from timeit import default_timer as timer
import matplotlib.pyplot as plt

import numpy as np

import Mandelbrot_functions as mb

from os import path, makedirs

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
                bbox_inches='tight', pad_inches=0.05, dpi=300)
    #plt.show()
    plt.close()

# %% save


def _save(mfractal, t, directory, title, res):
    """

    Save the value into a numpy file (.npy)

    Parameters
    ----------
    mfractal : matrix
        matrix with values from the mandelbrot methods
    t : float
        time it took to run the method
    directory : string
        subfolder name inside the data folder. e.g. "naive"
    title : string
        title of files. e.g. "Mandelbrot_Naive"
    res : int
        the resolution.

    Returns
    -------
    None.

    """
    f = open(f"data/{directory}/{title}_{res}.npy", "wb")
    np.save(f, t)
    np.save(f, mfractal)
    f.close()

# %% run


def _run(directory, title, res):
    """
    Checks is the data have already been created.
    If not create a path where the new data can be saved

    Parameters
    ----------
    directory : string
        subfolder name inside the data folder. e.g. "naive"
    title : string
        title of files. e.g. "Mandelbrot_Naive"
    res : int
        the resolution.

    Returns
    -------
    bool
        TRUE if no data is created.

    """
    if not path.isdir(f"data/{directory}"):
        makedirs(f"data/{directory}")

    if path.isfile(f"data/{directory}/{title}_{res}.npy"):
        return False
    else:
        return True

# %% Main


if __name__ == '__main__':
    # Number of processes
    p = 8

    # Constants - Limits
    lim = [-2, 1, -1.5, 1.5]  # [x_min, x_max, y_min, y_max]

    # Constants - Resolution
    resolutions = [100, 500, 1000, 2000, 5000]

    # Constants - Threshold
    T = 2

    # Constants - Number of Iterations
    iterations = 100

    for res in resolutions:

        # assign res
        p_re, p_im = [res, res]

        print(f"res: {res}")

        # ---- Naive ----
        title = "Mandelbrot_Naive"
        folder = "naive"
        if _run(folder, title, res):
            # Start timer
            t_start = timer()

            # calculate the fractals
            mfractal = mb.naive(lim=lim, res_re=p_re, res_im=p_im,
                                threshold=T, iterations=iterations)

            # Stop timer
            t_stop = timer()
            t_total = t_stop - t_start

            # save data
            _save(mfractal, t_total, folder, title, res)

            # Plot
            _plot(mfractal, lim, folder, title, res)

            print(f"Mandelbrot Naive took {t_total}s")

        # ---- Numba ----
        title = "Mandelbrot_Numba"
        folder = "numba"
        if _run(folder, title, res):
            # Start timer
            t_start = timer()

            # calculate the fractals
            mfractal = mb.numba(lim=lim, res_re=p_re, res_im=p_im,
                                threshold=T, iterations=iterations)

            # Stop timer
            t_stop = timer()
            t_total = t_stop - t_start

            # save data
            _save(mfractal, t_total, folder, title, res)

            # Plot
            _plot(mfractal, lim, folder, title, res)

            print(f"Mandelbrot Numba took {t_total}s")

        # ---- Numpy ----
        title = "Mandelbrot_Numpy"
        folder = "numpy"
        if _run(folder, title, res):
            # Start timer
            t_start = timer()

            # calculate the fractals
            mfractal = mb.numpy(lim=lim, res_re=p_re, res_im=p_im,
                                threshold=T, iterations=iterations)

            # Stop timer
            t_stop = timer()
            t_total = t_stop - t_start

            # save data
            _save(mfractal, t_total, folder, title, res)

            # Plot
            _plot(mfractal, lim, folder, title, res)

            print(f"Mandelbrot Numpy took {t_total}s")

        # ---- Multiprocessing ----
        title = "Mandelbrot_Multiprocessing"
        folder = "multiprocessing"
        if _run(folder, title, res):
            # Start timer
            t_start = timer()

            # calculate the fractals
            mfractal = mb.multiprocessing(lim=lim, res_re=p_re, res_im=p_im,
                                          threshold=T, iterations=iterations,
                                          p=p)

            # Stop timer
            t_stop = timer()
            t_total = t_stop - t_start

            # save data
            _save(mfractal, t_total, folder, title, res)

            # Plot
            _plot(mfractal, lim, folder, title, res)

            print(f"Mandelbrot Multiprocessing took {t_total}s")

        # ---- Dask ----
        title = "Mandelbrot_Dask"
        folder = "dask"
        if _run(folder, title, res):
            # Start timer
            t_start = timer()

            # calculate the fractals
            mfractal = mb.dask(lim=lim, res_re=p_re, res_im=p_im,
                               threshold=T, iterations=iterations, p=p)

            # Stop timer
            t_stop = timer()
            t_total = t_stop - t_start

            # save data
            _save(mfractal, t_total, folder, title, res)

            # Plot
            _plot(mfractal, lim, folder, title, res)

            print(f"Mandelbrot Dask took {t_total}s")

        # ---- GPU ----
        title = "Mandelbrot_GPU"
        folder = "GPU"
        if _run(folder, title, res):
            # Start timer
            t_start = timer()

            # calculate the fractals
            mfractal = mb.GPU(lim=lim, res_re=p_re, res_im=p_im,
                              threshold=T, iterations=iterations)

            # Stop timer
            t_stop = timer()
            t_total = t_stop - t_start

            # save data
            _save(mfractal, t_total, folder, title, res)

            # Plot
            _plot(mfractal, lim, folder, title, res)

            print(f"Mandelbrot GPU took {t_total}s")

        # ---- Cython - naive ----
        title = "Mandelbrot_Cython_naive"
        folder = "cython_naive"
        if _run(folder, title, res):
            # Start timer
            t_start = timer()

            # calculate the fractals
            mfractal = mb.cython_naive(lim=lim, res_re=p_re, res_im=p_im,
                                       threshold=T, iterations=iterations)

            # Stop timer
            t_stop = timer()
            t_total = t_stop - t_start

            # save data
            _save(mfractal, t_total, folder, title, res)

            # Plot
            _plot(mfractal, lim, folder, title, res)

            print(f"Mandelbrot Cython naive took {t_total}s")

        # ---- Cython - vector ----
        title = "Mandelbrot_Cython_vector"
        folder = "cython_vector"
        if _run(folder, title, res):
            # Start timer
            t_start = timer()

            # calculate the fractals
            mfractal = mb.cython_vector(lim=lim, res_re=p_re, res_im=p_im,
                                        threshold=T, iterations=iterations)

            # Stop timer
            t_stop = timer()
            t_total = t_stop - t_start

            # save data
            _save(mfractal, t_total, folder, title, res)

            # Plot
            _plot(mfractal, lim, folder, title, res)

            print(f"Mandelbrot Cython vector took {t_total}s")
