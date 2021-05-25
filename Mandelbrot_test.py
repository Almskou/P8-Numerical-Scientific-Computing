# -*- coding: utf-8 -*-
"""
Mini project for the course Numerical Scientific Computing

Test to see if mandelbrots created with different methods is
equal to the naive version

Can be run by a termial by: "python -m python [dir to file]"
or in console by running the command "pytest.main()"

@author: 871
"""

# %% Imports
import numpy as np
import pytest


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
        the resolution.

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


@pytest.mark.parametrize("title, folder", [("Mandelbrot_Numba", "numba"),
                                           ("Mandelbrot_Numpy", "numpy"),
                                           ("Mandelbrot_Multiprocessing",
                                            "multiprocessing"),
                                           ("Mandelbrot_Dask", "dask"),
                                           ("Mandelbrot_GPU", "GPU"),
                                           ("Mandelbrot_Cython_naive",
                                            "cython_naive"),
                                           ("Mandelbrot_Cython_vector",
                                            "cython_vector")])
@pytest.mark.parametrize("res", [100, 500, 1000, 2000, 5000])
def test_mandelbrot_compare(title, folder, res):
    # Load naive
    _, mfractal_naive = _load("naive", "Mandelbrot_Naive", res)

    # Load other
    _, mfractal_compare = _load(folder, title, res)
    # Check if they are equal

    assert (mfractal_naive == mfractal_compare).all()
