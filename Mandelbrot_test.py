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
import pytest
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
        the resolution.

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


@pytest.mark.parametrize("title, folder", [("Mandelbrot_Numba", "numba"),
                                           ("Mandelbrot_Numpy", "numpy"),
                                           ("Mandelbrot_Multiprocessing_1",
                                            "multiprocessing_1"),
                                           ("Mandelbrot_Multiprocessing_2",
                                            "multiprocessing_2"),
                                           ("Mandelbrot_Multiprocessing_4",
                                            "multiprocessing_4"),
                                           ("Mandelbrot_Multiprocessing_8",
                                            "multiprocessing_8"),
                                           ("Mandelbrot_Multiprocessing_16",
                                            "multiprocessing_16"),
                                           ("Mandelbrot_Dask_1", "dask_1"),
                                           ("Mandelbrot_Dask_2", "dask_2"),
                                           ("Mandelbrot_Dask_4", "dask_4"),
                                           ("Mandelbrot_Dask_8", "dask_8"),
                                           ("Mandelbrot_Dask_16", "dask_16"),
                                           ("Mandelbrot_GPU", "GPU"),
                                           ("Mandelbrot_Cython_naive",
                                            "cython_naive"),
                                           ("Mandelbrot_Cython_vector",
                                            "cython_vector")])
@pytest.mark.parametrize("res", [100, 500, 1000, 2000, 5000])
def test_mandelbrot_mfractal_compare(title, folder, res):
    """
    Test the mandelbrot fractals against naive implementation

    Parameters
    ----------
    title : str
        title of the data.
    folder : str
        folder name to the data inside the data folder.
    res : int
        resolution.

    Returns
    -------
    None.

    """
    # Load naive
    _, mfractal_naive, _ = _load("naive", "Mandelbrot_Naive", res)

    # Load other
    _, mfractal_compare, _ = _load(folder, title, res)
    # Check if they are equal

    assert (mfractal_naive == mfractal_compare).all()


@pytest.mark.parametrize("title, folder", [("Mandelbrot_Numba", "numba"),
                                           ("Mandelbrot_Numpy", "numpy"),
                                           ("Mandelbrot_Multiprocessing_1",
                                            "multiprocessing_1"),
                                           ("Mandelbrot_Multiprocessing_2",
                                            "multiprocessing_2"),
                                           ("Mandelbrot_Multiprocessing_4",
                                            "multiprocessing_4"),
                                           ("Mandelbrot_Multiprocessing_8",
                                            "multiprocessing_8"),
                                           ("Mandelbrot_Multiprocessing_16",
                                            "multiprocessing_16"),
                                           ("Mandelbrot_Dask_1", "dask_1"),
                                           ("Mandelbrot_Dask_2", "dask_2"),
                                           ("Mandelbrot_Dask_4", "dask_4"),
                                           ("Mandelbrot_Dask_8", "dask_8"),
                                           ("Mandelbrot_Dask_16", "dask_16"),
                                           ("Mandelbrot_GPU", "GPU"),
                                           ("Mandelbrot_Cython_naive",
                                            "cython_naive"),
                                           ("Mandelbrot_Cython_vector",
                                            "cython_vector")])
@pytest.mark.parametrize("res", [100, 500, 1000, 2000, 5000])
def test_mandelbrot_z_compare(title, folder, res):
    """
    Test the z values against naive implementation

    Parameters
    ----------
    title : str
        title of the data.
    folder : str
        folder name to the data inside the data folder.
    res : int
        resolution.

    Returns
    -------
    None.

    """
    # Load naive
    _, _, z_naive = _load("naive", "Mandelbrot_Naive", res)

    # Load other
    _, _, z_compare = _load(folder, title, res)

    # Check if they are equal
    assert (z_naive == z_compare).all()
