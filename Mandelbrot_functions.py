# -*- coding: utf-8 -*-
"""
Mini project for the course Numerical Scientific Computing

All the different mandelbrot functions

@author: 871
"""
# %% Imports
import Mandelbrot_cython

import numpy as np

from functools import partial
import multiprocessing as mp

from dask.distributed import Client, wait

import pyopencl as cl

from numba import jit

# %% Mandelbrot Naive


def naive(lim, res_re, res_im, threshold, iterations):
    """
    Naive implementation - for loops

    Parameters
    ----------
    lim : array
        array with limits. [x_min, x_max, y_min, y_max].
    res_re : int
        the resolution for real part
    res_im : int
        the resolution for imaginary part
    threshold : int
        threshold for the mandelbrot function.
    iterations : int
        max number of iterations.

    Returns
    -------
    mfractal : matrix
        A matrix with the mandelbrot values (number iterations).

    """
    x_min, x_max, y_min, y_max = lim
    # Form the complex grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                       np.linspace(y_min, y_max, res_im, endpoint=True))
    c = x + 1j*y

    # Compute and plot the Mandelbrot fractal
    mfractal = np.zeros(c.shape, dtype=np.float)
    z = np.zeros(c.shape, dtype=np.complex)
    for ix in range(mfractal.shape[1]):
        for iy in range(mfractal.shape[0]):
            mfractal[ix, iy], z[ix, iy] = _mpoint_naive(c[ix, iy],
                                                        threshold, iterations)

    return mfractal, z

# %% Mandelbrot Numpy


def numpy(lim, res_re, res_im, threshold, iterations):
    """
    Numpy implementation - Vectorised

    Parameters
    ----------
    lim : array
        array with limits. [x_min, x_max, y_min, y_max].
    res_re : int
        the resolution for real part
    res_im : int
        the resolution for imaginary part
    threshold : int
        threshold for the mandelbrot function.
    iterations : int
        max number of iterations.

    Returns
    -------
    mfractal : matrix
        A matrix with the mandelbrot values (number iterations).

    """
    x_min, x_max, y_min, y_max = lim
    # Form the complex grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                       np.linspace(y_min, y_max, res_im, endpoint=True))
    c = (x + 1j*y)

    # Compute the Mandelbrot fractal
    mfractal = np.zeros(c.shape, dtype=np.float)
    z = np.zeros(c.shape, dtype=np.complex)

    z = np.zeros(c.shape, dtype=np.complex)
    mfractal, z = _mpoint_numpy(c, threshold, iterations)

    return mfractal, z

# %% Mandelbrot Numba


def numba(lim, res_re, res_im, threshold, iterations):
    """
    Numba implementation

    Parameters
    ----------
    lim : array
        array with limits. [x_min, x_max, y_min, y_max].
    res_re : int
        the resolution for real part
    res_im : int
        the resolution for imaginary part
    threshold : int
        threshold for the mandelbrot function.
    iterations : int
        max number of iterations.

    Returns
    -------
    mfractal : matrix
        A matrix with the mandelbrot values (number iterations).

    """
    x_min, x_max, y_min, y_max = lim
    # Form the complex grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                       np.linspace(y_min, y_max, res_im, endpoint=True))
    c = x + 1j*y

    # Compute and plot the Mandelbrot fractal
    mfractal = np.zeros(c.shape, dtype=np.float)
    z = np.zeros(c.shape, dtype=np.complex)

    for ix in range(mfractal.shape[1]):
        for iy in range(mfractal.shape[0]):
            mfractal[ix, iy], z[ix, iy] = _mpoint_numba(c[ix, iy],
                                                        threshold, iterations)

    return mfractal, z


# %% Mandelbrot Multiprocessing


def multiprocessing(lim, res_re, res_im, threshold, iterations, p):
    """
    Multiprocessing implementation - Made with python package "multiprocessing"

    Parameters
    ----------
    lim : array
        array with limits. [x_min, x_max, y_min, y_max].
    res_re : int
        the resolution for real part
    res_im : int
        the resolution for imaginary part
    threshold : int
        threshold for the mandelbrot function.
    iterations : int
        max number of iterations.
    p : int
        number of worker processes.

    Returns
    -------
    result : matrix
        A matrix with the mandelbrot values (number iterations).

    """
    x_min, x_max, y_min, y_max = lim
    # Form the complex grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                       np.linspace(y_min, y_max, res_im, endpoint=True))
    c = x + 1j*y

    # Create a pool with p processes
    pool = mp.Pool(processes=p)

    # Take the _mpoint function and specify the two constants
    _mpoint_p = partial(_mpoint_numpy, T=threshold, Iter=iterations)

    results = [pool.apply_async(_mpoint_p, [c[ix, :]]) for ix in range(len(x))]
    result = [result.get() for result in results]

    # Split the resutls into z and mfractals
    mfractal = np.zeros(c.shape, dtype=np.float)
    z = np.zeros(c.shape, dtype=np.complex)

    z = [result[i][1] for i in range(len(x))]
    mfractal = [result[i][0] for i in range(len(x))]

    return mfractal, z


# %% Mandelbrot Dask


def dask(lim, res_re, res_im, threshold, iterations, p):
    """
    Multiprocessing implementation - Made with python package "dask"

    Parameters
    ----------
    lim : array
        array with limits. [x_min, x_max, y_min, y_max].
    res_re : int
        the resolution for real part
    res_im : int
        the resolution for imaginary part
    threshold : int
        threshold for the mandelbrot function.
    iterations : int
        max number of iterations.
    p : int
        number of workers.

    Returns
    -------
    mfractal : matrix
        A matrix with the mandelbrot values (number iterations).

    """
    x_min, x_max, y_min, y_max = lim
    # Form the complex grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                       np.linspace(y_min, y_max, res_im, endpoint=True))
    c = x + 1j*y

    # Compute and plot the Mandelbrot fractal
    mfractal = np.zeros(c.shape, dtype=np.float)
    z = np.zeros(c.shape, dtype=np.complex)

    client = Client(n_workers=p)
    # Take the _mpoint function and specify the two constants
    _mpoint_p = partial(_mpoint_numpy, T=threshold, Iter=iterations)

    # Maps all the values from 'c' into the function _mpoint_p
    futures = []
    # futures = client.map(_mpoint_p, c[range(len(x)), :])
    futures = client.map(_mpoint_p, c)

    results = client.gather(futures)

    client.close()
    # Split the resutls into z and mfractals
    z = [results[i][1] for i in range(len(x))]
    mfractal = [results[i][0] for i in range(len(x))]

    return mfractal, z

# %% Mandelbrot GPU


def GPU(lim, res_re, res_im, threshold, iterations):
    """
    GPU mplementation - Made with python package "pyopencl"

    Parameters
    ----------
    lim : array
        array with limits. [x_min, x_max, y_min, y_max].
    res_re : int
        the resolution for real part
    res_im : int
        the resolution for imaginary part
    threshold : int
        threshold for the mandelbrot function.
    iterations : int
        max number of iterations.

    Returns
    -------
    mfractal : matrix
        A matrix with the mandelbrot values (number iterations).

    """
    x_min, x_max, y_min, y_max = lim

    # Form the complex grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                       np.linspace(y_min, y_max, res_im, endpoint=True))
    c = x + 1j*y

    # Compute the Mandelbrot fractal
    mfractal = np.zeros([res_re, res_im], dtype=np.float64)
    z = np.zeros(c.shape, dtype=np.complex128)

    mfractal, z = _mpoint_opencl(c.astype(np.complex128),
                                 threshold, iterations)

    return mfractal, z

# %% Mandelbrot Cython


def cython_vector(lim, res_re, res_im, threshold, iterations):
    """
    Cython Vector mplementation - Made with python package "Cython" and "Numpy"

    Parameters
    ----------
    lim : array
        array with limits. [x_min, x_max, y_min, y_max].
    res_re : int
        the resolution for real part
    res_im : int
        the resolution for imaginary part
    threshold : int
        threshold for the mandelbrot function.
    iterations : int
        max number of iterations.

    Returns
    -------
    mfractal : matrix
        A matrix with the mandelbrot values (number iterations).

    """
    x_min, x_max, y_min, y_max = lim

    # Form the complex grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                       np.linspace(y_min, y_max, res_im, endpoint=True))
    c = x + 1j*y

    # Compute the Mandelbrot fractal
    mfractal = np.zeros([res_re, res_im], dtype=np.float64)
    z = np.zeros(c.shape, dtype=np.complex)

    mfractal, z = Mandelbrot_cython.vector(c, threshold, iterations)

    return mfractal, z


def cython_naive(lim, res_re, res_im, threshold, iterations):
    """
    Cython Naive mplementation - Made with python package "Cython"

    Parameters
    ----------
    lim : array
        array with limits. [x_min, x_max, y_min, y_max].
    res_re : int
        the resolution for real part
    res_im : int
        the resolution for imaginary part
    threshold : int
        threshold for the mandelbrot function.
    iterations : int
        max number of iterations.

    Returns
    -------
    mfractal : matrix
        A matrix with the mandelbrot values (number iterations).

    """
    x_min, x_max, y_min, y_max = lim
    # Form the complex grid

    x, y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                       np.linspace(y_min, y_max, res_im, endpoint=True))
    c = x + 1j*y

    # Compute and plot the Mandelbrot fractal
    mfractal = np.zeros(c.shape, dtype=np.float)
    z = np.zeros(c.shape, dtype=np.complex)

    for ix in range(mfractal.shape[0]):
        for iy in range(mfractal.shape[1]):
            mfractal[ix, iy], z[ix, iy] = Mandelbrot_cython.naive(c[ix, iy],
                                                                  threshold,
                                                                  iterations)

    return mfractal, z


# %% mpoint


def _mpoint_numpy(c, T, Iter):
    """
    Calculation of the number of iterations -  Made with python package "Numpy"

    Parameters
    ----------
    c : Array of complex numbers

    T : int, optional
        Threshold. The default is 2.
    Iter : int, optional
        Maximum number of iterations. The default is 100.

    Returns
    -------
    Iota : int
        Number of iterations.
    """
    z = np.zeros(c.shape, dtype=complex)
    Iota = np.full(z.shape, Iter)

    for i in range(1, Iter):
        # Only calculated for the ones which hasn't diverged
        z[Iota > (i-1)] = np.square(z[Iota > (i-1)]) \
            + c[Iota > (i-1)]

        # Write iteration number to the matrix
        Iota[np.logical_and(np.abs(z) > T, Iota == Iter)] = i

    return Iota, z


def _mpoint_naive(c, T=2, Iter=100):
    """
    Calculation of the number of iterations

    Parameters
    ----------
    c : complex number

    T : int, optional
        Threshold. The default is 2.
    Iter : int, optional
        Maximum number of iterations. The default is 100.

    Returns
    -------
    Iota : int
        Number of iterations.
    """
    z = 0
    for i in range(1, Iter):
        z = z**2 + c

        if np.abs(z) > T:
            return i, z
    return 100, z


@jit
def _mpoint_numba(c, T=2, Iter=100):
    """
    Calculation of the number of iterations -  Made with python package "numba"

    Parameters
    ----------
    c : complex number

    T : int, optional
        Threshold. The default is 2.
    Iter : int, optional
        Maximum number of iterations. The default is 100.

    Returns
    -------
    Iota : int
        Number of iterations.
    """
    z = 0
    for i in range(1, Iter):
        z = z**2 + c

        if np.abs(z) > T:
            return i, z
    return 100, z


def _mpoint_opencl(c, T=2, Iter=100):
    """
    Calculation of the number of iterations
    Made with python package "pyopencl"

    Parameters
    ----------
    c : Array of complex numbers

    T : int, optional
        Threshold. The default is 2.
    Iter : int, optional
        Maximum number of iterations. The default is 100.

    Returns
    -------
    Iota : int
        Number of iterations.
    """
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    output_m = np.empty(c.shape, dtype=np.int32)
    output_z = np.empty(c.shape, dtype=np.complex128)

    mf = cl.mem_flags
    c_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    m_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output_m.nbytes)
    z_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output_z.nbytes)

    prg = cl.Program(
        ctx,
        """
        #define PYOPENCL_DEFINE_CDOUBLE
        #include <pyopencl-complex.h>
        __kernel void mandelbrot
                 (
                 __global const cdouble_t *c,
                 __global int *out_m,
                 __global cdouble_t *out_z,
                 ushort const T,
                 ushort const I,
                 int dim
                 )
        {
            int idx = get_global_id(0);
            int idy = get_global_id(1);

            cdouble_t z = cdouble_new(0,0);

        int i = 0;
        while(i < I & cdouble_abs(z) <= T)
        {
            z = cdouble_mul(z,z);
            z = cdouble_add(z,c[idy*dim + idx]);
            i = i + 1;
        }
            out_z[idy*dim + idx] = z;
            out_m[idy*dim + idx] = i;
        }
        """,
        ).build()

    prg.mandelbrot(
        queue, output_m.shape, None, c_opencl, m_opencl, z_opencl,
        np.uint16(T), np.uint16(Iter), np.int32(c.shape[0])
    )

    cl.enqueue_copy(queue, output_m, m_opencl).wait()
    cl.enqueue_copy(queue, output_z, z_opencl).wait()

    return output_m, output_z
