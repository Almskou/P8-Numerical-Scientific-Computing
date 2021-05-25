"""
Mini project for the course Numerical Scientific Computing

Functions which are going to be compiled to c

@author: 871
"""

import numpy as np


def naive(complex c, int T, int I):
    cdef complex z = 0
    cdef int i
    
    for i in range(1, I):
        z = z**2 + c
        
        if np.abs(z) > T:
            return i, z
        
    return 100,z


def vector(c, int T, int I):  
    z = np.zeros(c.shape, dtype=complex)
    Iota = np.full(z.shape, I)

    for i in range(1, I+1):
        z[Iota > (i-1)] = np.square(z[Iota > (i-1)]) + c[Iota > (i-1)]   # Only calculated for the ones which hasn't diverged 
        
        Iota[np.logical_and(np.abs(z) > T, Iota == I)] = i # Write iteration number to the matrix 

    return Iota, z
