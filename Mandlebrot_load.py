# -*- coding: utf-8 -*-
"""
Mini project for the course Numerical Scientific Computing

Loads all the created files and save them in tuples. 

@author: Nicolai Almskou
"""

# %% Imports
import matplotlib.pyplot as plt

import numpy as np

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
def _plot(mfractal, lim, directory, title, res):
    """
    
    Plots the mandlbrot based on values from the different methods
    
    Parameters
    ----------
    mfractal : matrix
        matrix with values from the mandlebrot methods
    lim : array
        array with limits. [x_min, x_max, y_min, y_max].
    directory : string
        subfolder name inside the data folder. e.g. "naive"
    title : string
        title of files. e.g. "Mandlebrot_Naive"
    res : int
        the resolution

    Returns
    -------
    None.

    """
    x_min, x_max, y_min, y_max = lim    
    
    # Make plot and save figure
    plt.imshow(np.log(mfractal), cmap=plt.cm.hot, extent=[x_min, x_max, y_min, y_max])
    plt.title(f"{title}_{res}")
    plt.xlabel('Re[c]')
    plt.ylabel('Im[c]')
    plt.savefig(f"data/{directory}/{title}_{res}.pdf", bbox_inches='tight', pad_inches=0.05)
    #plt.show()
    plt.close()

# %% Main
if __name__ == '__main__':
    # Number of processes
    p = 8
    
    # Constants - Limits
    lim = [-2, 1, -1.5, 1.5] # [x_min, x_max, y_min, y_max]

    # Constants - Resolution
    res = [100, 500, 1000, 2000, 5000]
    
    
     # Constants - Threshold
    T = 2

    # Constants - Number of Iterations
    iterations = 100
 
    # Load
    title = ["Mandlebrot_Naive", "Mandlebrot_Numba", "Mandlebrot_Numpy", "Mandlebrot_Multiprocessing",
             "Mandlebrot_Dask", "Mandlebrot_GPU", "Mandlebrot_Cython_naive", "Mandlebrot_Cython_vector"]
    folder = [ "naive", "numba", "numpy", "multiprocessing", "dask", "GPU", "cython_naive", "cython_vector"]
    mfractal = []
    t = []
    for j in range(len(title)):
        print(title[j])
        for i in range(len(res)):
            print(i)
            # assign res
            p_re, p_im = [res[i], res[i]]
            
            #print(f"res: {res[i]}")
           
            t_output , mfractal_output = _load(folder[j],title[j],res[i])
            if i == 0:
                mfractal.append([mfractal_output])
            else:
                mfractal[j].append(mfractal_output)
    
            
            # Plot
            _plot(mfractal[j][i], lim, folder[j], title[j], res[i])
            