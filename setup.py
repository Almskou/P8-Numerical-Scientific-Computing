# -*- coding: utf-8 -*-
"""
Mini project for the course Numerical Scientific Computing

Setup script used to create the cython module

@author: 871
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("Mandelbrot_cython.pyx")
)