# -*- coding: utf-8 -*-
"""
Mini project for the course Numerical Scientific Computing

Setup script used to create the cython module

@author: Nicolai Almskou
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("Mandlebrot_cython.pyx")
)