# The main program

# misfit/fit.py
# Fit 1D + 2D emission lines for MISFIT
# 
# Copyright 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

try:
    from fit1D import *
    from fit2D import *
    from fit_core import MCMC2DOptions
except:
    from .fit1D import *
    from .fit2D import *
    from .fit_core import MCMC2DOptions