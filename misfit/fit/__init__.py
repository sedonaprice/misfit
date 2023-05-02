# The main program

# misfit/fit.py
# Fit 1D + 2D emission lines for MISFIT
# 
# Copyright 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

try:
    from fit1D import FitEmissionLines1D
    from fit2D import FitEmissionLines2D
    from fit_core import MCMC2DOptions, MCMCResults, FitEmissionLines2DResults
except:
    from .fit1D import FitEmissionLines1D
    from .fit2D import FitEmissionLines2D
    from .fit_core import MCMC2DOptions, MCMCResults, FitEmissionLines2DResults