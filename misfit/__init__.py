#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib
#matplotlib.use('agg')
try:
    _os.environ["DISPLAY"] 
except:
    matplotlib.use("agg")

from .galaxy import *
from .instrument import *

import fit
import general
import model
import plot
import mock

from model.emission_lines_model import EmissionLinesSpectrum1DModel

from model.kin_model import KinModel2DOptions, KinModel2D
from model.kin_classes import Theta2DSettings, Theta2DSettingsFiducial, ThetaPriorFlat, \
                        KinProfile, KinProfileFiducial, \
                        IntensityProfileFiducial, IntensityProfile
                        
from fit import FitEmissionLines2D
from fit.fit_core import MCMC2DOptions, MCMCResults



__version__ = "0.0"


# Should define tests here