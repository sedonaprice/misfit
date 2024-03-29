# Copyright Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import matplotlib
try:
    os.environ["DISPLAY"]
except:
    matplotlib.use("agg")

from .galaxy import GalaxyBasic, Galaxy, Spectrum, ObsSpectrum, \
        ObsSpectrum1D, ObsSpectrum2D, ObsSpectrum2DBasic, Pstamp
from .instrument import PSFBase, PSFGaussian, PSFMoffat, Instrument, \
        Spectrograph, Imager

try:
    import fit
    import general
    import model
    import plot
    import mock

    from galaxy import Galaxy

    from model.emission_lines_model import EmissionLinesSpectrum1DModel

    from model.kin_model import KinModel2DOptions, KinModel2D
    from model.kin_classes import Theta2DSettings, Theta2DSettingsFiducial, ThetaPriorFlat, \
                            KinProfile, KinProfileFiducial, \
                            IntensityProfileFiducial, IntensityProfileSersicExpScale, \
                            IntensityProfile

    from fit import FitEmissionLines2D
    from fit.fit_core import MCMC2DOptions, MCMCResults

except:
    from . import fit
    from . import general
    from . import model
    from . import plot
    from . import mock

    from .galaxy import Galaxy

    from .model.emission_lines_model import EmissionLinesSpectrum1DModel

    from .model.kin_model import KinModel2DOptions, KinModel2D
    from .model.kin_classes import Theta2DSettings, Theta2DSettingsFiducial, ThetaPriorFlat, \
                            KinProfile, KinProfileFiducial, \
                            IntensityProfileFiducial, IntensityProfileSersicExpScale, \
                            IntensityProfile

    from .fit import FitEmissionLines2D
    from .fit.fit_core import MCMC2DOptions, MCMCResults



__version__ = "0.0.alpha"

