# misfit/fit2D.py
# Fit 2D emission lines for MISFIT
#
# Copyright 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

from __future__ import print_function

import numpy as np
# import pandas as pd
import copy
import os
# import sys

import pickle

# from scipy.stats import norm

try:
    # import misfit.general.general_utils as utils
    import misfit.general.io as io

except:
    # from ..general import general_utils as utils
    from ..general import io as io

import misfit.plot as misfit_plot

try:
    import fit_core
    import fit_io
    from fit_core import MCMC2DOptions, MCMCResults, FitEmissionLines2DResults
except:
    from . import fit_core
    from . import fit_io
    from .fit_core import MCMC2DOptions, MCMCResults, FitEmissionLines2DResults



# from misfit.model import AperModel2D
from misfit.model import KinModel2D, KinModel2DOptions
from misfit.model.kin_classes import KinProfileFiducial, IntensityProfileFiducial, ThetaPriorFlat
from misfit import Galaxy

import astropy.constants as constants
c_cgs = constants.c.cgs.value
c_kms = c_cgs * 1.e-5 # from cm/s -> km/s
c_AA = c_cgs * 1.e8  # go from cm/s -> AA/s


class FitEmissionLines2D(object):
    """
    Class to fit the kinematics of a 2D spectrum, using a set of line(s).

    Parameters
    ----------
    galaxy : `misfit.Galaxy` instance
        Galaxy instance, holding a `misfit.spec2D` instance 
        containing the galaxy 2D spectrum, uncertainty, masks, etc.
        `galaxy` must have the following structural parameters specified: 
        n, q, R_E (arcsec), delt_PA (calculated as part of fit_prep_calc 
        from the inst+gal+spec2D angles). 
        Can also hold a `misfit.Pstamp` instance for plotting.
        Note `spec2D.instrument_resolution` should be in km/s

    instrument : `misfit.Spectrograph` instance
        Spectrograph instance. Requires the following attributes: 
        instrument_resolution [km/s], PSF_FWHM [arcsec], pixscale [arcsec/pixel]. 

        
    instrument_img: `misfit.Imager` instance
        Imager instance. Requires the imaging PA to be set 
        (for calculating misalignement angles.)


    flux_arr :  array-like
        Initial guess of the flux (erg/s/cm2) 
        of the brightest line for each in the set.

    z : float
        Initial guess of the redshift

    vel_disp : float       
        Initial guess of the line dispersion [km/s]

        
    kinProfile : `misfit.KinProfile` instance, optional
        Kinematic profile class with methods for the velocity and dispersion profiles 
        (`kinProfile.vel(r)`, `kinProfile.sigma(r)`). 
        Default: Arctan velocity + constant dispersion, 
        where the fit theta is [Va, rt, sigma0, m0shift, lam0]

    intensityProfile : `misfit:IntensityProfile` instance, optional
        Intensity profile class, containing info about light dist for galaxy.
        Default: Sersic profile radially + exponential in the vertical direction

    linegroup_name : array-like, optional
        Array of line names to fit, if `linenames_arr` is not specified. 
        eg: ['Halpha', 'NII'], ['OIII'], ['Hbeta']

    linenames_arr : array-like, optional
        Array of line group names.

    restwave_arr : array-like, optional
        Array of restframe wavelengths corresponding to the lines specified by 
        `linenames_arr`.

    flux_ratio_arr : array-like, optional
        Relative flux strengths of the lines in the groups defined in linenames_arr. 

    trim_restwave_range : array-like, optional
        Restframe wavelength range to which to trim the spectra for fitting

    trim_wave_paramfile : string, optional
        Filename with wavelength to trim the spectrum for fitting
        
    del_obs_lam : float, optional
        +- Observed angstroms for optimizing the redshift solution. Default: 20.


    Methods
    -------
    fit : 
        Run the MCMC sampling (with emcee), and analyze the posterior to 
        determine the "best-fit" (maximum a posteriori) values. 

    Notes
    -----
    To specify the lines to be modeled & fit, either input `linegroup_name`, or 
    explicitly set `linenames_arr`, `restwave_arr`, `flux_ratio_arr`.

    Will create a `model` attribute, and has an attribute `bestfit` 
    containing all the best-fit values + errors

    Initializing the instance ensures the fit preparation calculations 
    are done on galaxy.spec2D. Then the trimmed spectrum + prepared calculations 
    to determine **sane** theta bounds.

    """

    # def __init__(self, galaxy, instrument, **kwargs):
    def __init__(self, galaxy=None, instrument=None, instrument_img=None,
                    linegroup_name = None, linenames_arr = None,
                    restwave_arr = None, flux_ratio_arr = None,
                    kinProfile=KinProfileFiducial(),
                    intensityProfile=IntensityProfileFiducial(Galaxy(n=1.,q0=0.19,re_arcsec=0.)),
                    kinModelOptions=KinModel2DOptions(),
                    thetaPrior=None,
                    thetaSettings=None,
                    mcmcOptions = MCMC2DOptions(),
                    theta_linked_posteriors = None,
                    mcmc_results = MCMCResults(),
                    usetex=True,
                    **kwargs):

        self.galaxy = galaxy
        self.instrument = instrument
        self.instrument_img = instrument_img

        # Store stuff about the lines that will be fit:
        self.linegroup_name = linegroup_name
        self.linenames_arr = linenames_arr   # All lines to be included in 2D model
        self.restwave_arr = restwave_arr    # Rest wavelengths of these lines
        self.flux_ratio_arr = flux_ratio_arr  # Relative flux strengths

        # Kinematic profile function. Fiducial takes 5 parameters
        self.kinProfile = kinProfile

        # Intensity profile: Information about how the model light profile is generated
        ## # self.intensityProfile = IntensityProfileFiducial(galaxy=self.galaxy)
        self.intensityProfile = intensityProfile
        self.intensityProfile.update_values(galaxy=self.galaxy)


        # Options for model, to pass when kinModel is created:
        self.kinModelOptions = kinModelOptions

        # Priors for model:
        self.thetaPrior = thetaPrior

        # Parameter info:
        self.thetaSettings = thetaSettings

        # Options for MCMC fitting, including output filenames
        self.mcmcOptions = mcmcOptions

        # If set, an array of arrays giving indices of theta_fitting (ie, theta which vary)
        #       for which best-fit should be calculated in multiD space.
        self.theta_linked_posteriors = theta_linked_posteriors
        self.mcmc_results = mcmc_results     # Class for holding MCMC resuls


        # Kinematic model class
        self.kinModel = None

        # Holds results:
        self.sampler_dict = None

        # Skip prep calcs: eg, fast creation for re-read of saved objects:
        self.skip_fit_prep_calcs = False

        self.usetex = usetex

        self.setAttr(**kwargs)

        # Initializing trims the input 2D spectrum + calculates prep steps.
        self.setup()

    def setAttr(self,**kwargs):
        """Set/update arbitrary attribute list with kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))

    def copy(self):
        return copy.deepcopy(self)

    def setup(self):
        if (self.linenames_arr is None):
            if self.galaxy.spec2D.linenames_arr is None:
                # linenames_arr = []
                # set this from lib if not set:
                d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
                names_file = os.path.join(d, 'line_names_cat.dat')
                self.linenames_arr = io.read_line_names(names_file, name=self.linegroup_name)
                self.galaxy.spec2D.linenames_arr = self.linenames_arr
            else:
                self.linenames_arr = self.galaxy.spec2D.linenames_arr

        if (self.linenames_arr is not None) & (self.restwave_arr is None):
            if (self.galaxy.spec2D.linenames_arr is not None) & \
                    (self.galaxy.spec2D.restwave_arr is None):
                d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
                wave_file = os.path.join(d, 'line_wavelengths_ratios.dat')
                waves_arr = []
                for linename in self.linenames_arr:
                    waves_arr.append(io.read_restwave(wave_file, linename=linename))
                self.restwave_arr = np.array(waves_arr)
                self.galaxy.spec2D.restwave_arr = self.restwave_arr
            else:
                self.restwave_arr = self.galaxy.spec2D.restwave_arr

        if (self.linenames_arr is not None) & (self.flux_ratio_arr is None):
            if (self.galaxy.spec2D.linenames_arr is not None) & (self.galaxy.spec2D.flux_ratio_arr is None):
                d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
                wave_file = os.path.join(d, 'line_wavelengths_ratios.dat')
                line_ratio_arr = []
                for linename in self.linenames_arr:
                    line_ratio_arr.append(io.read_line_ratio(wave_file, linename=linename))
                self.flux_ratio_arr = np.array(line_ratio_arr)
                self.galaxy.spec2D.flux_ratio_arr = self.flux_ratio_arr
            else:
                self.flux_ratio_arr = self.galaxy.spec2D.flux_ratio_arr


        # Setup spectrum:
        if not self.skip_fit_prep_calcs:
            self.galaxy.spec2D.fit_prep_calcs(self.galaxy, self.instrument)


    def fit(self, **kwargs):
        """
        Run the MCMC fitting. Requires input of a `Theta2DSettings` instance,
        which handles the fitting parameters required for the `kinProfile` functions,
        with the last two paramters as `m0_shift` and `z`.
        """
        self.setAttr(**kwargs)

        # Check:
        if self.instrument.instrument_resolution < 0.:
            raise ValueError

        # Setup model + priors, if not already specified
        self.setup_model()

        # Run emcee: also pass a pared
        self.run_mcmc()

        # Do analysis on sampler + add other derived parameters
        self.analyze_results()

        # Save model to disk: entire fit2D to pickle
        self.save_bestfit_model()

        # Plot results:
        self.plot_results()


    def reload_results(self, doPlot=False, resave_bestfit=False,
                    **kwargs):
        """
        Reload results from files, without re-running
        """
        self.setAttr(**kwargs)

        if self.kinModel is None:
            self.setup_model()

        #self.reload_bestfit_model()

        print("  % fit2d: inst_res={:0.3f}".format(self.instrument.instrument_resolution))

        self.analyze_results()

        if resave_bestfit:
            # Save model to disk: entire fit2D to pickle
            self.save_bestfit_model()

        if doPlot:
            # Plot results:
            self.plot_results()


    def analyze_results(self):
        """
        Analyze the MCMC results:
        Calculations on sampler chain, plotting of best-fit values
        """
        if self.sampler_dict is None:
            self.reload_sampler()

        self.get_chain_results()

        self.set_bestfit(self.mcmc_results.bestfit_theta_fitting)


    def plot_results(self):
        self.plot_param_corner()        # Show posterior distributions
        self.plot_bestfit_model()       # Show bestfit 2D line model


    def make_model(self, theta_fitting=None):
        # Use methods in self.kinModel
        self.kinModel.make_model(galaxy=self.galaxy, instrument=self.instrument,
                theta_fitting=theta_fitting)

    def update_model(self, theta_fitting=None):
        self.kinModel.update_model(theta_fitting=theta_fitting)

    def setup_model(self):
        """
        Initialize model, parameters, + bounds.
        Can be called independently of fit.
        Need to set bounds for definiting initial walker positions.
        """
        ## Initialize model handling class:
        self.kinModel = KinModel2D(self.galaxy,
                        thetaSettings=copy.deepcopy(self.thetaSettings),
                        kinProfile=self.kinProfile,
                        intensityProfile=self.intensityProfile,
                        kinModelOptions=self.kinModelOptions)


        # Setup priors, if not already set in beginning
        if self.thetaPrior is None:
            self.thetaPrior = ThetaPriorFlat(theta=copy.deepcopy(self.kinModel.theta),
                                theta_vary=copy.deepcopy(self.kinModel.theta_vary),
                                theta_bounds=copy.deepcopy(self.thetaSettings.theta_bounds))

        if ((self.thetaPrior.name == 'ThetaPriorFlat') & \
                (self.thetaSettings.theta_bounds is not None)):
            self.thetaPrior.theta_bounds = copy.deepcopy(self.thetaSettings.theta_bounds)


        ## Initialize parameters to be fit:
        theta_fitting_init = fit_core.initialize_fitting_theta(self.kinModel.theta_init,
                                self.kinModel.theta_vary)

        # Save information about linking parameters for interpreting posterior distributions:
        self.theta_linked_posteriors = self.thetaSettings.theta_linked_posteriors

        ## Make an initial model:
        self.make_model(theta_fitting=theta_fitting_init)

        print("fitEmis2D.kinModel.aperModel.sigma_floor_value={}\n".format(self.kinModel.aperModel.sigma_floor_value))

    def run_mcmc(self):
        self = fit_core.run_mcmc(self)

    def get_chain_results(self):
        self = fit_core.get_chain_results(self)


    def set_bestfit(self, theta_fitting):
        """
        Save a copy of the best-fit parameter model, including model, theta, theta_var
        """
        self.make_model(theta_fitting)
        self.kinModel.bestfit = self.kinModel.copy()

    def save_bestfit_model(self):
        """
        Pickle the entire fitEmis2D class to a pickle for restoring later.
        May be *very large* depending on original data saved in the fitEmis2D class.
        """

        # Make the best-fit results class:

        fitEmis2DResults = self.make_pruned_fitEmis2D_result_class()


        if self.mcmcOptions.filename_bestfit_model is not None:
            pickle.dump(fitEmis2DResults, open(self.mcmcOptions.filename_bestfit_model, "wb"))
            #pickle.dump(self, open(self.mcmcOptions.filename_bestfit_model, "wb"))


    def make_pruned_fitEmis2D_result_class(self):

        if self.instrument_img is not None:
            inst_img = self.instrument_img.copy()
        else:
            inst_img = None

        fitEmis2DResults = FitEmissionLines2DResults(galaxy=self.galaxy.copy(),
                                instrument=self.instrument.copy(),
                                instrument_img=inst_img)

        # Copy over relevent attributes:
        # for key in fitEmis2DResults.galaxy.__dict__.keys():
        #     fitEmis2DResults.galaxy.__dict__[key] = copy.deepcopy(self.galaxy.__dict__[key])

        for key in fitEmis2DResults.__dict__.keys():
            #if key != 'galaxy':
            fitEmis2DResults.__dict__[key] = copy.deepcopy(self.__dict__[key])


        # Clean up a few things:
        del fitEmis2DResults.galaxy.pstamp
        if 'spec_hdr' in fitEmis2DResults.galaxy.spec2D.__dict__.keys():
            del fitEmis2DResults.galaxy.spec2D.spec_hdr


        return fitEmis2DResults

    def reload_sampler(self):
        if self.mcmcOptions.filename_sampler is not None:
            # Save stuff to file, for future use:
            self.sampler_dict = fit_io.loadpickle(self.mcmcOptions.filename_sampler)


    def reload_bestfit_model(self):
        """
        Restore pickled fitEmis2D class, including analysis calculations.
        """
        if self.mcmcOptions.filename_bestfit_model is not None:
            # fitEmis2D = pickle.load(open(self.mcmcOptions.filename_bestfit_model, "rb"))
            # for k in fitEmis2D.__dict__.keys():
            #     self.__dict__[k] = fitEmis2D.__dict__[k]
            fitEmis2DResults = pickle.load(open(self.mcmcOptions.filename_bestfit_model, "rb"))
            for k in fitEmis2DResults.__dict__.keys():
                self.__dict__[k] = fitEmis2DResults.__dict__[k]

    def plot_bestfit_model(self):
        #if self.mcmcOptions.filename_plot_bestfit is not None:
        misfit_plot.plot_bestfit_model(self,
            fileout=self.mcmcOptions.filename_plot_bestfit,
            verbose=False, usetex=self.usetex)

    def plot_param_corner(self):
        #if self.mcmcOptions.filename_plot_param_corner is not None:
        misfit_plot.plot_param_corner(self,
            fileout=self.mcmcOptions.filename_plot_param_corner,
            verbose=False, usetex=self.usetex)
