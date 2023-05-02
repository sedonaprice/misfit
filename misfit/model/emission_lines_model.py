# misfit/emission_lines_model.py
# Define emission lines for MISFIT: both 1D and 2D
#
# Copyright 2014, 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Hidden modules prepended with '_'
from __future__ import print_function

import numpy as np
# import pandas as pd
# import copy
import os
# import sys

# import misfit.general.general_utils as utils
import misfit.general.io as io
from scipy.stats import norm
# import six


import astropy.constants as constants
c_cgs = constants.c.cgs.value
c_kms = c_cgs * 1.e-5 # from cm/s -> km/s
c_AA = c_cgs * 1.e8  # go from cm/s -> AA/s


class EmissionLinesSpectrum1DModel(object):
    """
    Class to define a spectrum made of multiple linesets set of emission lines 
    and profiles for 1D spectra of a single lineset (eg, Ha, OIII doublet, ...)

    Parameters
    ----------
    names_arr : array-like
        Array of line names to fit, if `linenames_arr` is not specified. 
        eg: ['Halpha', 'NII'], ['OIII'], ['Hbeta']

    flux_arr :  array-like
        Flux in flam (erg/s/cm2/Angstrom) of the brightest line for each in the set.

    z : float
        Redshift

    vel_disp : float       
        Line dispersion [km/s]

    cont_coeff : float
        Coefficient of the continuum in 1.e-18 erg/s/cm2/Angstrom

    cont_order : int
        Order of the continuum fit. Default: 1

    shape1D : string, optional
        Type of 1D profile shape to fit. Options: ['gaussian']. Default: 'gaussian'

        
    Methods
    -------
    make1DProfile : 
        Generate 1D profile

    residual1DProfile : 
        Evaluate residual of the 1D profile relative to an input 1D spectrum. 

    """

    def __init__(self, **kwargs):
        self.names_arr = None

        self.vel_disp = None    # dispersion of the lines (std dev) -- [km/s]
        self.z = None           # redshift
        self.flux_arr = None    # flux of brightest line
        self.cont_coeff = None

        self.cont_order = None     # Order of the continuum fit (all in 1.e-18 flam): 0 is intercept only, etc
        self.shape1D = 'gaussian'  # eg, use

        self.obswave_arr = None

        #self.profile = None

        self.setAttr(**kwargs)

    def setAttr(self,**kwargs):
        """Set/update arbitrary attribute list with kwargs"""
        self.__dict__.update(kwargs)

        if self.shape1D != 'gaussian':
            raise ValueError('Non-gaussian line profiles are currently not supported')


    #
    def make1DProfile(self, params, obswave_arr):
        """
        Make a 1D profile for the line(s), including continuum
        """

        z = params['z'].value
        vel_disp = params['vel_disp'].value

        cont_coeff = np.array([])
        for i in range(self.cont_order+1):
            cont_coeff = np.append(cont_coeff, params['cont_coeff'+str(i)].value*1.e-18)
        #
        flux_arr = np.array([])
        for i in range(len(self.names_arr)):
            flux_arr = np.append(flux_arr, params['flux'+str(i)].value)

        # Update values
        self.obswave_arr = obswave_arr
        self.z = z
        self.vel_disp = vel_disp
        self.cont_coeff = cont_coeff
        self.flux_arr = flux_arr

        profile = np.zeros(len(obswave_arr))

        for i in range(len(self.names_arr)):
            lineModel = EmissionLines1DModel(name=self.names_arr[i])
            lineModel.make1DProfile(obswave_arr, z, vel_disp, self.flux_arr[i])
            profile += lineModel.profile

        # Now add the continuum:
        #cont_profile = np.zeros(len(obswave_arr))
        for i in range(self.cont_order+1):
            profile += cont_coeff[i]*np.power(obswave_arr,i)

        #profile += cont_profile

        #self.profile = profile
        return profile

    def residual1DProfile(self, params, obswave_arr, data, err, mask):
        # Mask bad data:
        wh_good = np.where(mask == 1)[0]

        model = self.make1DProfile(params, obswave_arr[wh_good])
        resid = (model-data[wh_good])/err[wh_good]
        # mask non-finite elements:
        resid[~np.isfinite(resid)] = 0.

        return resid


class EmissionLines(object):
    """
    Define basics of emission line sets (or singles)
    """
    def __init__(self, name, linenames_arr, restwave_arr, flux_ratio_arr):
        self.name = None
        self.linenames_arr = None
        self.restwave_arr = None
        self.flux_ratio_arr = None




class EmissionLines1DModel(EmissionLines):
    """
    Class to define an emission line set and profiles for 1D spectra of 
    a single lineset (eg, Ha, OIII doublet, ...)

    Parameters
    ----------

    flux : float
        Flux of the brightest line, in erg/s/cm2

    z : float
        Redshift

    vel_disp : float
        Line velocity dispersion [km/s]

    profile_function : function, optional
        Shape of the 1D line profile. Input is assumed to be: 
        x (Angstroms), mu (Angstroms), vel_sig (km/s), and total flux (erg/s/cm2)
        Default: Gaussian (`gaussian_line()`)

    name : string, optional
        Line/line set name. eg, Halpha, OIII
    
    linenames_arr : array-like, optional
        Array of line names in line set. eg, ['HA6565'] or ['OIII5008', 'OIII4960']
        Can be set from lib info based on the `name` attribute.

    restwave_arr : array-like, optional
        Restframe wavelength of the lines named in `linenames_arr`. 
        Can be set from lib info based on the `name` attribute.

    flux_ratio_arr : array-like, optional
        Ratio of line strengths for the lines named in `linenames_arr`, 
        normalized to 1. for the strongest line. eg, [1.], [1., 0.3333]
        Can be set from lib info based on the `name` attribute.
        
    Methods
    -------
    make1DProfile : 
        Generate 1D profile

    Notes
    -----
    To specify the specific line(s), either input `name`, or 
    explicitly set `linenames_arr`, `restwave_arr`, and `flux_ratio_arr`.

    """
    def __init__(self, profile_function=None, **kwargs):

        EmissionLines.__init__(self,None,None,None,None)

        self.vel_disp = None    # dispersion of the lines (std dev) -- [km/s]
        self.z = None           # redshift
        self.flux = None        # flux of brightest line

        self.obswave_arr = None
        self.profile = None

        self.profile_func = profile_function if profile_function else gaussian_line

        self.setAttr(**kwargs)

    def setAttr(self,**kwargs):
        """Set/update arbitrary attribute list with kwargs"""
        self.__dict__.update(kwargs)

        if (self.linenames_arr is None):
            # set this from lib if not set:
            d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
            names_file = os.path.join(d, 'line_names_cat.dat')
            self.linenames_arr = io.read_line_names(names_file, name=self.name)

        if (self.linenames_arr is not None) & ((self.restwave_arr is None) | \
                (self.flux_ratio_arr is None) ):
            d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
            wave_file = os.path.join(d, 'line_wavelengths_ratios.dat')

            wave_arr = []
            line_ratio_arr = []
            for linename in self.linenames_arr:
                wave_arr.append(io.read_restwave(wave_file, linename=linename))
                line_ratio_arr.append(io.read_line_ratio(wave_file, linename=linename))

            self.restwave_arr = np.array(wave_arr)
            self.flux_ratio_arr = np.array(line_ratio_arr)



    def make1DProfile(self, obswave_arr, z, vel_disp, flux):
        """
        Make a 1D profile for the given line set -- excluding any continuum
        """

        # Update values
        self.obswave_arr = obswave_arr
        self.z = z
        self.vel_disp = vel_disp
        self.flux = flux

        profile = np.zeros(len(obswave_arr))

        for i in range(len(self.linenames_arr)):
            profile += self.profile_func(self.obswave_arr,
                            self.restwave_arr[i]*(1.+self.z), self.vel_disp, self.flux*self.flux_ratio_arr[i])

        self.profile = profile





def gaussian_line(x, mu, vel_sig, flux):
    # Mu: wavelength

    # Convert vel_sig into sig in wavelength:
    sig = vel_sig / c_kms * mu

    # make gaussian
    gaus = norm.pdf(x, mu, sig)
    # scale to amplitude
    scale = flux/(sig*np.sqrt(2.*np.pi))
    gaus = gaus/gaus.max()*scale
    return gaus
