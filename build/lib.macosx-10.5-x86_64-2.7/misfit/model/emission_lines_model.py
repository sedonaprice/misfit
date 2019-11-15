# misfit/emission_lines_model.py
# Define emission lines for MISFIT: both 1D and 2D
# 
# Copyright 2014, 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Hidden modules prepended with '_'
from __future__ import print_function

import numpy as _np
import pandas as _pd
import copy as _copy
import os as _os
import sys as _sys

import misfit.general.general_utils as _utils
import misfit.general.io as _io
from scipy.stats import norm
from astropy.extern import six as _six


import astropy.constants as _constants
c_cgs = _constants.c.cgs.value
c_kms = c_cgs * 1.e-5 # from cm/s -> km/s
c_AA = c_cgs * 1.e8  # go from cm/s -> AA/s


#
class EmissionLinesSpectrum1DModel(object):
    """
    Define a spectrum made of multiple linesets set of emission lines and profiles for 1D spectra of a single lineset 
        (eg, Ha, OIII doublet, ...)
    
    Input:
        names_arr:           eg, ['Halpha', 'NII'], ['OIII'], ['Hbeta']
        
        Fit parameters:
        
        flux_arr:           flux in flam of brightests line for each in the set.
        z:                  redshift
        vel_disp:           dispersion of the line [km/s]
        cont_coeff:         coefficients of continuum in 1.e-18 flam
        
        cont_order:         order of the continuum fit
        shape1D:            currently only 'gaussian' is supported

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
        """Set/update arbitrary attribute list with **kwargs"""
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
        
        cont_coeff = _np.array([])
        for i in _six.moves.xrange(self.cont_order+1):
            cont_coeff = _np.append(cont_coeff, params['cont_coeff'+str(i)].value*1.e-18)
        #
        flux_arr = _np.array([])
        for i in _six.moves.xrange(len(self.names_arr)):
            flux_arr = _np.append(flux_arr, params['flux'+str(i)].value)
        
        # Update values
        self.obswave_arr = obswave_arr
        self.z = z
        self.vel_disp = vel_disp
        self.cont_coeff = cont_coeff
        self.flux_arr = flux_arr

        profile = _np.zeros(len(obswave_arr))

        for i in _six.moves.xrange(len(self.names_arr)):
            lineModel = EmissionLines1DModel(name=self.names_arr[i])
            lineModel.make1DProfile(obswave_arr, z, vel_disp, self.flux_arr[i])
            profile += lineModel.profile
            
        # Now add the continuum:
        #cont_profile = _np.zeros(len(obswave_arr))
        for i in _six.moves.xrange(self.cont_order+1):
            profile += cont_coeff[i]*_np.power(obswave_arr,i)
            
        #profile += cont_profile
            
        #self.profile = profile
        return profile
        
    def residual1DProfile(self, params, obswave_arr, data, err, mask):
        # Mask bad data:
        wh_good = _np.where(mask == 1)[0]
        
        model = self.make1DProfile(params, obswave_arr[wh_good])
        resid = (model-data[wh_good])/err[wh_good]
        # mask non-finite elements:
        resid[~_np.isfinite(resid)] = 0.
        
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
        


#
class EmissionLines1DModel(EmissionLines):
    """
    Define an emission line set and profiles for 1D spectra of a single lineset 
        (eg, Ha, OIII doublet, ...)
    
    Input:
        name:               eg, Halpha, OIII
        linenames_arr:      ['HA6565'] or ['OIII5008', 'OIII4960']
        restwave_arr:       can be set from the info in lib of MISFIT
        flux_ratio_arr:     can be set from the info in lib of MISFIT;
                                ratio of lines (1. for strongest); eg [1.], [1., 0.3333]
        flux:               flux in flam of brightests object
        
        z:                  redshift
        vel_disp:           dispersion of the line [km/s]
        
        profile_function:   shape of profile. Currently provies a gaussian if none is set.
                            profile_function input is assumed to be 
                            x (angs), mu (angs), vel_sig (km/s), flux

    """
    
    def __init__(self, profile_function=None, **kwargs):
        
        EmissionLines.__init__(self,None,None,None,None)
        
        self.vel_disp = None    # dispersion of the lines (std dev) -- [km/s]
        self.z = None           # redshift
        self.flux = None        # flux of brightest line
        
        self.obswave_arr = None
        self.profile = None
        
        self.profile_func = profile_function if profile_function else GaussianLine
        
        self.setAttr(**kwargs)

    def setAttr(self,**kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(kwargs)
        
        if (self.linenames_arr is None):
            # set this from lib if not set:
            d = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'lib')
            names_file = _os.path.join(d, 'line_names_cat.dat')
            self.linenames_arr = _io.read_line_names(names_file, name=self.name)
        
        if (self.linenames_arr is not None) & ((self.restwave_arr is None) | \
                (self.flux_ratio_arr is None) ):
            d = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'lib')
            wave_file = _os.path.join(d, 'line_wavelengths_ratios.dat')
            
            wave_arr = []
            line_ratio_arr = []
            for linename in self.linenames_arr:
                wave_arr.append(_io.read_restwave(wave_file, linename=linename))
                line_ratio_arr.append(_io.read_line_ratio(wave_file, linename=linename))
                
            self.restwave_arr = _np.array(wave_arr)
            self.flux_ratio_arr = _np.array(line_ratio_arr)
            
        
        
    def make1DProfile(self, obswave_arr, z, vel_disp, flux):
        """
        Make a 1D profile for the given line set -- excluding any continuum
        """
        
        # Update values
        self.obswave_arr = obswave_arr
        self.z = z
        self.vel_disp = vel_disp
        self.flux = flux
        
        profile = _np.zeros(len(obswave_arr))
        
        for i in _six.moves.xrange(len(self.linenames_arr)):
            profile += self.profile_func(self.obswave_arr, 
                            self.restwave_arr[i]*(1.+self.z), self.vel_disp, self.flux*self.flux_ratio_arr[i])
        
        self.profile = profile
        
    
    
    
    
def GaussianLine(x, mu, vel_sig, flux):
    # Mu: wavelength 
    
    # Convert vel_sig into sig in wavelength:
    sig = vel_sig / c_kms * mu
    
    # make gaussian
    gaus = norm.pdf(x, mu, sig)
    # scale to amplitude
    scale = flux/(sig*_np.sqrt(2.*_np.pi))
    gaus = gaus/gaus.max()*scale
    return gaus
        
        