# misfit/instrument.py
# General routines to have classes for MISFIT describing the instrument
# 
# Copyright 2014, 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

# Hidden modules prepended with '_'
from __future__ import print_function

import numpy as _np
from numba import jit
import copy as _copy


class PSFBase(object):
    """
    Basic PSF class
    """
    
    def __init__(self, **kwargs):
        self.PSF_type = 'plain' # name
        self.PSF_FWHM = None  # in arcsec
        self.yspace_dither_arc = None #2.7  # arcsec dither separation from reduction -- set to None for no neg images
        
        self._conv_stamp = None
        
        self.setAttr(**kwargs)
    
    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        #self.__dict__.update(kwargs)
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))
        
    def set_conv_stamp(self, conv_stamp):
        self._conv_stamp = conv_stamp
        
    def generate_conv_stamp(self, x, y, xycent, PSF_FWHM=None):
        """ Define for each subtype """
        pass

class PSFGaussian(PSFBase):
    def __init__(self, **kwargs):
        super(PSFGaussian, self).__init__(**kwargs)
        
        self.PSF_type = 'Gaussian'
        
        self.setAttr(**kwargs)
        
    def generate_conv_stamp(self, x, y, PSF_FWHM=None):
        # x, y should already be *CENTERED*, and in arcsec.
        if PSF_FWHM is None:
            PSF_FWHM = self.PSF_FWHM
            
        return PSF_stamp_gauss(x, y, PSF_FWHM=PSF_FWHM, yspace_dither_arc=self.yspace_dither_arc)
        
class PSFMoffat(PSFBase):
    def __init__(self, PSF_beta=2.5, **kwargs):
        super(PSFMoffat, self).__init__(**kwargs)
        
        self.PSF_type = 'Moffat'
        self.PSF_beta = PSF_beta
        
        self.setAttr(**kwargs)
        
    def generate_conv_stamp(self, x, y, PSF_FWHM=None):    
        # x, y should already be *CENTERED*, and in arcsec.
        if PSF_FWHM is None:
            PSF_FWHM = self.PSF_FWHM
        beta = self.PSF_beta
        alpha = PSF_FWHM/(2. * _np.sqrt(_np.power(2., 1./beta) -1))
        
        return PSF_stamp_moffat(x, y, alpha=alpha, beta=beta)
        
        
class Instrument(object):
    """
    Class to describe the instrument for MISFIT
    """
    def __init__(self, **kwargs):
        
        self.instrument_name = None
        self.band = None
        
        self.pixscale = None    # In arcsec/pixel
        
        if kwargs['PSF_type'].strip().lower() == 'Gaussian'.lower():
            self.PSF = PSFGaussian(**kwargs)
        elif kwargs['PSF_type'].strip().lower() == 'Moffat'.lower():
            self.PSF = PSFMoffat(**kwargs)
        else:
            # Assume Gaussian:
            self.PSF = PSFGaussian(**kwargs)
        
        # self.PSF_FWHM = None    # in arcsec
        # self.PSF_type = 'Gaussian'  # option: 'Gaussian' / 'Moffat'
        # self.PSF_beta = None        # must be not none if Moffat
        
        self.setAttr(**kwargs)

    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        #self.__dict__.update(kwargs)
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))
        
    def copy(self):
        return _copy.deepcopy(self)
        

class Spectrograph(Instrument):
    """
    Class to describe the spectrograph for MISFIT, including slit properties
    """
    def __init__(self, **kwargs):
        super(Spectrograph, self).__init__(**kwargs)
    
        self.slit_width = None       # arcsec
        self.slit_length = None      # arcsec
        self.instrument_resolution = None # velocity *dispersion* instrument resolution [km/s]
                                          # measure from skylines, or set for simulation.
        
        self.conv_sigma = None       # To convolve other, higher-res instrument to match this resolution.
        
        
        self.slit_mask = None
        
        self.setAttr(**kwargs)
        
        
class Imager(Instrument):
    #
    """
    Class to describe the spectrograph for MISFIT, including slit properties
    """
    def __init__(self, **kwargs):
        super(Imager, self).__init__(**kwargs)
        
        
        self.setAttr(**kwargs)
        
        
# TEST 2019.11.14
@jit
def PSF_stamp_gauss(x, y, PSF_FWHM=None, yspace_dither_arc=None):
    # x, y should already be *CENTERED*, and in arcsec.
    PSF_sigma = PSF_FWHM/(2.*_np.sqrt(2.*_np.log(2.)))  # FWHM -> sigma
    
    xarr_flat = _np.tile(x, (len(y),1))
    yarr_flat = _np.tile(_np.array([y]).T, (1,len(x)))
    
    conv_stamp = _np.exp(- (xarr_flat**2 + yarr_flat**2)/(2.*(PSF_sigma**2)))
    
    if yspace_dither_arc is not None:
        conv_stamp_negl = _np.exp(- (xarr_flat**2 + (yarr_flat+yspace_dither_arc)**2)/(2.*(PSF_sigma**2)))
        conv_stamp_negu = _np.exp(- (xarr_flat**2 + (yarr_flat-yspace_dither_arc)**2)/(2.*(PSF_sigma**2)))
        conv_stamp -= 0.5*(conv_stamp_negl + conv_stamp_negu)
    
    
    return conv_stamp
    
# TEST 2019.11.14
@jit
def PSF_stamp_moffat(x, y, alpha=None, beta=None, yspace_dither_arc=None):
    # x, y should already be *CENTERED*, and in arcsec.

    xarr_flat = _np.tile(x, (len(y),1))
    yarr_flat = _np.tile(_np.array([y]).T, (1,len(x)))

    conv_stamp = (beta-1)/(_np.pi * alpha**2) * \
            _np.power((1.+(xarr_flat **2 + yarr_flat**2)/(alpha**2)), -beta)
            
    if yspace_dither_arc is not None:
        conv_stamp_negl = (beta-1)/(_np.pi * alpha**2) * \
                _np.power((1.+(xarr_flat **2 + (yarr_flat+yspace_dither_arc)**2)/(alpha**2)), -beta)
        conv_stamp_negu = (beta-1)/(_np.pi * alpha**2) * \
                _np.power((1.+(xarr_flat **2 + (yarr_flat-yspace_dither_arc)**2)/(alpha**2)), -beta)
        conv_stamp -= 0.5*(conv_stamp_negl + conv_stamp_negu)
    
    return conv_stamp