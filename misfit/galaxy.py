# misfit/galaxy.py
# General routines to have classes for MISFIT
#
# Copyright 2014, 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function

import numpy as np
# import pandas as pd
import copy
import os
# import sys

try:
    # import general.general_utils as utils
    from general import galaxy_utils
    from general import io
except:
    # from .general import general_utils as utils
    from .general import galaxy_utils
    from .general import io

import astropy.constants as constants
c_cgs = constants.c.cgs.value
c_AA = c_cgs * 1.e8  # go from cm/s -> AA/s


class GalaxyBasic(object):
    """
    Basic galaxy class.
    """
    def __init__(self, **kwargs):

        self.z = None

        # Results of Sersic fitting:
        self.n = None
        self.re_arcsec = None
        self.q = None

        self.re_mass_arcsec = None
        self.q_mass = None

        self.q0 = 0.19          # Assumed galaxy intrinsic thickness

        self.delt_PA = None

        self.generate_model = False  # Set this to 'True' if just generating a model of a galaxy

        self.spec2D = None

        self.setAttr(**kwargs)

    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))

    def copy(self):
        return copy.deepcopy(self)

    def set_spectrum_2D(self, obsSpectrum2DBasic):
       self.spec2D = obsSpectrum2DBasic

       # Ensure few basic things are loaded
       self.spec2D.load_basic_defaults(self)


class Galaxy(GalaxyBasic):
    """
    Galaxy class to contain information about spectra for MISFIT.

    
    Parameters
    ----------
    field : string
        Galaxy field

    maskname : string
        Galaxy mask name (for determining spectra filenames)

    ID : int
        Galaxy ID

    RA : float
        RA of the galaxy, in decimal degrees

    DEC : float
        Dec of the galaxy, in decimal degrees

    z : float
        Redshift

    Av : float 
        Attenuation towards the stellar continuum

    lmass : float 
        log10 of the galaxies' stellar mass

    lage : float
        log10 of the age of the parametric SFH

    ltau : float 
        log10 of the delay time of the parametric SFH
    
    lsfr : float
        log10 SFR of the galaxy

    Av_neb : float
        Nebular attenuation towards the star-forming regions in the galaxy

    sersicRA : float
        RA of the model center from the parametric structural fitting
    
    sersicDEC : float
        Dec of the model center from the parametric structural fitting
    
    n : float 
        Sersic index of the parametric fitting to the imaging.
    
    re_arcsec : float
        Effective radius (projected) of the light distribution from the Sersic fitting 
        to the imaging, in arcsec.

    sersicPA : float
        Position angle of the galaxy major axis, from the parametric fitting, in degrees. 
        Convention is CCW relative to the image y axis (up=0, left=90)
    
    q : float
        Observed axis ratio for the light of the galaxy.    
    
    re_mass_arcsec : float
        Effective radius (projected) of the mass distribution, in arcsec. 

    q_mass : float
        Observed axis ratio for the mass distribution. 

    q0 : float
        Intrinsic galaxy thickness (c/a). Default: 0.19

    deltPA : float
        Angle between the galaxy major axis and the slit, in degrees.

    spec1D : `ObsSpectrum1D` instance
        Galaxy 1D spectrum

    spec2D : `ObsSpectrum2D` instance
        Galaxy 2D spectrum

    pstamp : `Pstamp` instance
        Galaxy image postage stamp
    
    dither : bool, optional
        Specify whether data was dithered (eg, neg images). Default: True

    debug : bool, optional
        Default: False

    
    Methods
    -------
    set_spectrum_1D :
        Set the galaxy object's 1D spectrum to a `ObsSpectrum1D` instance

    set_spectrum_2D :
        Set the galaxy object's 2D spectrum to a `ObsSpectrum2D` instance

    set_pstamp :
        Set the galaxy object's image postage stamp to a `Pstamp` instance

    calculate_slit_object_delta_PA :
        Calculate misalignment of galaxy within slit: store in galaxy

        
    Notes
    -----

    """
    def __init__(self, **kwargs):
        super(Galaxy, self).__init__(**kwargs)
        self.field = None
        self.maskname = None
        self.ID = None

        self.RA = None
        self.DEC = None

        self.z = None

        # Stellar population modeling parameters
        self.Av = None
        self.lmass = None
        self.lage = None
        self.ltau = None

        # Info about star formation, SF regions
        self.lsfr = None
        self.Av_neb = None


        # Results of Sersic fitting:
        self.sersicRA = None
        self.sersicDEC = None
        self.n = None
        self.re_arcsec = None
        self.sersicPA = None       # PA of major axis ; CCW relative to y axis (up = 0deg, left=90deg)
        self.q = None

        self.re_mass_arcsec = None
        self.q_mass = None

        self.q0 = 0.19          # Assumed galaxy intrinsic thickness

        self.delt_PA = None

        self.spec1D = None
        self.spec2D = None
        self.pstamp = None

        self.dither = True      # Specify whether data was dithered (eg, neg images)


        self.debug = False

        self.setAttr(**kwargs)

    def set_spectrum_1D(self, obsSpectrum1D):
       self.spec1D = obsSpectrum1D

       # Ensure few basic things are loaded
       self.spec1D.load_basic_defaults(self)

    def set_spectrum_2D(self, obsSpectrum2D):
       self.spec2D = obsSpectrum2D

       # Ensure few basic things are loaded
       self.spec2D.load_basic_defaults(self)

    def set_pstamp(self, pstamp):
        self.pstamp = pstamp

    def calculate_slit_object_delta_PA(self):
        """Calculate misalignment of galaxy within slit: store in galaxy"""

        self = galaxy_utils.calculate_slit_offset_angles(self)





class Spectrum(object):
    """
    Basic Spectrum class

    Parameters
    ----------
    wave : array-like
        Observed wavelength array

    flux : array-like
        Observed flux array

    flux_err : array-like
        Observed flux uncertainty array

    spec_mask : array-like
        Spectral mask array

    units_wave : string
        Wavelength array units. Default: 'angstroms

    units_flux : string 
        Flux array unit convention. Default: 'flam'

    spec_type : string
        Type of spectrum: either 'wave' / 'velocity'. Default: 'wave'

    """
    def __init__(self, **kwargs):
        self.wave = None
        self.flux = None
        self.flux_err = None
        self.spec_mask = None
        self.units_wave = 'angstroms'
        self.units_flux = 'flam'

        self.spec_type = 'wave'   #  wave / velocity

        self.setAttr(**kwargs)

    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))

    def copy(self):
        return copy.deepcopy(self)



    """
    Galaxy class to contain information about spectra for MISFIT.

    
    Parameters
    ----------
    field : string
        Galaxy field

    maskname : string
        Galaxy mask name (for determining spectra filenames)

    ID : int
        Galaxy ID

    RA : float
        RA of the galaxy, in decimal degrees

    DEC : float
        Dec of the galaxy, in decimal degrees

    z : float
        Redshift

    Av : float 
        Attenuation towards the stellar continuum

    lmass : float 
        log10 of the galaxies' stellar mass

    lage : float
        log10 of the age of the parametric SFH

    ltau : float 
        log10 of the delay time of the parametric SFH
    
    lsfr : float
        log10 SFR of the galaxy

    Av_neb : float
        Nebular attenuation towards the star-forming regions in the galaxy

    sersicRA : float
        RA of the model center from the parametric structural fitting
    
    sersicDEC : float
        Dec of the model center from the parametric structural fitting
    
    n : float 
        Sersic index of the parametric fitting to the imaging.
    
    re_arcsec : float
        Effective radius (projected) of the light distribution from the Sersic fitting 
        to the imaging, in arcsec.

    sersicPA : float
        Position angle of the galaxy major axis, from the parametric fitting, in degrees. 
        Convention is CCW relative to the image y axis (up=0, left=90)
    
    q : float
        Observed axis ratio for the light of the galaxy.    
    
    re_mass_arcsec : float
        Effective radius (projected) of the mass distribution, in arcsec. 

    q_mass : float
        Observed axis ratio for the mass distribution. 

    q0 : float
        Intrinsic galaxy thickness (c/a). Default: 0.19

    deltPA : float
        Angle between the galaxy major axis and the slit, in degrees.

    spec1D : `ObsSpectrum1D` instance
        Galaxy 1D spectrum

    spec2D : `ObsSpectrum2D` instance
        Galaxy 2D spectrum

    pstamp : `Pstamp` instance
        Galaxy image postage stamp
    
    dither : bool, optional
        Specify whether data was dithered (eg, neg images). Default: True

    debug : bool, optional
        Default: False

    
    Methods
    -------
    set_spectrum_1D :
        Set the galaxy object's 1D spectrum to a `ObsSpectrum1D` instance

    set_spectrum_2D :
        Set the galaxy object's 2D spectrum to a `ObsSpectrum2D` instance

    set_pstamp :
        Set the galaxy object's image postage stamp to a `Pstamp` instance

    calculate_slit_object_delta_PA :
        Calculate misalignment of galaxy within slit: store in galaxy

        
    Notes
    -----

    """


class ObsSpectrum(Spectrum):
    """
    Class to hold observed spectra.
    Note self.wave holds the observed wavelength array.


    Parameters
    ----------
    band : string
        Spectrum band name

    slit_PA : float
        Slit PA (parallel to slit) CCW relative to North (North = 0deg, East=90deg)


    Methods
    -------
    load_basic_defaults : 
        Taking galaxy input, calculate the basic defaults

    calculate_restwave : 
        Taking galaxy input, calculate the restframe wavelength array of the galaxy

    """
    def __init__(self, **kwargs):
        super(ObsSpectrum, self).__init__(**kwargs)

        self.band = None

        # Set this within ObsSpectrum:
        self.slit_PA = None        # Slit PA CCW relative to North (North = 0deg, East=90deg)

        self.setAttr(**kwargs)

    def load_basic_defaults(self, galaxy, **kwargs):
        """Set basic attributes after setting attributes"""
        if (galaxy.z is not None) and (self.wave is not None) and (self.spec_type == 'wave'):
            self.calculate_restwave(galaxy)

        if self.spec_mask is None:
            self.spec_mask = np.ones(len(self.wave))

    def calculate_restwave(self, galaxy):
        self.obswave = self.wave
        self.restwave = self.wave/(1.+galaxy.z)


class ObsSpectrum1D(ObsSpectrum):
    """
    Class to hold the observed 1D spectra.

    Parameters
    ----------
    extraction_method : string
        Default: 'optimal'

    extraction_spatial_profile : array-like
        Array characterizing the spatial extraction profile

    units_spatial : string 
        Units of the spatial array profile. Default: 'arcsec'

    trim_obswave_range : array-like
        Array of lower, upper limit values for trimming the observed wavelength 

    trim_restwave_range : array-like
        Array of lower, upper limit values for trimming the restframe wavelength 

    num_mask_edge : int
        Number of points near the spectral edges to mask

    Methods
    -------
    trim_spectrum_wavelength : 
        Method to trim the spectrum in the spectral direction, using either 
        self.trim_obswave_range or self.trim_restwave_range

    mask_edges : 
        Calculate a spectral mask to mask the edges of the spectrum, based on self.num_mask_edge
    
    """
    

    def __init__(self, **kwargs):

        super(ObsSpectrum1D, self).__init__(**kwargs)

        self.extraction_method = 'optimal'    # Change to indicate how the 1D spectrum as extracted
        self.extraction_spatial_profile = None
        self.units_spatial='arcsec'

        self.trim_obswave_range = None
        self.trim_restwave_range = None

        self.num_mask_edge = 10

        self.setAttr(**kwargs)

    def load_basic_defaults(self, galaxy, **kwargs):
        """Set basic attributes after setting attributes"""
        self.setAttr(**kwargs)

        if (galaxy.z is not None) & (self.wave is not None) & (self.spec_type == 'wave'):
            self.calculate_restwave(galaxy)

        if self.spec_mask is None:
            self.spec_mask = np.ones(len(self.wave))

        # Mask missing data:

    def trim_spectrum_wavelength(self, galaxy, trim_restwave_range=None,
                trim_obswave_range=None,
                param_restwave_filename=None, linename=None):
        """
        Method to trim the 1D spectrum to a given wavelength range.

        Notes
        -----

        Either input: 
        trim_restwave_range (and have galaxy.z set) to do in restframe, or
        trim_obsframe_range  to trim in observed frame, or 
        param_filename and linename  to read the set from a parameters file.

        Returns a trimmed copy of self

        """
        return galaxy_utils.trim_spectrum_1D(self, galaxy, trim_restwave_range=trim_restwave_range,
                    trim_obswave_range=trim_obswave_range,
                    param_restwave_filename=param_restwave_filename, linename=linename)

    
    def mask_edges(self):
        self.spec_mask = galaxy_utils.mask_edges_1D(spec=self.flux, spec_err=self.flux_err,
                    spec_mask=self.spec_mask,num_edge=self.num_mask_edge)




class ObsSpectrum2DBasic(ObsSpectrum):
    """
    Class to hold basic 2D spectra.

    Parameters
    ----------
    flux : 2D array-like
        Array holding 2D spectrum flux. Shape: (spatial, spectral). Eg, [ind_spatial, ind_wave]

    flux_err : 2D array-like
        Array holding 2D spectrum flux uncertainty. Shape: (spatial, spectral). Eg, [ind_spatial, ind_wave]

    spec_mask : 2D array-like
        Array holding 2D spectrum mask. Shape: (spatial, spectral). Eg, [ind_spatial, ind_wave]

    spec_weight : 2D array-like
        Array holding 2D spectrum weight. Shape: (spatial, spectral). Eg, [ind_spatial, ind_wave]

    wave : 1D array-like
        Array holding 1D spectrum wavelength array.

    units_wave : string, optional
        Units of the wavelength array. Default: 'angstroms'

    units_flux : string, optional
        Flux array type. Default: 'flam' (eg, erg/s/cm2/Angstrom)

    Methods
    -------
    load_basic_defaults : 
        Load basic default values

    """
        
    def __init__(self, **kwargs):

        super(ObsSpectrum2DBasic, self).__init__(**kwargs)


        self.setAttr(**kwargs)

    def load_basic_defaults(self, galaxy, **kwargs):
        """Set basic attributes after setting attributes"""
        self.setAttr(**kwargs)

        if (galaxy.z is not None) & (self.wave is not None) & (self.spec_type == 'wave'):
            self.calculate_restwave(galaxy)

        if (self.spec_mask is None) & (self.flux is not None):
            self.spec_mask = np.ones(self.flux.shape)




class ObsSpectrum2D(ObsSpectrum2DBasic):
    """
    Class to hold 2D spectra.

    Parameters
    ----------
    flux : 2D array-like
        Array holding 2D spectrum flux. Shape: (spatial, spectral). Eg, [ind_spatial, ind_wave]

    flux_err : 2D array-like
        Array holding 2D spectrum flux uncertainty. Shape: (spatial, spectral). Eg, [ind_spatial, ind_wave]

    spec_mask : 2D array-like
        Array holding 2D spectrum mask. Shape: (spatial, spectral). Eg, [ind_spatial, ind_wave]

    spec_weight : 2D array-like
        Array holding 2D spectrum weight. Shape: (spatial, spectral). Eg, [ind_spatial, ind_wave]

    wave : 1D array-like
        Array holding 1D spectrum wavelength array.

    band : string 
        Spectrum band name. 

    slit_PA : float
        slit PA (parallel to slit) CCW relative to North (North = 0deg, East=90deg). Units: degrees

    linegroup_name : string, optional
        Name describing the line group to be modeled & fit. (eg, 'OIII' to model the OIII doublet)

    linenames_arr : array-like, optional
        Array holding the names of the lines to be fit.

    restwave_arr : array-like, optional
        Array holding the restframe wavelengths of the lines to be fit.

    flux_ratio_arr : array-like, optional
        Array holding the fixed flux ratios of the lines to be fit.

    units_wave : string, optional
        Units of the wavelength array. Default: 'angstroms'

    units_flux : string, optional
        Flux array type. Default: 'flam' (eg, erg/s/cm2/Angstrom)

    Methods
    -------

    Notes
    -----
    To specify the lines to be modeled & fit, either input `linegroup_name`, or 
    explicitly set `linenames_arr`, `restwave_arr`, `flux_ratio_arr`.

    """
        
    def __init__(self, **kwargs):

        # ObsSpectrum.__init__(self,None,None,None, None)
        # #self.setAttr( **kwargs)
        super(ObsSpectrum2D, self).__init__(**kwargs)

        # # Also setup some of the unique parts of 2D spectra:
        self.dlam = None

        # # Also setup some of the unique parts of 2D spectra needed for MISFIT
        self.m0 = None         # Center of spatial profile, in pixel units
        self.m_lims = None     # Bounds of spatial profile, in pixels
        self.lam0 = None       # Central *observed* line wavelength of ref line
        self.ypos = None       # Position of galaxy ; eg center of 1D extraction
        self.yoffset = 0.      # Offset of object postion from center of slit (arcsec)
        self.y_width = None    # Rough extent of 2D spectrum ; eg from 1D extraction

        # Also needs weight array:
        self.spec_weight = None

        # Empty 2D mask array:
        self.spec_mask = None

        self.median_flux_error = None              # Median error level -- to determine skylines
        self.band_cutoff = None                    # Multiplicative factor by median err for skyline cut
        self.band_cutoff_units = 'timesMedianErr'  # Units of mult factor
        self.band_cutoff_param_filename = None
        self.wh_sky = None
        self.wh_nosky = None
        self.wh_sky_fit = None

        self.prepared_for_fitting = False
        self.no_continuum_subtraction = False

        self.row_snr_cut = 2.
        self.low_snr_row_inds = None
        self.mask_qual = None
        self.mask_sky = None
        self.mask_low_snr = None


        # Option for weighting spectrum when fitting:
        ##   Options:  'up-edges', 'none', 'optimal'
        self.weighting_type = 'up-edges'
        self.fitting_weight_profile = None
        self.fitting_weight_matrix = None


        # Lines to be fit:
        # Define fitting ranges or filenames from which to read them
        self.linegroup_name = None
        self.linenames_arr = None
        self.restwave_arr = None
        self.flux_ratio_arr = None

        self.trim_obswave_range = None
        self.trim_restwave_range = None

        self.wave_full_range_rest = None
        self.wave_line_range_rest = None
        self.mask_lines_wave_range_rest = None

        self.linenames_param_filename = None
        self.restwave_param_filename = None
        self.wave_full_range_rest_param_filename = None
        self.wave_line_range_rest_param_filename = None
        self.mask_lines_wave_range_rest_param_filename = None


        self.setAttr(**kwargs)

    def _ensure_mask_weight(self):
        if self.spec_weight is None:
            self.spec_weight = np.ones(self.flux.shape)

        if self.spec_mask is None:
            self.spec_mask = np.ones(self.flux.shape)

    def fit_prep_calcs(self, galaxy, instrument, **kwargs):
        """
        Prepare a copy ObsSpectrum2D class which holds the input spec2D,
        then subtract continuum, mask, and trim spec2D for fitting.
        """
        # Update keywords, etc:
        self.setAttr(**kwargs)

        if not self.prepared_for_fitting:
            self.initialize_lineset()
            self.initialize_skylevels()
            self.initialize_derived_variables(galaxy)

            self._ensure_mask_weight()

            ###########
            # if not self.generate_model:
            #     self.subtract_continuum(galaxy)
            self.subtract_continuum(galaxy, no_cont_subtraction= self.no_continuum_subtraction)

            ###########
            # Trim in wavelength:
            self.trim_spectrum_wavelength(galaxy,
                        trim_restwave_range=self.wave_line_range_rest,
                        param_restwave_filename=self.wave_line_range_rest_param_filename,
                        linename=self.linenames_arr[0], inplace=True)

            ###########
            # Get skyline mask for full line+cont range:
            self.get_skyline_mask(full=True)
            # if not self.generate_model:
            #     self.get_skyline_mask(full=True)
            # else:
            #     self.mask_sky = np.ones(self.flux.shape)
            #     self.wh_sky = []
            #     self.wh_nosky = np.arange(self.flux.shape[1])
            #     self.wh_sky_fit = []


            ###########
            # Get inds for row with low S/N
            self.get_low_snr_mask(debug=galaxy.debug)
            # Combine full mask: missing + sky + low S/N
            self.spec_mask *= self.mask_low_snr


            ###########
            # Trim vertically: identify yposition and bounding pixel indices:
            self.trim_spectrum_spatial(galaxy, instrument)

            ##########
            # Determine weighting for fitting:
            self.calculate_fit_weights()

            ##########
            # Calculate deltPA of galaxy major axis within slit
            if galaxy.delt_PA is None:
                galaxy.calculate_slit_object_delta_PA()

            ##########
            # Calculate m0_shift, lam0 for fitting if they're fixed:
            self.get_m0_lam0_pos(galaxy)

            # Note that this is completed
            self.prepared_for_fitting = True

        # Make sure galaxy is updated:
        galaxy.spec2D = self


    def initialize_derived_variables(self, galaxy):
        """
        Set up derived variables (obs-frame wavelengths)
        """
        self.dlam = np.average(self.wave[1:]-self.wave[:-1])
        # Should have set instrument.instrument_resolution: vel disp [km/s]
        if (self.lam0 is None) & (self.spec_type == 'wave'):
            self.lam0 = self.restwave_arr[0]*(1.+galaxy.z)

    def initialize_lineset(self):
        """
        Initialize the set of lines for fitting. Use either `self.linegroup_name`, 
        or use information from `self.linenames_arr`, `self.restwave_arr`, `self.flux_ratio_arr`.
        """
        # Read in the set of lines from the line group, if not set
        if (self.linenames_arr is None):
            if self.linenames_param_filename is None:
                d = os.path.join(os.path.dirname(__file__), 'lib')
                self.linenames_param_filename  = os.path.join(d, 'line_names_cat.dat')
            self.linenames_arr = io.read_line_names(self.linenames_param_filename,
                            name=self.linegroup_name)

        # Read the wavelength of the target lines:
        if (self.linenames_arr is not None) & (self.restwave_arr is None):
            if self.restwave_param_filename is None:
                d = os.path.join(os.path.dirname(__file__), 'lib')
                self.restwave_param_filename = os.path.join(d, 'line_wavelengths_ratios.dat')
            wave_arr = []
            for linename in self.linenames_arr:
                wave_arr.append(io.read_restwave(self.restwave_param_filename, linename=linename))
            self.restwave_arr = np.array(wave_arr)

        # Read the line flux ratios:
        if (self.linenames_arr is not None) & (self.flux_ratio_arr is None):
            if self.restwave_param_filename is None:
                d = os.path.join(os.path.dirname(__file__), 'lib')
                self.restwave_param_filename = os.path.join(d, 'line_wavelengths_ratios.dat')
            flux_ratio_arr = []
            for linename in self.linenames_arr:
                flux_ratio_arr.append(io.read_line_ratio(self.restwave_param_filename, linename=linename))
            self.flux_ratio_arr = np.array(flux_ratio_arr)


    def initialize_skylevels(self):
        """
        Initialize the skyline threshold levels for the specific band. (MOSFIRE-specificy currently.)
        """
        # Read in band_cutoff, if not set
        if (self.band_cutoff is None):
            if self.band_cutoff_param_filename is None:
                d = os.path.join(os.path.dirname(__file__), 'lib')
                self.band_cutoff_param_filename = os.path.join(d, 'band_sky_cutoff_mosfire.dat')

            self.band_cutoff, self.band_cutoff_units = \
                    io.read_skyline_band_cutoff(self.band_cutoff_param_filename, band=self.band)


    def get_skyline_mask(self, full=True):
        if self.mask_qual is None:
            self.mask_qual = self.spec_mask.copy()
        self.mask_sky, self.wh_sky, self.wh_nosky = galaxy_utils.get_skyline_mask_2D(self, full=full)
        # Don't use wh_sky for fitting:
        self.wh_sky_fit = []

    def get_low_snr_mask(self, debug=False):
        if self.mask_qual is None:
            self.mask_qual = self.spec_mask.copy()
        self.mask_low_snr, self.low_snr_row_inds = \
                            galaxy_utils.get_low_snr_mask_2D(self,
                                                snr_cut=self.row_snr_cut,
                                                debug=debug)

    def subtract_continuum(self, galaxy, no_cont_subtraction=False):
        """
        Subtract continuum from 2D spectrum, fixing slope of every row to that of 1D spectrum.
        Requires wave, flux, flux_err, spec_weight
        """
        self = galaxy_utils.subtract_continuum_2D(self, galaxy,
                    no_cont_subtraction=no_cont_subtraction)

    def trim_spectrum_wavelength(self, galaxy, trim_restwave_range=None,
                trim_obswave_range=None,
                param_restwave_filename=None, linename=None, inplace=False):
        """
        Method to trim the 2D spectrum to a given wavelength range.
        Either input: 
        trim_restwave_range (and have galaxy.z set) to do in restframe, or
        trim_obsframe_range  to trim in observed frame, or 
        param_filename and linename  to read the set from a parameters file.

        Returns a trimmed copy of self
        """
        return galaxy_utils.trim_spectrum_2D_wavelength(self, galaxy,
                    trim_restwave_range=trim_restwave_range,
                    trim_obswave_range=trim_obswave_range,
                    param_restwave_filename=param_restwave_filename,
                    linename=linename, inplace=inplace)

    def trim_spectrum_spatial(self, galaxy, instrument):
        """
        Method to trim the 2D spectrum to a given spatial range -- in preparation for fitting.
        Trim vertically: identify yposition and bounding pixel indices:
        """
        self, galaxy = galaxy_utils.trim_spectrum_2D_spatial(self, galaxy, instrument)

    def calculate_fit_weights(self):
        """
        Determine weighting for fitting -- eg, upweight edges if weighting_type='up-edges'
        """
        self = galaxy_utils.calculate_2D_fit_weights(self)

    def get_m0_lam0_pos(self, galaxy):
        """
        Determine m0_shift, lam0 for fitting -- exact position, not rounded.
        """
        self = galaxy_utils.get_m0_lam0_pos_2D(self, galaxy)

    def fit_emission_y_profile(self, galaxy, instrument, filename_plot=None, plot=False):
        if not self.prepared_for_fitting:
            raise ValueError("Spectrum must have continuum subtracted first, etc")

        self = galaxy_utils.fit_emission_y_profile(self, galaxy, instrument, filename_plot=filename_plot, plot=plot)



class Pstamp(object):
    """
    Class to hold basic postage stamp image. 

    Parameters
    ----------
    img_PA : float
        PA of image (y axis) CCW relative to North (North = 0deg, East=90deg). Unit: degrees

    Methods
    -------

    """
    def __init__(self, **kwargs):
        self.pstamp = None
        self.pstamp_hdr = None

        # Set this within Pstamp:
        self.img_PA = None     # PA of image CCW relative to North (North = 0deg, East=90deg)


        self.setAttr(**kwargs)

    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with kwargs"""
        #self.__dict__.update(kwargs)

        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))

    # def rotate_pstamp(self):
    #     """Implement eventually"""
    #     pass
    #
    # def convolve_FWHM(self):
    #     """Implement eventually"""
    #     pass
    #
    # def downsample_pixscale(self, galaxy, instr):
    #     """Implement eventually"""
    #
    #     pass
    #
