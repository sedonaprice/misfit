# Copyright 2018 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

from __future__ import print_function

import numpy as _np

# from misfit import Galaxy, ObsSpectrum2D, Spectrograph, \
#         IntensityProfileFiducial, KinProfileFiducial, \
#         Theta2DSettingsFiducial, KinModel2D, KinModel2DOptions
#
from ..galaxy import Galaxy, ObsSpectrum2D
from ..instrument import Spectrograph
from ..model.kin_model import KinModel2DOptions, KinModel2D
from ..model.kin_classes import Theta2DSettingsFiducial, \
                        KinProfileFiducial, IntensityProfileFiducial

def generate_mock_basis(z=None, 
        ID=None, 
        Va = None, 
        rt = None, 
        sigma0 = None, 
        yshift=None, 
        n=None, re_arcsec=None, q = None, 
        q0 = 0.19, 
        delt_PA = None, 
        dither = False, 
        wave=None,
        instrument_name=None, 
        pixscale=None,
        instrument_resolution = None,
        slit_width=None,
        slit_length=None,
        nymin = 13, 
        PSF_FWHM=None, 
        PSF_type=None, 
        band = None, 
        line_primary = None, 
        linegroup_name = None,
        spec_type = 'wave', 
        nSubpixels=2,
        pad_factor=0.5, 
        do_position_wave_shift=False,
        do_inst_res_conv_effective=False,
        kinProfile = KinProfileFiducial(), 
        intensityProfile = IntensityProfileFiducial(Galaxy(n=1.,q0=0.19,re_arcsec=0.)), 
        debug=False):
        
    # Enforce min slit-length
    if slit_length < nymin*pixscale:
        slit_length = nymin*pixscale
    
    instrument = Spectrograph(instrument_name=instrument_name, PSF_FWHM=PSF_FWHM, 
                    PSF_type = PSF_type, yspace_dither_arc=None, 
                    pixscale=pixscale,slit_width=slit_width,slit_length=slit_length)
    instrument.instrument_resolution = instrument_resolution
    instrument.band = band  
    
    
    nwave = len(wave)
    galaxy = Galaxy(z=z, ID=ID, 
                n=n, re_arcsec=re_arcsec, 
                delt_PA=delt_PA, q=q, q0=q0, 
                dither=False, generate_model=True)
    spec2D = ObsSpectrum2D(band=band, wave=wave, 
                    line_primary=line_primary,linegroup_name=linegroup_name,
                    no_continuum_subtraction=True,
                    spec_type=spec_type)
    
    spec2D.shape = [ _np.int(_np.ceil(instrument.slit_length/instrument.pixscale)), nwave]
    if spec_type == 'wave':
        spec2D.initialize_lineset()
    spec2D.m0 = _np.ceil(instrument.slit_length/instrument.pixscale)/2. - 0.5
    spec2D.ypos = spec2D.m0
    galaxy.set_spectrum_2D(spec2D)
    
    galaxy.namebase = str(ID)
    
    
    intensityProfile.update_values(galaxy=galaxy)
    
    ###############
    # Fiducial settings
    kinModelOptions = KinModel2DOptions(nSubpixels=nSubpixels,
                        pad_factor=pad_factor, 
                        do_position_wave_shift=do_position_wave_shift,
                        do_inst_res_conv_effective=do_inst_res_conv_effective)
    
    # Uses the fiducial curve: arctan
    
    ###############
    # Definite initial fitting parameters:
    # Order: [V_a, r_t, sigma0, m0_shift, z]
    
    # Start with blank choice
    theta_init = _np.array([Va, rt, sigma0, yshift, galaxy.z])
    theta_vary = _np.array([False, False, False, False, False])
    theta_bounds = _np.array([None, None, None, None, None])
    theta_linked_posteriors = None
    
    thetaSettings = Theta2DSettingsFiducial(theta=theta_init,
                            theta_vary=theta_vary, theta_bounds=theta_bounds,
                            theta_linked_posteriors=theta_linked_posteriors)
    
    ## Initialize model handling class:
    kinModel = KinModel2D(galaxy, 
                    thetaSettings=thetaSettings, 
                    kinProfile=kinProfile, 
                    intensityProfile=intensityProfile,
                    kinModelOptions=kinModelOptions,
                    spec_type=spec_type, 
                    no_scale=True,
                    debug=debug)
    
    return kinModel, galaxy, instrument



def generate_mock_IFU_cube(debug=False, **kwargs):
    kinModel, galaxy, instrument = generate_mock_basis(debug=debug, **kwargs)
    kinModel.make_3D_int_model(galaxy=galaxy, instrument=instrument, 
                 theta_fitting=[])
    if debug:
        # model_cube, model_int, model_vel, model_int_velsq
        return kinModel.model_cube, kinModel.I_wide, kinModel.V_wide, \
                kinModel.I_Vsq_wide
    else:
        return kinModel.model_cube


def generate_mock_slit_obs(**kwargs):
    
    kinModel, galaxy, instrument = generate_mock_basis(**kwargs)
    kinModel.make_model(galaxy=galaxy, instrument=instrument, 
                 theta_fitting=[])
    model_img = kinModel.model
    
    return model_img
