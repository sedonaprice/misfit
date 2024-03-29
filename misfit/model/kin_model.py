# Copyright 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function

import numpy as np
import copy
# import six

from misfit.model.kin_classes import KinProfileFiducial, IntensityProfileFiducial, \
                                Theta2DSettingsFiducial
from misfit.model import AperModel2D
from misfit import Galaxy

#from numba import jit
##from numba import njit

try:
    import kin_functions as kfuncs
except:
    from . import kin_functions as kfuncs


# Options for model:
class KinModel2DOptions(object):
    """
    Class to hold model options for KinModel2D
    """
    def __init__(self, nSubpixels=2, pad_factor=0.5,
            do_position_wave_shift=False, do_inst_res_conv_effective=False,
            absvalsigma=False, adaptive_upsample_wave=True, adaptive_upsample_factor=3.,
            sigma_floor=True, **kwargs):

        self.nSubpixels = nSubpixels
        self.pad_factor = pad_factor
        self.do_position_wave_shift = do_position_wave_shift
        self.do_inst_res_conv_effective = do_inst_res_conv_effective

        # Options for handling v. small dispersion calculations:
        self.absvalsigma = absvalsigma
        self.adaptive_upsample_wave = adaptive_upsample_wave
        self.adaptive_upsample_factor = adaptive_upsample_factor
        self.sigma_floor = sigma_floor

        self.setAttr(**kwargs)

    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))

    def copy(self):
        return copy.deepcopy(self)



# Kinematic 2D model:
class KinModel2D(object):
    """
    Class to handle kinematic fitting models.
    Handles variable + fixed parameters, talks to the Aper2DModel class,
    creates scaled model to data.
    """
    def __init__(self, galaxy, thetaSettings=Theta2DSettingsFiducial(),
                    kinProfile=KinProfileFiducial(),
                    intensityProfile=IntensityProfileFiducial(Galaxy(n=1.,q0=0.19,re_arcsec=0.)),
                    kinModelOptions = KinModel2DOptions(),
                    do_position_wave_shift=False,
                    do_inst_res_conv_effective=False, **kwargs):
        # Galaxy()
        # Unpack thetaSettings:
        self.theta = thetaSettings.theta
        self.theta_vary = thetaSettings.theta_vary
        self.theta_names = thetaSettings.theta_names
        self.theta_names_nice = thetaSettings.theta_names_nice
        self.theta_init = None

        # self.kinProfile = KinProfileFiducial(theta=self.theta)          # Default
        # self.intensityProfile = IntensityProfileFiducial(galaxy=galaxy) # Default

        self.kinProfile = kinProfile
        self.intensityProfile = intensityProfile

        self.kinProfile.update_theta(theta=self.theta)
        self.intensityProfile.update_values(galaxy=galaxy)

        self.spec_type = 'wave'   # wave / velocity

        self.do_position_wave_shift = do_position_wave_shift
        self.do_inst_res_conv_effective = do_inst_res_conv_effective

        self.no_scale = False    # Set this to true for no scaling (eg, just generate models)

        self.kinModelOptions = kinModelOptions

        self.model = None
        self.bestfit = None

        self.aperModel = None

        self.debug = False

        self.setAttr(**kwargs)

    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))


        # Save input theta:
        if self.theta_init is None:
            try:
                self.theta_init = self.theta.copy()
            except:
                self.theta_init = self.theta

        # Count number of free variables:
        wh_free = np.where(self.theta_vary)[0]
        self.n_free_param = len(wh_free)

        # Make sure profiles are updated:
        self.update_profiles()

    def copy(self):
        return copy.deepcopy(self)

    def update_parameters(self, theta_fitting):
        """
        Unpack current fitting parameters into the whole parameter array
        """
        j = 0
        for i in range(len(self.theta)):
            if self.theta_vary[i]:
                self.theta[i] = theta_fitting[j]
                j += 1
        #self.update_profiles()

    def update_profiles(self):
        self.kinProfile.update_theta(self.theta)
        # self.intensityProfile.update_theta(self.theta)

    def scale_model(self, galaxy):
        # Use the galaxy spectrum to scale the model:
        wh_no_sky = galaxy.spec2D.wh_nosky.copy()

        num_arr = np.sum(galaxy.spec2D.flux[:,wh_no_sky]*self.model[:,wh_no_sky]/\
                            galaxy.spec2D.flux_err[:,wh_no_sky]**2, axis=1)
        den_arr = np.sum(self.model[:,wh_no_sky]**2/galaxy.spec2D.flux_err[:,wh_no_sky]**2, axis=1)

        num_arr[~np.isfinite(num_arr)] = 0.
        num_arr[np.abs(den_arr) == 0.] = 0.
        den_arr[np.abs(den_arr) == 0.] = -99.

        ratio_arr = num_arr/den_arr
        # Tile it: column over rows: like nY
        ratio_tile = np.tile(np.array([ratio_arr]).T, (1, self.model.shape[1]))

        self.model *= ratio_tile


    def make_model(self, galaxy=None, instrument=None, theta_fitting=None):
        # Check that the input matches what is already setup:
        if len(theta_fitting) != self.n_free_param:
            raise ValueError("Number of fitting parameters does not match initialized number of free parameters")

        self.update_parameters(theta_fitting)
        self.update_profiles()  # Needed to update kinProfile theta values for calculations

        if self.aperModel is None:
            self.aperModel = AperModel2D(galaxy=galaxy, instrument=instrument,
                                theta=self.theta,
                                spec_type = self.spec_type,
                                kinProfile=self.kinProfile,
                                intensityProfile=self.intensityProfile,
                                nSubpixels=self.kinModelOptions.nSubpixels,
                                pad_factor=self.kinModelOptions.pad_factor,
                                do_position_wave_shift=self.kinModelOptions.do_position_wave_shift,
                                do_inst_res_conv_effective=self.kinModelOptions.do_inst_res_conv_effective,
                                absvalsigma=self.kinModelOptions.absvalsigma,
                                adaptive_upsample_wave=self.kinModelOptions.adaptive_upsample_wave,
                                adaptive_upsample_factor=self.kinModelOptions.adaptive_upsample_factor,
                                sigma_floor=self.kinModelOptions.sigma_floor)#,
        else:
            # self.aperModel.update_model(theta=self.theta,
            #                     kinProfile=self.kinProfile)

            self.aperModel.update_model(theta=self.theta)

        # # Save the 2D model to base part of class
        self.model = self.aperModel.model.copy()
        if not self.no_scale:
            self.scale_model(self.aperModel.galaxy)

    def update_model(self, theta_fitting=None):
        # Check that the input matches what is already setup:
        if len(theta_fitting) != self.n_free_param:
            raise ValueError("Number of fitting parameters does not match initialized number of free parameters")

        self.update_parameters(theta_fitting)       # expand theta_fitting into full model theta
        #self.aperModel.kinProfile.update_theta(self.theta)

        # self.aperModel.update_model(theta=self.theta,
        #                         kinProfile=self.aperModel.kinProfile)
        self.aperModel.update_model(theta=self.theta)

        # # Save the 2D model to base part of class
        self.model = self.aperModel.model.copy()
        if not self.no_scale:
            self.scale_model(self.aperModel.galaxy)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def make_3D_int_model(self, galaxy=None, instrument=None, theta_fitting=None):
        # For illustrative purposes / toy-model use for mock analysis of IFU data:
        # Note: Has no normalization applied. For the user to deal with later.

        # Check that the input matches what is already setup:
        if len(theta_fitting) != self.n_free_param:
            raise ValueError("Number of fitting parameters does not match initialized number of free parameters")

        self.update_parameters(theta_fitting)
        self.update_profiles()  # Needed to update kinProfile theta values for calculations

        if self.aperModel is None:
            # Change the "slit width" to match the y dim for the IFU cube test case:
            if not galaxy.generate_model:
                n_pix_y_whole = galaxy.spec2D.flux.shape[0]
            else:
                n_pix_y_whole = galaxy.spec2D.shape[0]
            instrument.slit_width = n_pix_y_whole*instrument.pixscale
            n_pix_x_whole = n_pix_y_whole

            self.aperModel = AperModel2D(galaxy=galaxy, instrument=instrument,
                                theta=self.theta,
                                spec_type = self.spec_type,
                                kinProfile=self.kinProfile,
                                intensityProfile=self.intensityProfile,
                                nSubpixels=self.kinModelOptions.nSubpixels,
                                pad_factor=self.kinModelOptions.pad_factor,
                                do_position_wave_shift=self.kinModelOptions.do_position_wave_shift,
                                do_inst_res_conv_effective=self.kinModelOptions.do_inst_res_conv_effective,
                                absvalsigma=self.kinModelOptions.absvalsigma,
                                adaptive_upsample_wave=self.kinModelOptions.adaptive_upsample_wave,
                                sigma_floor=self.kinModelOptions.sigma_floor,
                                debug=self.debug)#,
        else:
            # self.aperModel.update_model(theta=self.theta,
            #                     kinProfile=self.kinProfile)
            self.aperModel.update_model(theta=self.theta)

        ###################
        # Do downsampling:
        # # Save the 2D model to base part of class

        # Shape: nSpec, nY, nX
        nWave = len(galaxy.spec2D.wave.copy())
        # if self.debug:
        #     print("spec cube conv shape={}".format(self.aperModel.spectra_cube_conv.shape))
        #     print("n_pix_y_whole={}, padY={}, n_pix_x_whole={}, padX={}".format(n_pix_y_whole,
        #             self.aperModel.padY, n_pix_x_whole, self.aperModel.padX))
        self.model_cube = kfuncs.rebin(self.aperModel.spectra_cube_conv.copy(),
                    nWave, n_pix_y_whole+2*self.aperModel.padY, n_pix_x_whole+2*self.aperModel.padX)

        if self.debug:
            # Downsample others too
            self.I_wide = kfuncs.rebin(self.aperModel.I_wide.copy(),
                        n_pix_y_whole, n_pix_y_whole, n_pix_x_whole)
            #
            self.V_wide = kfuncs.rebin(self.aperModel.V_wide.copy(),
                        n_pix_y_whole, n_pix_y_whole, n_pix_x_whole)
            #
            self.I_Vsq_wide = kfuncs.rebin(self.aperModel.I_Vsq_wide.copy(),
                        n_pix_y_whole, n_pix_y_whole, n_pix_x_whole)

    def update_3D_int_model(self, theta_fitting=None):
        # For illustrative purposes / toy-model use for mock analysis of IFU data:
        # Note: Has no normalization applied. For the user to deal with later.

        # Check that the input matches what is already setup:
        if len(theta_fitting) != self.n_free_param:
            raise ValueError("Number of fitting parameters does not match initialized number of free parameters")

        self.update_parameters(theta_fitting)
        # self.aperModel.kinProfile.update_theta(self.theta)
        #
        # self.aperModel.update_model(theta=self.theta,
        #                         kinProfile=self.aperModel.kinProfile)
        self.aperModel.update_model(theta=self.theta)

        # # Save the 2D model to base part of class
        self.model_cube = self.aperModel.spectra_cube.copy()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    def model_llike(self):
        return _model_llike(self)

    def model_chisq(self):
        return _model_chisq(self)


    def model_redchisq(self):
        chisq = _model_chisq(self)

        wh = np.where(self.aperModel.galaxy.spec2D.spec_mask == 1)

        nFree = self.n_free_param
        nPix = len(wh[0])

        return chisq/float(nPix-nFree)

    def model_redchisq_weight_ref(self):
        # Eg, take an array of all ones and weight it.
        chisq_ref_allones = self.aperModel.galaxy.spec2D.fitting_weight_matrix.sum()

        wh = np.where(self.aperModel.galaxy.spec2D.spec_mask == 1)

        nFree = self.n_free_param
        nPix = len(wh[0])

        return chisq_ref_allones/float(nPix-nFree)

    def model_redchisq_no_weight(self):
        chisq = _model_chisq_no_weight(self)

        wh = np.where(self.aperModel.galaxy.spec2D.spec_mask == 1)

        nFree = self.n_free_param
        nPix = len(wh[0])

        return chisq/float(nPix-nFree)


#
#@jit
def _model_llike(kinModel, model=None):
    if model is None:
        model = kinModel.model

    chisq_arr_raw_modified = kinModel.aperModel.galaxy.spec2D.spec_mask * \
            ( kinModel.aperModel.galaxy.spec2D.fitting_weight_matrix * \
                 ((((kinModel.aperModel.galaxy.spec2D.flux-model))/\
                kinModel.aperModel.galaxy.spec2D.flux_err)**2.) + \
            np.log10(2.*np.pi*kinModel.aperModel.galaxy.spec2D.flux_err**2) )

    # chisq_arr_raw_modified = kinModel.aperModel.galaxy.spec2D.spec_mask * \
    #         ( ((((kinModel.aperModel.galaxy.spec2D.flux-model))/\
    #             kinModel.aperModel.galaxy.spec2D.flux_err)**2.) + \
    #         np.log10(2.*np.pi*kinModel.aperModel.galaxy.spec2D.flux_err**2) )
    return -0.5*chisq_arr_raw_modified.sum()

#
#@jit
def _model_chisq(kinModel, model=None):
    """
    Function to return the chisq of the residual between the
    2D spectrum amd model

    Parameters
    ----------

    kinModel : `KinModel2D` instance
        Kinematic model instance, holding a `spec2D` instance within 
        `kinModel.aperModel.galaxy.spec2D`, which contains the 
        trimmed 2D spectrum and uncertainty, mask, and weighting. 
        Also contains a model 2D spectrum within `kinModel.model`, 
        if `model` is not passed.


    model : 2D array-like, optional
        Model 2D emission line. 

    Returns
    -------
    chisq : float
        Weighted chisq statistic for goodness of fit of the model.

    """
    # Data:  fitEmis2D.galaxy.spec2D.flux, flux_err, spec_mask, fitting_weight_matrix
    # Model: fitEmis2D.kinModel.model

    ##################################

    # Mask spectrum, do diff with model, divide error, and multiply by fitting weight matrix
    #       (eg, upweighting edges)
    if model is None:
        model = kinModel.model

    chisq_arr_raw = ((((kinModel.aperModel.galaxy.spec2D.flux-model)*\
                kinModel.aperModel.galaxy.spec2D.spec_mask)/kinModel.aperModel.galaxy.spec2D.flux_err)**2.)*\
                kinModel.aperModel.galaxy.spec2D.fitting_weight_matrix

    chisq = chisq_arr_raw.sum()

    return chisq
#
#@jit
def _model_chisq_no_weight(kinModel, model=None):
    """
    Function to return the chisq of the residual between the
    2D spectrum amd model, without any weighting

    Parameters
    ----------

    kinModel : `KinModel2D` instance
        Kinematic model instance, holding a `spec2D` instance within 
        `kinModel.aperModel.galaxy.spec2D`, which contains the 
        trimmed 2D spectrum and uncertainty, and mask.  
        Also contains a model 2D spectrum within `kinModel.model`, 
        if `model` is not passed.


    model : 2D array-like, optional
        Model 2D emission line. 

    Returns
    -------
    chisq : float
        Unweighted chisq statistic for goodness of fit of the model.

    """
    # Data:  fitEmis2D.galaxy.spec2D.flux, flux_err, spec_mask, fitting_weight_matrix
    # Model: fitEmis2D.kinModel.model

    ##################################

    # Mask spectrum, do diff with model, divide error, and multiply by fitting weight matrix
    #       (eg, upweighting edges)
    if model is None:
        model = kinModel.model

    chisq_arr_raw = ((((kinModel.aperModel.galaxy.spec2D.flux-model)*\
                kinModel.aperModel.galaxy.spec2D.spec_mask)/kinModel.aperModel.galaxy.spec2D.flux_err)**2.)

    chisq = chisq_arr_raw.sum()

    return chisq
