# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Testing of DYSMALPY model (component) calculations

# import os
# import shutil

import math
import copy

import numpy as np

import misfit

import astropy.constants as _constants
c_cgs = _constants.c.cgs.value
c_kms = c_cgs * 1.e-5 # from cm/s -> km/s
c_AA = c_cgs * 1.e8  # go from cm/s -> AA/s

rt_prefactor_global = 0.4

_mock_galaxy_kin_parameters_defaults = {
    'z':  2.5, 
    'name': 'test', 
    'lam0': 6564.60,   # Halpha
    'line_primary':  'HA6565', 
    'linegroup_name': 'Halpha', 
    'band': 'K', 
    # Model parameters:
    'n': 1., 
    're_arcsec': 0.6, 
    'delt_PA': 0., 
    'q': 0.5, 
    'q0': 0.19, 
    'Va': 200, 
    'rt': 0.4*0.6, 
    'sigma0': 60., 
    'yshift': 0., 
    # INSTRUMENT:
    'instrument_name': 'MOSFIRE', 
    'dlam': 2.1691,   # MOSFIRE K band
    'PSF_type': "Gaussian", 
    'PSF_FWHM': 0.7,    # arcsec
    'pixscale': 0.1799, # arcsec / pix
    'slit_width': 0.7,  # arcsec
    'slit_length': 3.,  # arcsec
    'instrument_resolution': 39., # stddev, km/s
    'yspace_dither_arc': None, 
    # Model settings: 
    'dither': False, 
    'no_continuum_subtraction': True, 
    'nwave': 25, 
    # Mock settings:
    'lamshift': 0., 
    'snr': 15., 
    'nymin': 13, 
    'nSubpixels': 2, 
    'pad_factor': 0.5, 
    'do_position_wave_shift': False, 
    'do_inst_res_conv_effective': False, 
    # Fit settings: 
    'cpuFrac': 0.5, 
    'scale_param_a': 2.0, 
    'weighting_type': 'up-edges', 
    'nWalkers': 2000, 
    'nSteps': 100, 
    'nBurn': 50, 
    'minAF': 0.2, 
    'maxAF': 0.5, 
    'nEff': 10, 
    'overwrite': True, 
    'NoThread': False, 
    'runAllSteps': True, 
}


_mock_galaxy_kin_parameters_2D_specific = {
    're_arcsec': 0.8, 
    'rt': 0.4*0.8, 
}

def _extend_dict(dict1, dict2):
    for key in dict2.keys(): 
        if key not in dict1.keys():
            dict1[key] = dict2[key]

    return dict1


def _setup_2D_output_filenames(gal, alt_folder=None):
    # Output info for the MISFIT plots, data:
    if alt_folder is None:
        raise NotImplementedError
    else:
        path_mcmc = alt_folder+'mcmc_results/'
        
    plot_path_2D_bestfits = path_mcmc+'plots/'
    log_path_2D = path_mcmc+'logs/'
    sampler_path_2D = path_mcmc+'samplers/'
    bestfit_path_2D = path_mcmc+'pickle_bestfits/'
    plot_path_2D_trace = path_mcmc+'plots/'
    plot_path_2D_trace_burnin = path_mcmc+'plots/'
    plot_path_2D_mcmc_param_corner = path_mcmc+'plots/'
    
    misfit.general.io.ensure_dir(plot_path_2D_bestfits)
    misfit.general.io.ensure_dir(log_path_2D)
    misfit.general.io.ensure_dir(sampler_path_2D)
    misfit.general.io.ensure_dir(bestfit_path_2D)
    misfit.general.io.ensure_dir(plot_path_2D_trace)
    misfit.general.io.ensure_dir(plot_path_2D_trace_burnin)
    misfit.general.io.ensure_dir(plot_path_2D_mcmc_param_corner)
    
    namebase = gal.ID
    
    fnames_out_dict = {'filename_log': log_path_2D+namebase+'.mcmc.log', 
                       'filename_sampler': sampler_path_2D+namebase+'.mcmc_sampler.pickle', 
                       'filename_sampler_h5': sampler_path_2D+namebase+'.mcmc_sampler.h5', 
                       'filename_plot_trace_burnin': plot_path_2D_trace_burnin+namebase+'.burnin_trace.png', 
                       'filename_plot_trace': plot_path_2D_trace+namebase+'.trace.png', 
                       'filename_plot_param_corner': plot_path_2D_mcmc_param_corner+\
                                                     namebase+'.mcmc_param_triangle.png', 
                       'filename_plot_bestfit': plot_path_2D_bestfits+namebase+'.best_fit.png', 
                       'filename_bestfit_model': bestfit_path_2D+namebase+'.bestfit_model.pickle', 
    }
    
    return fnames_out_dict


class HelperSetups(object):


    def __init__(self, **kwargs):
        # self.z = 2.5
        # self.name = 'test'
        # self.band = 'K'
        
        params = copy.deepcopy(kwargs)
        params = _extend_dict(params, _mock_galaxy_kin_parameters_defaults)

        self.z = params['z']
        self.name = params['name']
        self.band = params['band']
        


    def setup_galaxy(self, 
                # n=1., re_arcsec=0.6, delt_PA=0., q=0.5,
                # dither=False, 
                generate_model=False, 
                inst=None,
                # no_continuum_subtraction=True, q0 = 0.19
                **kwargs ):
        params = copy.deepcopy(kwargs)
        params = _extend_dict(params, _mock_galaxy_kin_parameters_defaults)

        if inst is None:
            inst = self.setup_instrument()


        ##########################
        # Setup misfit structure: kinematic profile, galaxy info
        # line_primary = 'HA6565'
        # linegroup_name = 'Halpha'

        # # centered on shifted Ha, then extra shift:
        # dlam = 2.1691  # for K band MOSFIRE
        # nwave = 25
        # lam0 = 6564.60


        gal = misfit.Galaxy(z=self.z, ID=self.name, 
                            # n=n, re_arcsec=re_arcsec,
                            # delt_PA=delt_PA, q=q, re_mass_arcsec=re_arcsec, q0=q0,
                            # q_mass=q, dither=dither, 
                            n=params['n'], re_arcsec=params['re_arcsec'],
                            delt_PA=params['delt_PA'], q=params['q'], 
                            re_mass_arcsec=params['re_arcsec'], q0=params['q0'],
                            q_mass=params['q'], dither=params['dither'], 
                            generate_model=generate_model)

        #####
        # wave =  np.linspace(lam0*(1.+gal.z) -dlam*(nwave-1)/2.,
        #                 lam0*(1.+gal.z) +dlam*(nwave-1)/2.,num=nwave )
        # spec2D = misfit.ObsSpectrum2D(band=self.band, wave=wave,
        #                 line_primary=line_primary,linegroup_name=linegroup_name,
        #                 no_continuum_subtraction=no_continuum_subtraction)

        # spec2D.shape = [ int(np.ceil(inst.slit_length/inst.pixscale)), nwave]
        wave =  np.linspace(params['lam0']*(1.+gal.z) - params['dlam']*(params['nwave']-1)/2.,
                            params['lam0']*(1.+gal.z) + params['dlam']*(params['nwave']-1)/2.,
                            num=params['nwave'] ) + params['lamshift'] 
        spec2D = misfit.ObsSpectrum2D(band=self.band, wave=wave,
                        line_primary=params['line_primary'],
                        linegroup_name=params['linegroup_name'],
                        no_continuum_subtraction=params['no_continuum_subtraction'],
                        weighting_type=params['weighting_type'])

        spec2D.shape = [ int(np.ceil(inst.slit_length/inst.pixscale)), params['nwave']]

        spec2D.initialize_lineset()


        spec2D.m0 = np.ceil(inst.slit_length/inst.pixscale)/2.
        spec2D.ypos = spec2D.m0


        gal.set_spectrum_2D(spec2D)

        return gal

    def setup_instrument(self, **kwargs ):
                        #  PSF_type='Gaussian', psf_fwhm=0.7,
                        #  pixscale=0.1799, slit_width=0.7, slit_length=3.,
                        #  instrument_resolution=39.,
                        #  yspace_dither_arc = None):

        params = copy.deepcopy(kwargs)
        params = _extend_dict(params, _mock_galaxy_kin_parameters_defaults)

        # # Instrument settings:
        # slit_width = 0.7
        # slit_length = 3.
        # pixscale = 0.1799  # MOSFIRE
        # band = 'K'
        # yspace_dither_arc = None
        # instrument_resolution = 39.   # Placeholder ~inst dispersion
        #
        #
        # # PSF:
        # PSF_type = 'Gaussian'
        # psf_fwhm = 0.7


        inst = misfit.Spectrograph(**params)
            # instrument_name=params['instrument_name'], 
            #                        PSF_FWHM=psf_fwhm,
            #             PSF_type=PSF_type, yspace_dither_arc=yspace_dither_arc,
            #             pixscale=pixscale,slit_width=slit_width,slit_length=slit_length)
        inst.instrument_resolution = params['instrument_resolution']

        inst.band = self.band

        return inst

    def setup_kinModelOptions(self, **kwargs):
        # do_position_wave_shift=False
        # do_inst_res_conv_effective=False
        # kinModelOptions = misfit.KinModel2DOptions(nSubpixels=2,
        #             pad_factor=0.5,
        #             do_position_wave_shift=do_position_wave_shift,
        #             do_inst_res_conv_effective=do_inst_res_conv_effective)
                
        params = copy.deepcopy(kwargs)
        params = _extend_dict(params, _mock_galaxy_kin_parameters_defaults)

        kinModelOptions = misfit.KinModel2DOptions(**params)

        return kinModelOptions

    def setup_kinProfile(self):

        kinProfile = misfit.KinProfileFiducial()

        return kinProfile

    def setup_intensityProfile(self, gal=None):
        intensityProfile = misfit.IntensityProfileFiducial(galaxy=gal)

        return intensityProfile
    

    def setup_mock(self, gal=None, **kwargs):
        params = copy.deepcopy(kwargs)
        params = _extend_dict(params, _mock_galaxy_kin_parameters_defaults)

        # Remove some duplicate possible kwargs:
        for key in ['z', 'name']:
            _ = params.pop(key)

        # Use new function:
        model = misfit.mock.generate_mock_slit_obs(z=gal.z, ID=gal.ID, 
                                                   wave=gal.spec2D.wave, **params) 
        
        stddev_noise = (model.max()/params['snr'])
    
        noise = np.random.normal(loc=0.0, scale=stddev_noise, size=model.shape)

        mock = model.copy()+noise.copy()
        err = np.ones(model.shape)*stddev_noise

        return mock, err
    

    def setup_theta_bounds_2Dfit(self, fitEmis2D=None):
        # , **kwargs
        # params = copy.deepcopy(kwargs)
        # params = _extend_dict(params, _mock_galaxy_kin_parameters_defaults)

        # # Remove some duplicate possible kwargs:
        # for key in ['z', 'name']:
        #     _ = params.pop(key)

        # Set default boundaries defining the priors
        #V_a_bound_abs = 500.  # old
        Va_bound_abs = ((fitEmis2D.galaxy.spec2D.wave.max()-fitEmis2D.galaxy.spec2D.wave.min()))/\
                            (fitEmis2D.galaxy.spec2D.lam0)*c_kms
        Va_fac = 2.   # V_a initial bound: full V implied by diff of wavelengths from spec stamp.
                    # set full prior to [V_a from initial stamp coverage]* V_a_fac
        Va_bound = np.array([-1.*Va_bound_abs*Va_fac, Va_bound_abs*Va_fac])
        
        sigma0_bound = np.array([0., Va_bound_abs*Va_fac/2.35]) 
        # Assuming whole range = FWHM
        
        r_t_fac = 1.  
        r_t_bound = np.array([0.,  np.shape(fitEmis2D.galaxy.spec2D.flux)[0]*\
                                        fitEmis2D.instrument.pixscale*0.5*r_t_fac])
        # r_t bounds: r_t up to whole width of trimmed 2D emission line image:
        #       r_t from 0 to twice distance to edge: ie r_t = 1*radius to edge.
        
        
        theta_init = np.array([100., 0.5, 50., 0., fitEmis2D.galaxy.z])
        
        theta_bounds = np.array([Va_bound, r_t_bound, sigma0_bound, None, None], dtype=object)
        
        theta_names = ['Va', 'rt', 'sigma0', 'm0_shift', 'z']
        theta_vary = np.array([True, True, True, False, False])
        
        
        ###############
        # Link V_a, r_t when interpreting posteriors IF BOTH ARE FREE:
        if theta_vary[0] & theta_vary[1]:
            theta_linked_posteriors = np.array([[0, 1]])
        else:
            theta_linked_posteriors = None

        thetaSettings = misfit.Theta2DSettingsFiducial(theta=theta_init,
                                theta_vary=theta_vary, theta_bounds=theta_bounds,
                                theta_linked_posteriors=theta_linked_posteriors,
                                theta_names=theta_names)

        return thetaSettings

class TestModels1DDisp:
    helper = HelperSetups()

    def test_AperModel1DDisp(self):
        inst = self.helper.setup_instrument()
        gal = self.helper.setup_galaxy(inst=inst)

        # Use -FWHM to +FWHM for extrac width
        extraction_width = 2. * inst.PSF.PSF_FWHM
        extraction_method = 'optimal'
        nPixels_apermodel = 201

        # Elliptical: dispersion + misalignment in slit
        aperModel1DDisp = misfit.model.AperModel1DDisp(galaxy=gal,
            instrument=inst,extraction_width=extraction_width,
            extraction_method=extraction_method,
            disp_aper_radius_arcsec=gal.re_arcsec*np.sqrt(gal.q),
            nPixels=int(np.round((nPixels_apermodel-1)/2)))
        disp_aper_ratio = aperModel1DDisp.disp_aper_ratio
        print(disp_aper_ratio)

        ftol = 1.e-9
        # This case just uses deltPA = 0, and q=1 (the spherical approx)
        disp_aper_ratio_true = 0.9883303067395043
        assert math.isclose(disp_aper_ratio, disp_aper_ratio_true, rel_tol=ftol)

    def test_AperModel1DDispMisalign(self):
        inst = self.helper.setup_instrument()
        gal = self.helper.setup_galaxy(inst=inst)

        # Use -FWHM to +FWHM for extrac width
        extraction_width = 2. * inst.PSF.PSF_FWHM
        extraction_method = 'optimal'
        nPixels_apermodel = 201

        # Elliptical: dispersion + misalignment in slit
        aperModel1DDispMisalign = misfit.model.AperModel1DDispMisalign(galaxy=gal,
            instrument=inst,extraction_width=extraction_width,
            extraction_method=extraction_method,
            disp_aper_radius_arcsec=gal.re_arcsec*np.sqrt(gal.q),
            nPixels=int(np.round((nPixels_apermodel-1)/2)))
        disp_aper_ratio = aperModel1DDispMisalign.disp_aper_ratio
        print(disp_aper_ratio)

        ftol = 1.e-9
        # disp_aper_ratio_true = 0.9884434977112926 # old method, involving rotation calcs
        disp_aper_ratio_true = 0.9888142642293415
        assert math.isclose(disp_aper_ratio, disp_aper_ratio_true, rel_tol=ftol)


    def test_AperModel1DDispMisalign_deltPA60(self):
        inst = self.helper.setup_instrument()
        gal = self.helper.setup_galaxy(inst=inst, delt_PA=60.)

        # Use -FWHM to +FWHM for extrac width
        extraction_width = 2. * inst.PSF.PSF_FWHM
        extraction_method = 'optimal'
        nPixels_apermodel = 201

        # Elliptical: dispersion + misalignment in slit
        aperModel1DDispMisalign = misfit.model.AperModel1DDispMisalign(galaxy=gal,
            instrument=inst,extraction_width=extraction_width,
            extraction_method=extraction_method,
            disp_aper_radius_arcsec=gal.re_arcsec*np.sqrt(gal.q),
            nPixels=int(np.round((nPixels_apermodel-1)/2)))
        disp_aper_ratio = aperModel1DDispMisalign.disp_aper_ratio
        print(disp_aper_ratio)

        ftol = 1.e-9
        #disp_aper_ratio_true = 0.9892714489739454  # old method, involving rotation calcs
        disp_aper_ratio_true = 0.9903832322294895
        assert math.isclose(disp_aper_ratio, disp_aper_ratio_true, rel_tol=ftol)

    def test_AperModel1DDispMisalign_deltPA60_offset(self):
        inst = self.helper.setup_instrument()
        gal = self.helper.setup_galaxy(inst=inst, delt_PA=60.)

        # Use -FWHM to +FWHM for extrac width
        extraction_width = 2. * inst.PSF.PSF_FWHM
        extraction_method = 'optimal'
        nPixels_apermodel = 201

        xc = 1. # offset of slit in x direction, in ARCSECS

        # Elliptical: dispersion + misalignment in slit
        aperModel1DDispMisalign = misfit.model.AperModel1DDispMisalign(galaxy=gal,
            instrument=inst,extraction_width=extraction_width,
            extraction_method=extraction_method,
            disp_aper_radius_arcsec=gal.re_arcsec*np.sqrt(gal.q),
            nPixels=int(np.round((nPixels_apermodel-1)/2)),
            xc=xc)
        disp_aper_ratio = aperModel1DDispMisalign.disp_aper_ratio
        print(disp_aper_ratio)

        ftol = 1.e-9
        disp_aper_ratio_true = 0.9045425257539351  # xc=0 value: 0.9903832322294895
        assert math.isclose(disp_aper_ratio, disp_aper_ratio_true, rel_tol=ftol)


class TestModels1DRot:
    helper = HelperSetups()

    def test_AperModel1DRot(self):
        inst = self.helper.setup_instrument()
        gal = self.helper.setup_galaxy(inst=inst)
        kinProfile = self.helper.setup_kinProfile()
        kinModelOptions = self.helper.setup_kinModelOptions()
        intensityProfile = self.helper.setup_intensityProfile(gal=gal)

        # Use -FWHM to +FWHM for extrac width
        extraction_width = 2. * inst.PSF.PSF_FWHM
        extraction_method = 'optimal'
        nPixels_apermodel = 201

        # Disk with V/sigma
        Va_dummy = 100.
        rt_factor = rt_prefactor_global/(2.*1. - 0.324)   # do rt = 0.4*R_E/1.676 = 0.4*r_s, for n=1.
        v_to_sig_re = 2.

        theta_1D_model = np.array([v_to_sig_re, Va_dummy, rt_factor*gal.re_arcsec])
        aperModel1DRot = misfit.model.AperModel1DRot(galaxy=gal,
                        instrument=inst,extraction_width=extraction_width,
                        extraction_method=extraction_method,
                        do_position_wave_shift=kinModelOptions.do_position_wave_shift,
                        do_inst_res_conv_effective=kinModelOptions.do_inst_res_conv_effective,
                        theta=theta_1D_model,
                        kinProfile=kinProfile,
                        intensityProfile=intensityProfile,
                        disp_aper_radius_arcsec=gal.re_arcsec,
                        nPixels = nPixels_apermodel)
        disp_aper_ratio = aperModel1DRot.disp_aper_ratio
        print(disp_aper_ratio)


        ftol = 1.e-9
        disp_aper_ratio_true = 0.6346111811716774
        assert math.isclose(disp_aper_ratio, disp_aper_ratio_true, rel_tol=ftol)

    def test_AperModel1DRot_deltPA60(self):
        inst = self.helper.setup_instrument()
        gal = self.helper.setup_galaxy(inst=inst, delt_PA=60.)
        kinProfile = self.helper.setup_kinProfile()
        kinModelOptions = self.helper.setup_kinModelOptions()
        intensityProfile = self.helper.setup_intensityProfile(gal=gal)

        # Use -FWHM to +FWHM for extrac width
        extraction_width = 2. * inst.PSF.PSF_FWHM
        extraction_method = 'optimal'
        nPixels_apermodel = 201

        # Disk with V/sigma
        Va_dummy = 100.
        rt_factor = rt_prefactor_global/(2.*1. - 0.324)   # do rt = 0.4*R_E/1.676 = 0.4*r_s, for n=1.
        v_to_sig_re = 2.

        theta_1D_model = np.array([v_to_sig_re, Va_dummy, rt_factor*gal.re_arcsec])
        aperModel1DRot = misfit.model.AperModel1DRot(galaxy=gal,
                        instrument=inst,extraction_width=extraction_width,
                        extraction_method=extraction_method,
                        do_position_wave_shift=kinModelOptions.do_position_wave_shift,
                        do_inst_res_conv_effective=kinModelOptions.do_inst_res_conv_effective,
                        theta=theta_1D_model,
                        kinProfile=kinProfile,
                        intensityProfile=intensityProfile,
                        disp_aper_radius_arcsec=gal.re_arcsec,
                        nPixels = nPixels_apermodel)
        disp_aper_ratio = aperModel1DRot.disp_aper_ratio
        print(disp_aper_ratio)


        ftol = 1.e-9
        disp_aper_ratio_true = 0.6242592269404217 # 0.6346111811716774
        assert math.isclose(disp_aper_ratio, disp_aper_ratio_true, rel_tol=ftol)



    def test_AperModel1DRot_deltPA90(self):
        inst = self.helper.setup_instrument()
        gal = self.helper.setup_galaxy(inst=inst, delt_PA=90.)
        kinProfile = self.helper.setup_kinProfile()
        kinModelOptions = self.helper.setup_kinModelOptions()
        intensityProfile = self.helper.setup_intensityProfile(gal=gal)

        # Use -FWHM to +FWHM for extrac width
        extraction_width = 2. * inst.PSF.PSF_FWHM
        extraction_method = 'optimal'
        nPixels_apermodel = 201

        # Disk with V/sigma
        Va_dummy = 100.
        rt_factor = rt_prefactor_global/(2.*1. - 0.324)   # do rt = 0.4*R_E/1.676 = 0.4*r_s, for n=1.
        v_to_sig_re = 2.

        theta_1D_model = np.array([v_to_sig_re, Va_dummy, rt_factor*gal.re_arcsec])
        aperModel1DRot = misfit.model.AperModel1DRot(galaxy=gal,
                        instrument=inst,extraction_width=extraction_width,
                        extraction_method=extraction_method,
                        do_position_wave_shift=kinModelOptions.do_position_wave_shift,
                        do_inst_res_conv_effective=kinModelOptions.do_inst_res_conv_effective,
                        theta=theta_1D_model,
                        kinProfile=kinProfile,
                        intensityProfile=intensityProfile,
                        disp_aper_radius_arcsec=gal.re_arcsec,
                        nPixels = nPixels_apermodel)
        disp_aper_ratio = aperModel1DRot.disp_aper_ratio
        print(disp_aper_ratio)


        ftol = 1.e-9
        disp_aper_ratio_true = 0.6193325246886088
        assert math.isclose(disp_aper_ratio, disp_aper_ratio_true, rel_tol=ftol)




    def test_AperModel1DRot_vtosig0(self):
        inst = self.helper.setup_instrument()
        gal = self.helper.setup_galaxy(inst=inst)
        kinProfile = self.helper.setup_kinProfile()
        kinModelOptions = self.helper.setup_kinModelOptions()
        intensityProfile = self.helper.setup_intensityProfile(gal=gal)

        # Use -FWHM to +FWHM for extrac width
        extraction_width = 2. * inst.PSF.PSF_FWHM
        extraction_method = 'optimal'
        nPixels_apermodel = 201

        # Disk with V/sigma
        Va_dummy = 100.
        rt_factor = rt_prefactor_global/(2.*1. - 0.324)   # do rt = 0.4*R_E/1.676 = 0.4*r_s, for n=1.
        v_to_sig_re = 0.

        theta_1D_model = np.array([v_to_sig_re, Va_dummy, rt_factor*gal.re_arcsec])
        aperModel1DRot = misfit.model.AperModel1DRot(galaxy=gal,
                        instrument=inst,extraction_width=extraction_width,
                        extraction_method=extraction_method,
                        do_position_wave_shift=kinModelOptions.do_position_wave_shift,
                        do_inst_res_conv_effective=kinModelOptions.do_inst_res_conv_effective,
                        theta=theta_1D_model,
                        kinProfile=kinProfile,
                        intensityProfile=intensityProfile,
                        disp_aper_radius_arcsec=gal.re_arcsec,
                        nPixels = nPixels_apermodel)
        disp_aper_ratio = aperModel1DRot.disp_aper_ratio
        print(disp_aper_ratio)


        ftol = 1.e-9
        disp_aper_ratio_true = 0.9999999999999999
        assert math.isclose(disp_aper_ratio, disp_aper_ratio_true, rel_tol=ftol)


    def test_AperModel1DRot_vtosig0_deltPA60(self):
        inst = self.helper.setup_instrument()
        gal = self.helper.setup_galaxy(inst=inst, delt_PA=60.)
        kinProfile = self.helper.setup_kinProfile()
        kinModelOptions = self.helper.setup_kinModelOptions()
        intensityProfile = self.helper.setup_intensityProfile(gal=gal)

        # Use -FWHM to +FWHM for extrac width
        extraction_width = 2. * inst.PSF.PSF_FWHM
        extraction_method = 'optimal'
        nPixels_apermodel = 201

        # Disk with V/sigma
        Va_dummy = 100.
        rt_factor = rt_prefactor_global/(2.*1. - 0.324)   # do rt = 0.4*R_E/1.676 = 0.4*r_s, for n=1.
        v_to_sig_re = 0.

        theta_1D_model = np.array([v_to_sig_re, Va_dummy, rt_factor*gal.re_arcsec])
        aperModel1DRot = misfit.model.AperModel1DRot(galaxy=gal,
                        instrument=inst,extraction_width=extraction_width,
                        extraction_method=extraction_method,
                        do_position_wave_shift=kinModelOptions.do_position_wave_shift,
                        do_inst_res_conv_effective=kinModelOptions.do_inst_res_conv_effective,
                        theta=theta_1D_model,
                        kinProfile=kinProfile,
                        intensityProfile=intensityProfile,
                        disp_aper_radius_arcsec=gal.re_arcsec,
                        nPixels = nPixels_apermodel)
        disp_aper_ratio = aperModel1DRot.disp_aper_ratio
        print(disp_aper_ratio)


        ftol = 1.e-9
        disp_aper_ratio_true = 1.0
        assert math.isclose(disp_aper_ratio, disp_aper_ratio_true, rel_tol=ftol)

    def test_AperModel1DRot_puredisk(self):
        inst = self.helper.setup_instrument()
        gal = self.helper.setup_galaxy(inst=inst)
        kinProfile = self.helper.setup_kinProfile()
        kinModelOptions = self.helper.setup_kinModelOptions()
        intensityProfile = self.helper.setup_intensityProfile(gal=gal)

        # Use -FWHM to +FWHM for extrac width
        extraction_width = 2. * inst.PSF.PSF_FWHM
        extraction_method = 'optimal'
        nPixels_apermodel = 201

        # Disk with V/sigma
        Va_dummy = 100.
        rt_factor = rt_prefactor_global/(2.*1. - 0.324)   # do rt = 0.4*R_E/1.676 = 0.4*r_s, for n=1.
        v_to_sig_re = None   # None means pure disk

        theta_1D_model = np.array([v_to_sig_re, Va_dummy, rt_factor*gal.re_arcsec])
        aperModel1DRot = misfit.model.AperModel1DRot(galaxy=gal,
                        instrument=inst,extraction_width=extraction_width,
                        extraction_method=extraction_method,
                        do_position_wave_shift=kinModelOptions.do_position_wave_shift,
                        do_inst_res_conv_effective=kinModelOptions.do_inst_res_conv_effective,
                        theta=theta_1D_model,
                        kinProfile=kinProfile,
                        intensityProfile=intensityProfile,
                        disp_aper_radius_arcsec=gal.re_arcsec,
                        nPixels = nPixels_apermodel)
        disp_aper_ratio = aperModel1DRot.disp_aper_ratio
        print(disp_aper_ratio)


        ftol = 1.e-9
        disp_aper_ratio_true = 0.5034026113213353
        assert math.isclose(disp_aper_ratio, disp_aper_ratio_true, rel_tol=ftol)




class TestModels2DRot:
    helper = HelperSetups()

    def test_FitEmissionLines2D(self, **kwargs):
        params = copy.deepcopy(kwargs)
        # Override values specifically for 2D test:
        for key in _mock_galaxy_kin_parameters_2D_specific:
            params[key] = _mock_galaxy_kin_parameters_2D_specific[key]
            
        params = _extend_dict(params, _mock_galaxy_kin_parameters_defaults)


        inst = self.helper.setup_instrument(**params)
        gal = self.helper.setup_galaxy(inst=inst, **params)


        mock, err = self.helper.setup_mock(gal=gal, **params)

        gal.spec2D.flux = mock
        gal.spec2D.flux_err = err

        fnames_fit = _setup_2D_output_filenames(gal, alt_folder='PYTEST_OUTPUT/')

        mcmcOptions = misfit.MCMC2DOptions(**{**params, **fnames_fit})
        kinModelOptions = self.helper.setup_kinModelOptions(**params)
        kinProfile = self.helper.setup_kinProfile()


        fitEmis2D = misfit.fit.FitEmissionLines2D(galaxy=gal, 
                linegroup_name=gal.spec2D.linegroup_name, 
                instrument=inst, 
                instrument_img=None,
                mcmcOptions=mcmcOptions, 
                kinModelOptions=kinModelOptions,
                kinProfile=kinProfile)

        thetaSettings = self.helper.setup_theta_bounds_2Dfit(fitEmis2D=fitEmis2D)

        # raise ValueError

        fitEmis2D.fit(thetaSettings=thetaSettings)

        # fitEmis2D.mcmc_results.bestfit_theta
        # fitEmis2D.mcmc_results.theta_names

        kinProf = self.helper.setup_kinProfile()
        theta_true = np.array([params['Va'], params['rt'], params['sigma0'], 
                               params['yshift'], gal.z])
        kinProf.update_theta(theta_true)
        Vre_true = kinProf.vel(params['re_arcsec'], 0.)   
        V22_true = kinProf.vel(2.2/1.676* params['re_arcsec'], 0.)  


        # meas_inds = [fitEmis2D.mcmc_results.bestfit_theta[3], 
        #                fitEmis2D.mcmc_results.bestfit_theta[4], 
        #                fitEmis2D.mcmc_results.bestfit_theta[2]]
        meas_inds = [3, 4, 2]
        true_values = [Vre_true, V22_true, params['sigma0']]

        # Must match within uncertainty, this is high SNR:
        for i in range(len(meas_inds)):
            ind = meas_inds[i]
            assert (true_values[i] - (fitEmis2D.mcmc_results.bestfit_theta[ind]-\
                     fitEmis2D.mcmc_results.err_theta_1sig[i][0])> 0) or \
                    (fitEmis2D.mcmc_results.bestfit_theta[ind]+\
                     fitEmis2D.mcmc_results.err_theta_1sig[i][1] - true_values[i] > 0)