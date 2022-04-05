# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Testing of DYSMALPY model (component) calculations

import os
import shutil

import math

import numpy as np


import misfit


rt_prefactor_global = 0.4


class HelperSetups(object):


    def __init__(self):
        self.z = 2.5
        self.name = 'test'
        self.band = 'K'


    def setup_galaxy(self, n=1., re_arcsec=0.6, delt_PA=0., q=0.5,
                dither=False, generate_model=False, inst=None,
                q0 = 0.19):


        if inst is None:
            inst = self.helper.setup_instrument()


        ##########################
        # Setup misfit structure: kinematic profile, galaxy info
        extraction_method = 'optimal'

        line_primary = 'HA6565'
        linegroup_name = 'Halpha'



        gal = misfit.Galaxy(z=self.z, ID=self.name, n=n, re_arcsec=re_arcsec,
                    delt_PA=delt_PA, q=q, re_mass_arcsec=re_arcsec, q0=q0,
                    q_mass=q, dither=dither, generate_model=generate_model)

        #####

        # centered on shifted Ha, then extra shift:
        dlam = 2.1691  # for K band MOSFIRE
        nwave = 25
        lam0 = 6564.60
        wave =  np.linspace(lam0*(1.+gal.z) -dlam*(nwave-1)/2.,
                        lam0*(1.+gal.z) +dlam*(nwave-1)/2.,num=nwave )
        spec2D = misfit.ObsSpectrum2D(band=self.band, wave=wave,
                        line_primary='HA6565',linegroup_name='Halpha',
                        no_continuum_subtraction=True)

        spec2D.shape = [ np.int(np.ceil(inst.slit_length/inst.pixscale)), nwave]
        spec2D.initialize_lineset()


        spec2D.m0 = np.ceil(inst.slit_length/inst.pixscale)/2.
        spec2D.ypos = spec2D.m0


        gal.set_spectrum_2D(spec2D)


        return gal

    def setup_instrument(self, PSF_type='Gaussian', psf_fwhm=0.7,
                    pixscale=0.1799, slit_width=0.7, slit_length=3.,
                    instrument_resolution=39.,
                    yspace_dither_arc = None):

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


        inst = misfit.Spectrograph(instrument_name='MOSFIRE', PSF_FWHM=psf_fwhm,
                        PSF_type=PSF_type, yspace_dither_arc=yspace_dither_arc,
                        pixscale=pixscale,slit_width=slit_width,slit_length=slit_length)
        inst.instrument_resolution = instrument_resolution

        inst.band = self.band

        return inst

    def setup_kinModelOptions(self):
        do_position_wave_shift=False
        do_inst_res_conv_effective=False

        kinModelOptions = misfit.KinModel2DOptions(nSubpixels=2,
                            pad_factor=0.5,
                            do_position_wave_shift=do_position_wave_shift,
                            do_inst_res_conv_effective=do_inst_res_conv_effective)

        return kinModelOptions

    def setup_kinProfile(self):

        kinProfile = misfit.KinProfileFiducial()

        return kinProfile

    def setup_intensityProfile(self, gal=None):
        intensityProfile = misfit.IntensityProfileFiducial(galaxy=gal)

        return intensityProfile



class TestModels:
    helper = HelperSetups()

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
            nPixels=np.int(np.round((nPixels_apermodel-1)/2)))
        disp_aper_ratio = aperModel1DDispMisalign.disp_aper_ratio
        print(disp_aper_ratio)

        ftol = 1.e-9
        disp_aper_ratio_true = 0.9884434977112926
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
            nPixels=np.int(np.round((nPixels_apermodel-1)/2)))
        disp_aper_ratio = aperModel1DDispMisalign.disp_aper_ratio
        print(disp_aper_ratio)

        ftol = 1.e-9
        disp_aper_ratio_true = 0.9892714489739454
        assert math.isclose(disp_aper_ratio, disp_aper_ratio_true, rel_tol=ftol)


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
