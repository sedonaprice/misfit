# Copyright 2014, 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

from __future__ import print_function

import numpy as _np
import kin_functions as _kfuncs

from scipy.stats import norm as _norm
from astropy.extern import six as _six

from misfit.general import general_utils as _utils

import astropy.constants as _const
c_kms = _const.c.cgs.value/1.e5

class AperModel1DBase(object):
    def __init__(self, **kwargs):
        self.galaxy=None
        self.instrument=None
        
        self.nPixels = 201
        self.extraction_method = 'optimal'
        self.extraction_width = None  # Full width of extraction aperture, in arcsec.
        
        # Radius for aperture correction to be calculated at:
        self.disp_aper_radius_arcsec = None
        
        
        self.disp_aper_ratio = None
        
        
        self.setAttr(**kwargs)
        
    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))
        
        
class AperModel1DDispBase(AperModel1DBase):
    def __init__(self, **kwargs):
        super(AperModel1DDispBase, self).__init__(**kwargs)
        
        # theta should be [???]
        self.r_core = 1./300.
        self.d = -0.089
        self.Ie = 1.
        
        self.nPixels = 100
        
        self.setAttr(**kwargs)
        
        
    def setup_model_grid(self,re_arcsec = None, q=None, delt_PA=None):
        
        # If it's too small, use extra pixels:
        if re_arcsec <= 0.1:
            self.nPixels *= 2
            
        self.nX = 4*self.nPixels+3
        self.nY = 4*self.nPixels+3
        self.padX = self.nPixels+1
        self.padY = self.nPixels+1
        
        #########
        # Setup aperture width if not set:
        if self.extraction_width is None:
            re_arcsec_conv = 0.5*_np.sqrt((2.*re_arcsec)**2 + self.instrument.PSF.PSF_FWHM**2)
            minor_conv_arcsec = 0.5*_np.sqrt((2.*q*re_arcsec)**2 + \
                        self.instrument.PSF.PSF_FWHM**2)
            q_conv = minor_conv_arcsec/re_arcsec_conv
            
            x_proj_conv = _utils.x_proj_major(re_arcsec_conv, delt_PA, q_conv)
            y_proj_conv = _utils.y_proj_major(re_arcsec_conv, delt_PA)
            # HWHM
            
            ## This goes into the APERTURE SIZE, so should AT LEAST be minor axis size, 
            #       even if misaligned!!
            if y_proj_conv < re_arcsec_conv*q_conv:
                y_proj_conv = re_arcsec_conv*q_conv
                
            self.extraction_width = 4.*y_proj_conv.copy()
        
        
        self.x_aper = self.instrument.slit_width
        
        
        self.delt_x = 0.5*self.x_aper/(self.nPixels+0.5)
        self.delt_y = 0.5*self.extraction_width/(self.nPixels+0.5)
        
        # coordinates at each point:
        self.x_arr = _np.linspace((-2.*self.nPixels-1.)*self.delt_x, 
                    (2.*self.nPixels+1.)*self.delt_x, num=self.nX)
        self.y_arr = _np.linspace((-2.*self.nPixels-1.)*self.delt_y, 
                    (2.*self.nPixels+1.)*self.delt_y, num=self.nY)
        # x_arr, y_arr are in *ARCSECONDS*
        
        
        self.x_arr_full = _np.tile(self.x_arr, (self.nY,1))
        self.y_arr_full = _np.tile(_np.array([self.y_arr]).T, (1,self.nX))
        
        ###########################
        ###########################
        # Setup grid for rotate case:
        
        self.nXp = 4*self.nPixels+3
        self.nYp = 4*self.nPixels+3
        self.padXp = self.nPixels+1
        self.padYp = self.nPixels+1
        
        # WE WILL NEED TO TRIM TO x_aper, y_aper later, after rotation. For now:
        self.xp_aper = self.extraction_width
        self.yp_aper = self.extraction_width
        
        # Should be 2*n_pix_xp + 1 across xp_aper_full
        # or n_pix_xp+0.5 across xp_aper_half
        self.delt_xp = 0.5*self.xp_aper/(self.nPixels+0.5)
        self.delt_yp = 0.5*self.yp_aper/(self.nPixels+0.5)
        
        # coordinates at each point:
        self.xp_arr = _np.linspace(-(self.nPixels+self.padXp)*self.delt_xp, 
                                (self.nPixels+self.padXp)*self.delt_xp, num=self.nXp)
        self.yp_arr = _np.linspace(-(self.nPixels+self.padYp)*self.delt_yp, 
                                (self.nPixels+self.padYp)*self.delt_yp, num=self.nYp)
        # x_arr, y_arr are in *ARCSECONDS*
        self.xp_arr_full = _np.tile(self.xp_arr, (self.nYp,1))
        self.yp_arr_full = _np.tile(_np.array([self.yp_arr]).T, (1,self.nXp))
        
        self.delt_PAp = 0.
        
        
        
class AperModel1DDisp(AperModel1DDispBase):
    def __init__(self, **kwargs):
        super(AperModel1DDisp, self).__init__(**kwargs)
        
        self.setAttr(**kwargs)
        self.setup_calcs()
        self.calc_aper_correction()
        
    def setup_calcs(self):
        self.galaxy.re_arcsec_circ = self.galaxy.re_arcsec* _np.sqrt(self.galaxy.q)

        if self.disp_aper_radius_arcsec is None:
            self.disp_aper_radius_arcsec = self.galaxy.re_arcsec_circ

        self.setup_model_grid(re_arcsec = self.galaxy.re_arcsec_circ, q=1., delt_PA=0.)

        # Set PSF convolution:
        cent = [_np.mean(self.x_arr), _np.mean(self.y_arr)]
        x_off_arc = self.x_arr-cent[0]
        y_off_arc = self.y_arr-cent[1]
        conv_stamp = self.instrument.PSF.generate_conv_stamp(x_off_arc, y_off_arc)
        self.instrument.PSF.set_conv_stamp(conv_stamp)
        
    def calc_aper_correction(self):
        self.sigma_ap = _kfuncs.sigma_aper_dispersion(aperModel1DDisp = self, 
                        re_arcsec=self.galaxy.re_arcsec_circ)
                        
        self.sigma_e = _kfuncs.sigma_e_dispersion(aperModel1DDisp = self, 
                        re_arcsec=self.galaxy.re_arcsec_circ,
                        r_outer = self.disp_aper_radius_arcsec)
        
        self.disp_aper_ratio = self.sigma_ap/self.sigma_e
        
        
        
class AperModel1DDispMisalign(AperModel1DDispBase):
    def __init__(self, **kwargs):
        super(AperModel1DDispMisalign, self).__init__(**kwargs)
        
        self.setAttr(**kwargs)
        self.setup_calcs()
        self.calc_aper_correction()
        
    def setup_calcs(self):
        self.galaxy.re_arcsec_circ = self.galaxy.re_arcsec* _np.sqrt(self.galaxy.q)
        
        if self.disp_aper_radius_arcsec is None:
            self.disp_aper_radius_arcsec = self.galaxy.re_arcsec_circ
            
        self.setup_model_grid(re_arcsec = self.galaxy.re_arcsec, q=self.galaxy.q, 
                    delt_PA=self.galaxy.delt_PA)
        
        # Set PSF convolution:
        cent = [_np.mean(self.xp_arr), _np.mean(self.yp_arr)]
        x_off_arc = self.xp_arr-cent[0]
        y_off_arc = self.yp_arr-cent[1]
        conv_stamp = self.instrument.PSF.generate_conv_stamp(x_off_arc, y_off_arc)
        self.instrument.PSF.set_conv_stamp(conv_stamp)
        
    def calc_aper_correction(self):
        self.sigma_ap = _kfuncs.sigma_aper_dispersion_misalign(aperModel1DDisp = self,
                        re_arcsec=self.galaxy.re_arcsec_circ)
        
        self.sigma_e = _kfuncs.sigma_e_dispersion(aperModel1DDisp = self, 
                        re_arcsec=self.galaxy.re_arcsec_circ,
                        r_outer = self.disp_aper_radius_arcsec)
        
        self.disp_aper_ratio = self.sigma_ap/self.sigma_e
        
        
class AperModel1DRot(AperModel1DBase):
    def __init__(self, **kwargs):
        super(AperModel1DRot, self).__init__(**kwargs)
        
        
        self.do_position_wave_shift = False
        self.do_inst_res_conv_effective = False
        
        self.pad_factor = 0.5
        
        # Position of galaxy within slit:
        self.xc = 0.    # Spatial direction offest, in ARCSEC.
        self.yc = 0.    # Defined to be 0 given how an aperture is extracted. in arcsec.
        
        # theta should be kin input: 
        # theta should be [V/sigma_RE, V_tmp, r_t]
        self.theta = None
        self.kinProfile = None
        self.intensityProfile = None
        
        # Containers to hold some of the longest calculation bits, for faster re-calculations.
        self.delta_position_velspace = None
        self.inst_disp_vel = None
        self.I_wide = None
        
        
        
        self.setAttr(**kwargs)
        self.setup_calcs()
        self.calc_aper_correction()
        
    def setup_calcs(self):
        if self.disp_aper_radius_arcsec is None:
            self.disp_aper_radius_arcsec = self.galaxy.re_arcsec
        self.setup_model_grid()
        
        # Initialize PSF:
        # Set PSF convolution:
        
        # Convolve arrays with PSF:
        # Trim x_arr, y_arr so you're only making a convolution kernal for +- FWHM on the 
        # x_arr, y_arr are in *ARCSECONDS*
        xc = _np.average(self.x_arr)
        yc = _np.average(self.y_arr)
        x_arr_trim = self.x_arr[_np.abs(self.x_arr-xc) <= self.instrument.PSF.PSF_FWHM]
        y_arr_trim = self.y_arr[_np.abs(self.y_arr-yc) <= self.instrument.PSF.PSF_FWHM]
        
        # x_arr_flat = _np.tile(x_arr_trim, (len(y_arr_trim),1))
        # y_arr_flat = _np.tile(_np.array([y_arr_trim]).T, (1,len(x_arr_trim)))
        
        
        conv_stamp = self.instrument.PSF.generate_conv_stamp(x_arr_trim, y_arr_trim)
        self.instrument.PSF.set_conv_stamp(conv_stamp)
        

    def calc_aper_correction(self, recalc=False):
        self.update_kinProfile()
        
        
        self.calculate_kin_cube()
        self.convolve_kin_arr_PSF()
        self.trim_sum_aperture()
        # Calculates self.sigma_aper
        
        
        # Get VRMS at aperture radius (default: R_E):
        V_aper = self.kinProfile.vel(self.disp_aper_radius_arcsec, 0.)
        sigma_at_aper = self.kinProfile.sigma(self.disp_aper_radius_arcsec, 0.)
        self.V_RMS_aper = _np.sqrt(V_aper**2 + sigma_at_aper**2)
        
        self.disp_aper_ratio = self.sigma_aper/self.V_RMS_aper
        
        
        
        
    def update_kinProfile(self):
        # Eg, for arctan: input theta is [V/sig(RE), V_a_dummy, r_t]
        # Kin param input is [V_a, r_t, sigma_0]
        
        # Check if V/sig(RE) is None:
        if self.theta[0] is not None:
            theta_kinprof = _np.array([self.theta[1], self.theta[2], None])
        else:
            theta_kinprof = _np.array([self.theta[1], self.theta[2], 0.])
            
        # Get V(RE), then calc sigma_0: self.theta[1]/self.theta[0]
        
        self.kinProfile.update_theta(theta_kinprof)
        
        if self.theta[0] is not None:
            V_aper = self.kinProfile.vel(self.disp_aper_radius_arcsec, 0.)
            theta_kinprof = _np.array([self.theta[1], self.theta[2], V_aper/self.theta[0]])
            self.kinProfile.update_theta(theta_kinprof)
        
        
    def setup_model_grid(self):
        #########
        # Setup aperture width if not set:
        if self.extraction_width is None:
            re_arcsec_conv = 0.5*_np.sqrt((2.*self.galaxy.re_arcsec)**2 + self.instrument.PSF.PSF_FWHM**2)
            minor_conv_arcsec = 0.5*_np.sqrt((2.*self.galaxy.q*self.galaxy.re_arcsec)**2 + \
                        self.instrument.PSF.PSF_FWHM**2)
            q_conv = minor_conv_arcsec/re_arcsec_conv
            
            x_proj_conv = _utils.x_proj_major(re_arcsec_conv, self.galaxy.delt_PA, q_conv)
            y_proj_conv = _utils.y_proj_major(re_arcsec_conv, self.galaxy.delt_PA)
            # HWHM
            
            ## This goes into the APERTURE SIZE, so should AT LEAST be minor axis size, 
            #       even if misaligned!!
            if y_proj_conv < re_arcsec_conv*q_conv:
                y_proj_conv = re_arcsec_conv*q_conv
                
            self.extraction_width = 4.*y_proj_conv.copy()
        
        
        #########
        # Setup grid, values:
        self.sini = _kfuncs.sin_i(self.galaxy.q, q0=self.galaxy.q0)
        self.i_rad = _np.arcsin(self.sini)
        
        self.y_aper = self.extraction_width    # Convenience
        
        self.n_pix_y = self.nPixels 
        self.delt_y = self.y_aper/self.nPixels 
        
        
        # Make num pix in x dir such that delt_x ~ delt_y, but delt_x*n_pix_x = slit_wid
        # so n_pix_x = slit_wid/delt_x
        self.n_pix_x = _np.int(_np.ceil(self.instrument.slit_width/self.delt_y))
        self.delt_x = self.instrument.slit_width/self.n_pix_x
        
        # Use same number of pixels as in y in the z dir:
        self.n_pix_z = self.n_pix_y
        self.delt_z = self.delt_y
        
        # +-----------------------------------------------------------
        # Pad in the x, y directions: PSF convolution.
        #   pad U/D, L/R by SEEING_FWHM*self.pad_factor
        # +-----------------------------------------------------------
        self.padY = _np.int(_np.round(((self.instrument.PSF.PSF_FWHM*self.pad_factor)/self.delt_y) ))
        self.padX = _np.int(_np.round(((self.instrument.PSF.PSF_FWHM*self.pad_factor)/self.delt_x) ))
        
        self.nX = self.n_pix_x + 2*self.padX
        self.nY = self.n_pix_y + 2*self.padY
        self.nZ = self.n_pix_z
        
        # coordinates at each point:
        self.x_arr = _np.linspace(-(self.n_pix_x/2.+self.padX-0.5)*self.delt_x, 
                        (self.n_pix_x/2.+self.padX-0.5)*self.delt_x, num=self.nX)
        
        self.z_arr = _np.linspace(-(self.n_pix_z/2.-0.5)*self.delt_z, 
                            (self.n_pix_z/2.-0.5)*self.delt_z, num=self.nZ)
                            
        self.y_arr = _np.linspace(-(self.n_pix_y/2.+self.padY-0.5)*self.delt_y, 
                    (self.n_pix_y/2.+self.padY-0.5)*self.delt_y, num=self.nY)
        
        # Offset by xc, yc, if non-zero:
        self.x_arr = self.x_arr - self.xc
        self.y_arr = self.y_arr - self.yc
        
        # Make 3D arrays:
        # xyz_arr are in *ARCSECONDS*
        self.x_arr_full = _np.tile(self.x_arr, (self.nZ, self.nY,1))
        self.y_arr_full = _np.tile(_np.array([self.y_arr]).T, (self.nZ, 1,self.nX))
        self.z_arr_full = _np.repeat(self.z_arr,
                            self.nY*self.nX).reshape((self.nZ,self.nY,self.nX))
                            
        self.xint, self.yint, self.zint = _kfuncs.transform_int_3d(self.x_arr_full,
                                self.y_arr_full, self.z_arr_full, 
                                self.galaxy.delt_PA, self.i_rad)
        self.r = _np.sqrt(self.xint**2+self.yint**2)
        
        
        # Shape: (nZ,nY,nX)
        self.I_wide  = self.intensityProfile.int(self.r, self.zint)
        
        # Get primary lam0:
        self.z = self.galaxy.z
        
        
        self.lam0_primary = self.galaxy.spec2D.restwave_arr[0]*(1.+self.z)


    def calculate_kin_cube(self):
        I_wide = self.I_wide
        cos_phi = self.yint/self.r
        
        # Mask the cos_phi where r == 0:
        cos_phi[self.r == 0.] = 0.
        
        V_wide = self.kinProfile.vel(self.r, self.zint)*cos_phi*self.sini
        sigma_wide = self.kinProfile.sigma(self.r, self.zint)
        
        # Add position offset, if calculating that shift:
        if self.do_position_wave_shift or self.do_inst_res_conv_effective:
            if (self.delta_position_velspace is None) or (self.inst_disp_wave is None):
                #################
                # Exclude the padded regions ????
                wh_in_slit_x = _np.array(_six.moves.xrange(self.padX,self.n_pix_x+self.padX))
                # Currently doesn't support MSA-style calculations.
                # Want instrument res in vel_FWHM, not vel_disp -> *2.35
                R = c_kms/(self.instrument.instrument_resolution*(2.*_np.sqrt(2.*_np.log(2.))))
                #print("R=", R)
                slit_width_R_meas = self.instrument.slit_width
                tot_slit_width =  self.instrument.slit_width
                
                self.delta_position_velspace, inst_disp_wave = \
                        _kfuncs.calculate_effective_inst_res_lam_shift(I_wide_3d=I_wide,
                            xarr = self.x_arr,
                            yarr = self.y_arr,
                            wh_in_slit_x=wh_in_slit_x, 
                            R=R, slit_width_R_meas=slit_width_R_meas, 
                            tot_slit_width=tot_slit_width, lam0=self.lam0_primary,
                            PSF=self.instrument.PSF)
                self.inst_disp_vel = inst_disp_wave/self.lam0_primary * c_kms # [km/s]
        #################
        
        if self.do_position_wave_shift:
            # Add delta_position_velspace calculated from deltax_y to V matrix
            for i in _six.moves.xrange(self.nY):
                V_wide[:,i,:] += self.delta_position_velspace[i]
        
        self.RMS_sq_wide = (V_wide)**2 + (sigma_wide)**2
        
        sigsq_I_wide = self.RMS_sq_wide*self.I_wide

        # Collapse in z direction: get avg RMS vel in each x,y position!
        # shape: (nY,nX)
        self.sigsq_I_coll = _np.sum(sigsq_I_wide, axis=0)*self.delt_z
        self.I_coll = _np.sum(self.I_wide, axis=0)*self.delt_z
        
    def convolve_kin_arr_PSF(self):
        # -----------------------------------------------
        # Convolve arrays with PSF:
        # Trim x_arr, y_arr so you're only making a convolution kernal for +- FWHM on the 
        # x_arr, y_arr are in *ARCSECONDS*
        
        ###
        # _kfuncs.PSF_convolve(spectra_cube, nWave, x_arr_flat, y_arr_flat, PSF_FWHM):
        #     # Convolve with seeing:
        #     # Should have shape (nWave, nY, nX)
        
        self.sigsq_I_conv_wide = _kfuncs.PSF_convolve_flat(self.sigsq_I_coll, self.instrument.PSF)
        self.I_conv_wide =  _kfuncs.PSF_convolve_flat(self.I_coll, self.instrument.PSF)
        
        
        
        
    def trim_sum_aperture(self):
        # Trim down to the real size:
        trimx = [self.padX,self.n_pix_x+self.padX]
        trimy = [self.padY,self.n_pix_y+self.padY]
        
        self.I_conv = self.I_conv_wide[trimy[0]:trimy[1],trimx[0]:trimx[1]]
        self.sigsq_I_conv = self.sigsq_I_conv_wide[trimy[0]:trimy[1],trimx[0]:trimx[1]]

        # -----------------------------------------------
        # Sum to find sigma_sq in aperture:

        if self.extraction_method == 'optimal':
            prof_sigma = (0.5*self.y_aper/self.delt_y)/(2.*_np.sqrt(2.*_np.log(2.)))
            # y extent = number of ROWS - shape[0]
            yy_prof = _np.array(_six.moves.xrange(self.I_conv.shape[0])) - (self.I_conv.shape[0]-1.)/2.

            g_y = _norm.pdf(yy_prof, 0., prof_sigma)
            self.g_y = g_y/_np.sum(g_y)  # normalize

            self.G_y = _np.tile(_np.array([self.g_y]).transpose(), (1, self.I_conv.shape[1]))
        elif self.extraction_method == 'boxcar':
            self.G_y = _np.ones(self.I_conv.shape)
        else:
            raise ValueError("Specified extraction method is not supported yet!: "+self.extraction_method)


        self.sigmasq_aper = _np.sum(self.sigsq_I_conv*self.G_y*self.delt_x*self.delt_y)/\
                                _np.sum(self.I_conv*self.G_y*self.delt_x*self.delt_y)
        self.sigma_aper = _np.sqrt(self.sigmasq_aper)
        

class AperModel2D(object):
    """
    Use galaxy + instrument properties to generate mock obs kinematics 
    observation described by kinProfile and theta
    Fiducial theta: [Va, rt, sigma0, m0shift, z]
        m0shift, z should always be the last two entries.
    """
    def __init__(self, **kwargs):
        
        self.galaxy=None
        self.instrument=None
        self.theta = None
        self.kinProfile = None
        self.intensityProfile = None
        
        self.nSubpixels = 1
        self.pad_factor = 0.5
        self.rebin_output = True   # Turn off to get high-res model.
        self.do_position_wave_shift = False
        self.do_inst_res_conv_effective = False
        
        # Position of galaxy within slit:
        self.xc = 0.    # Spatial direction offest, in ARCSEC.
        self.yc = 0.    # Defined to be 0 given m0+m0_shift offset. in arcsec.
        
        self.model = None
        
        # Containers to hold some of the longest calculation bits, for faster re-calculations.
        self.delta_position_velspace = None
        self.inst_disp_wave = None
        self.I_wide = None
        
        self.debug = False
        
        # Return calcs in wave / velocity space
        self.spec_type = 'wave'  # wave / velocity
        
        
        # Options for handling v. small dispersion calculations:
        self.absvalsigma = False
        self.adaptive_upsample_wave = False
        self.adaptive_upsample_factor = 3.
        self.sigma_floor = False
        self.sigma_floor_value = None
        
        
        self.setAttr(**kwargs)
        self.make_model()

    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))
        
    def setup_calcs(self):
        self.setup_model_grid()
        
        if self.adaptive_upsample_wave:
            self.wave_arr_real = self.wave_arr.copy()
            self.nWave_real = self.nWave
            self.delt_wave_real = self.delt_wave
            self.I_matrix_real = self.I_matrix.copy()
            self.wave_ref_matrix_real = self.wave_ref_matrix.copy()
            
        if (self.sigma_floor) | (self.adaptive_upsample_wave):
            if self.spec_type == 'wave':
                self.sigma_floor_value = (self.delt_wave/self.lam0_primary)*c_kms /(2.35*2.)
            elif self.spec_type == 'velocity':
                self.sigma_floor_value = self.delt_wave/(2.35*2.)
        
        # Initialize PSF:
        # Set PSF convolution:
        ## Convolve the spectra cube with the PSF:
        ## Trim x_arr, y_arr so you're only making a convolution kernal for +- FWHM
        #       x_arr, y_arr are in *ARCSECONDS*
        xc = _np.average(self.x_arr)
        yc = _np.average(self.y_arr)
        x_arr_trim = self.x_arr[_np.abs(self.x_arr-xc) <= self.instrument.PSF.PSF_FWHM]
        y_arr_trim = self.y_arr[_np.abs(self.y_arr-yc) <= self.instrument.PSF.PSF_FWHM]
        
        #x_arr_flat = _np.tile(x_arr_trim, (len(y_arr_trim),1))
        #y_arr_flat = _np.tile(_np.array([y_arr_trim]).T, (1,len(x_arr_trim)))
        
        
        #    # x and y should be coordinates in *ARCSECONDS*
        #    # Input x_arr_full, y_arr_full for matrix calculation
        #    # z can either be true z, or WAVELENGTH
        
        
        conv_stamp = self.instrument.PSF.generate_conv_stamp(x_arr_trim, y_arr_trim)
        # Test new:
        kern3D = _np.zeros(shape=(1, conv_stamp.shape[0], conv_stamp.shape[1],))
        kern3D[0, :, :] = conv_stamp
        
        self.instrument.PSF.set_conv_stamp(kern3D)
        
    def make_model(self):
        
        self.setup_calcs()                 # Setup grid, values
        
        self.do_model_calcs()
        
    def update_model(self, **kwargs):
        self.setAttr(**kwargs)
        
        self.do_model_calcs()
        
    def do_model_calcs(self):    
        
        self.kinProfile.update_theta(self.theta)
        
        if self.adaptive_upsample_wave:
            self.check_do_upsample_wave()
            
        self.calculate_spectral_cube()                     # Make specral cube
        
        self.convolve_spectral_cube_PSF()                  # Convolve spectral cube with PSF
        self.trim_downsample_spectral_cube()               # Trim and downsample cube
        self.convolve_instrument_resolution()              # Convolve with instrument res
        
        
        if self.adaptive_upsample_wave:
            self.downsample_to_real_wave()
        
        ## galaxy.spec2D.flux: shape is nY, nWave, so transpose to return!
        self.model = self.model_out.T
        
        
    def setup_model_grid(self):
        #########
        # Setup grid, values:
        self.sini = _kfuncs.sin_i(self.galaxy.q, q0=self.galaxy.q0)
        self.i_rad = _np.arcsin(self.sini)
        
        if not self.galaxy.generate_model:
            self.n_pix_y_whole = self.galaxy.spec2D.flux.shape[0]
        else:
            self.n_pix_y_whole = self.galaxy.spec2D.shape[0]
            
        self.n_pix_y = self.n_pix_y_whole*self.nSubpixels
        
        self.delt_y = self.instrument.pixscale/self.nSubpixels # Subpixel scale
        
        self.y_aper = self.n_pix_y*self.delt_y
        
        
        # Make num pix in x dir such that delt_x ~ delt_y, 
        #   but delt_x*n_pix_x = slit_wid,  so n_pix_x = slit_wid/delt_x
        self.n_pix_x = _np.int(_np.ceil(self.instrument.slit_width/self.delt_y))
        self.delt_x = self.instrument.slit_width/self.n_pix_x
        
        # Use same number of pixels as in y in the z dir:
        self.n_pix_z = self.n_pix_y
        self.delt_z = self.delt_y
        
        # +-----------------------------------------------------------
        # Pad in the x, y directions: PSF convolution.
        #   pad U/D, L/R by SEEING_FWHM*self.pad_factor
        # +-----------------------------------------------------------
        self.padY = _np.int(_np.round(((self.instrument.PSF.PSF_FWHM*self.pad_factor)/self.delt_y) ))
        self.padX = _np.int(_np.round(((self.instrument.PSF.PSF_FWHM*self.pad_factor)/self.delt_x) ))
        
        
        self.nX = self.n_pix_x + 2*self.padX
        self.nY = self.n_pix_y + 2*self.padY
        self.nZ = self.n_pix_z
        
        # coordinates at each point:
        self.x_arr = _np.linspace(-(self.n_pix_x/2.+self.padX-0.5)*self.delt_x, 
                        (self.n_pix_x/2.+self.padX-0.5)*self.delt_x, num=self.nX)
                        
        # Offset by xc, if non-zero:
        self.x_arr = self.x_arr - self.xc
        
                        
        self.z_arr = _np.linspace(-(self.n_pix_z/2.-0.5)*self.delt_z, 
                            (self.n_pix_z/2.-0.5)*self.delt_z, num=self.nZ)
                            
        self.y_arr = _np.linspace(-(self.padY-0.5)*self.delt_y, 
                ((self.n_pix_y_whole-1)*self.nSubpixels + self.padY-0.5)*self.delt_y, 
                num=self.nY)
        
        self.z = self.theta[-1]
        
        # Offset m0_fit to be at 0.:
        self.m0_fit = self.galaxy.spec2D.m0 + self.theta[-2]
        # Get primary lam0:
        if self.spec_type == 'wave': 
            self.lam0_primary = self.galaxy.spec2D.restwave_arr[0]*(1.+self.z)
        else:
            self.lam0_primary = 1.   # multiplicative unit
        
        self.y_arr -= ( self.m0_fit*self.nSubpixels )*self.delt_y

        # Make 3D arrays:
        # xyz_arr are in *ARCSECONDS*
        # shape: nZ, nY, nX
        self.x_arr_full = _np.tile(self.x_arr, (self.nZ, self.nY,1))
        self.y_arr_full = _np.tile(_np.array([self.y_arr]).T, (self.nZ, 1,self.nX))
        self.z_arr_full = _np.repeat(self.z_arr,
                            self.nY*self.nX).reshape((self.nZ,self.nY,self.nX))
                            
        self.xint, self.yint, self.zint = _kfuncs.transform_int_3d(self.x_arr_full,
                                self.y_arr_full, self.z_arr_full, 
                                self.galaxy.delt_PA, self.i_rad)
        self.r = _np.sqrt(self.xint**2+self.yint**2)
        
        
        self.wave_arr = self.galaxy.spec2D.wave.copy()
        self.nWave = len(self.wave_arr)
        self.delt_wave = _np.average(self.wave_arr[1:]-self.wave_arr[:-1])
        
        
        self.I_wide  = self.intensityProfile.int(self.r, self.zint)
        # Shape nWave, nZ, nY, nX
        self.I_matrix = _np.tile(self.I_wide, (self.nWave,1,1,1))
        self.wave_ref_matrix = _np.repeat(self.wave_arr, 
                self.nZ*self.nY*self.nX).reshape((self.nWave,self.nZ,self.nY,self.nX))
        
    def check_do_upsample_wave(self):
        if (_np.abs(self.kinProfile.dispProfile.theta) < self.sigma_floor_value) : 
            # Create new array that has finer sampling:
            self.delt_wave /= _np.float(self.adaptive_upsample_factor)
            
            wave_arr = _np.linspace(self.wave_arr.min(), self.wave_arr.max(), 
                    num=((self.wave_arr.max()-self.wave_arr.min())/self.delt_wave + 1) )
            
            self.wave_arr = wave_arr
            self.nWave = len(self.wave_arr)
            
            # Shape nWave, nZ, nY, nX
            self.I_matrix = _np.tile(self.I_wide, (self.nWave,1,1,1))
            self.wave_ref_matrix = _np.repeat(self.wave_arr, 
                    self.nZ*self.nY*self.nX).reshape((self.nWave,self.nZ,self.nY,self.nX))
            
            
            
    def calculate_spectral_cube(self):
        # if self.I_wide is None:
        #     I_wide = self.intensityProfile.int(self.r, self.zint)
        #     self.I_wide = I_wide
        # else:
        I_wide = self.I_wide
        
        cos_phi = self.yint/self.r
        # Mask the cos_phi where r == 0:
        cos_phi[self.r == 0.] = 0.
        
        V_wide = self.kinProfile.vel(self.r,self.zint)*cos_phi*self.sini
        if self.debug:
            self.V_wide = V_wide
            self.I_Vsq_wide = I_wide * (V_wide**2)
        
        #
        if self.absvalsigma:
            self.kinProfile.dispProfile.theta = _np.abs(self.kinProfile.dispProfile.theta)
        if self.sigma_floor:
            if _np.abs(self.kinProfile.dispProfile.theta) < self.sigma_floor_value:
                self.kinProfile.dispProfile.theta = _np.array([self.sigma_floor_value])
        
        sigma_wide = self.kinProfile.sigma(self.r,self.zint)
        
        #################
        if self.do_position_wave_shift or self.do_inst_res_conv_effective:
            if (self.delta_position_velspace is None) or (self.inst_disp_wave is None):
                #################
                # Exclude the padded regions ????
                wh_in_slit_x = _np.array(_six.moves.xrange(self.padX,self.n_pix_x+self.padX))
                # Currently doesn't support MSA-style calculations.
                # Want instrument res in vel_FWHM, not vel_disp -> *2.35
                R = c_kms/(self.instrument.instrument_resolution*(2.*_np.sqrt(2.*_np.log(2.))))
                #print("R=", R)
                slit_width_R_meas = self.instrument.slit_width
                tot_slit_width =  self.instrument.slit_width
                
                self.delta_position_velspace, self.inst_disp_wave = \
                        _kfuncs.calculate_effective_inst_res_lam_shift(I_wide_3d=I_wide,
                            xarr = self.x_arr, #+self.xc, 
                            yarr = self.y_arr, #+self.yc, 
                            wh_in_slit_x=wh_in_slit_x, 
                            R=R, slit_width_R_meas=slit_width_R_meas, 
                            tot_slit_width=tot_slit_width, lam0=self.lam0_primary,
                            PSF=self.instrument.PSF)
                            
                        
        #################
        
        if self.do_position_wave_shift:
            # Add delta_position_velspace calculated from deltax_y to V matrix
            for i in _six.moves.xrange(self.nY):
                V_wide[:,i,:] += self.delta_position_velspace[i]
        
        # start = time.time()
        self.spectra_cube = _kfuncs.add_sigma_collapse_z(I_wide, V_wide, sigma_wide, 
                    self.wave_arr, self.z, self.galaxy.spec2D.restwave_arr, 
                    self.galaxy.spec2D.flux_ratio_arr, 
                    self.nWave, self.nZ, self.nY, self.nX, 
                    self.delt_z, self.delt_wave,
                    self.I_matrix, self.wave_ref_matrix,
                    spec_type=self.spec_type)
        ## Shape: (nWave,nY,nX)
        # end = time.time()
        # print("             add_sigma_collapse_z time:", end-start)
        
        
    def convolve_spectral_cube_PSF(self):
        ## Convolve the spectra cube with the PSF:
        self.spectra_cube_conv = _kfuncs.PSF_convolve(self.spectra_cube, self.instrument.PSF)
                        


    def trim_downsample_spectral_cube(self):
        # If slit has a mask (eg, not a simple rectangle):
        if self.instrument.slit_mask is not None:
            raise ValueError("implement, eg for JWST/NIRSpec")
        
        
        # Trim down to the real size:
        trimx = [self.padX,self.n_pix_x+self.padX]
        trimy = [self.padY,self.n_pix_y+self.padY]
        
        self.spectra_cube_conv = \
            self.spectra_cube_conv[:,trimy[0]:trimy[1],trimx[0]:trimx[1]]
            
        # collapse over x, ie along slit direction:
        self.model_out = _np.sum(self.spectra_cube_conv,axis=2)*self.delt_x
        
        # Downsample to same size as emis_t: collapse the subpixels in the y direction
        if self.rebin_output:
            self.model_out = _kfuncs.rebin(self.model_out, 
                                self.nWave, self.n_pix_y_whole)
                       
            

    def convolve_instrument_resolution(self):
        # self.instrument.instrument_resolution # vel disp [km/s]
        # sigma_lam = (sigma_kms/c_kms)*lam0
        if self.do_position_wave_shift:
            if self.spec_type == 'velocity':
                raise ValueError('pos wave shift + spectype=velocity not supported!')
            if not self.do_inst_res_conv_effective:
                self.inst_disp_wave = (self.instrument.instrument_resolution/c_kms)*self.lam0_primary
            if self.inst_disp_wave.min() > 0.:
                self.model_out = _kfuncs.inst_convol_complete(self.model_out, 
                                self.wave_arr, self.inst_disp_wave)
        else:
            if self.spec_type == 'wave':
                self.inst_disp_wave = (self.instrument.instrument_resolution/c_kms)*self.lam0_primary
            else:
                self.inst_disp_wave = self.instrument.instrument_resolution
            if self.inst_disp_wave > 0.:
                self.model_out = _kfuncs.inst_convol_complete(self.model_out, 
                                self.wave_arr, self.inst_disp_wave)


    def downsample_to_real_wave(self):
        self.model_out = _kfuncs.rebin(self.model_out, 
                            self.nWave_real, self.n_pix_y_whole)
        # Reset other parameters to original values:
        self.wave_arr = self.wave_arr_real.copy()
        self.nWave = self.nWave_real
        self.delt_wave = self.delt_wave_real
        self.I_matrix = self.I_matrix_real.copy()
        self.wave_ref_matrix = self.wave_ref_matrix_real.copy()



        