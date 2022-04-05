# Copyright 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function

import numpy as np
#from numba import jit
from numba import njit

import astropy.constants as const
c_kms = const.c.cgs.value/1.e5

from scipy.signal import fftconvolve
from scipy import integrate
from scipy.stats import norm
import six

# from skimage.transform import rotate as skimagerotate
# from scipy.ndimage.interpolation import map_coordinates as scipymapcoords


import lmfit

# Useful constants
deg2rad = np.pi/180.

def _between(bounds, value):
    return bounds[0] <= value <= bounds[1]


#@overload jitfftconvolve

############################################################################
############################################################################
############################################################################

# Functions for defining 3D model:

#@jit
#@jit(nopython=True)
@njit
def transform_int_3d(x, y, z, deltPA, i_rad):
    # INPUT deltPA is defined to be positive CCW from up.


    # First transform to projected, aligned with major axis coords:
    rad_deltPA = np.radians(deltPA)

    # # ORIGINAL convention:
    # # This coords are for positive deltPA that are CW from up.
    # xp = x*np.cos(rad_deltPA) - y*np.sin(rad_deltPA) # minor axis
    # yp = x*np.sin(rad_deltPA) + y*np.cos(rad_deltPA) # major axis
    # zp = z

    # SWITCH unit convention: deltPA is defined to be positive CCW from up.
    xp = x*np.cos(rad_deltPA) + y*np.sin(rad_deltPA) # minor axis
    yp = -x*np.sin(rad_deltPA) + y*np.cos(rad_deltPA) # major axis
    zp = z

    # yint = yp       # Major axis
    # xint = np.cos(i_rad)*xp - np.sin(i_rad)*zp      # minor axis direction
    # zint = np.sin(i_rad)*xp + np.cos(i_rad)*zp

    # Also switch convention
    yint = yp       # Major axis
    xint = np.cos(i_rad)*xp + np.sin(i_rad)*zp      # minor axis direction
    zint = -np.sin(i_rad)*xp + np.cos(i_rad)*zp

    return (xint,yint,zint)


#
def add_sigma_collapse_z(I_wide, V_wide, sigma_wide,
            wave_arr, z, restwave_arr, flux_ratio_arr,
            nWave, nZ, nY, nX,
            delt_z, delt_wave,
            I_matrix, wave_ref_matrix,
            spec_type = 'wave'):
    # make a spectra cube: each x,y has a spectrum with
    # intensity profile = sum(I(V,z,sigma_int))
    #   total intensity = I(x,y,z), mu=V(x,y,z), and simga=sigma_int

    V_matrix = np.tile(V_wide, (nWave,1,1,1))
    # I_matrix = np.tile(I_wide, (nWave,1,1,1))
    # wave_ref_matrix = np.repeat(wave_arr, nZ*nY*nX).reshape((nWave,nZ,nY,nX))

    # Convert velocities into wavelengths:
    # del_lam = lam0*(1+V/c)
    # sigma conversion:
    # Convert to wavelength offsets, sigmas:
    # wave_matrix = lam0*(1.+V_matrix/c_kms)

    I_Vxy = np.zeros( (nWave,nY,nX) )    # Shape: (nWave,nY,nX))

    if spec_type == 'wave':
        for i in six.moves.xrange(len(restwave_arr)):
            lam0 = restwave_arr[i] * (1.+z)
            # Sigma = sigma_profile(r)
            # Crude way of handling potential sigma = 0: just checking for MAX, not all values:
            if np.abs(sigma_wide).max() > 0.:
                sigma_lam_arr = (sigma_wide/c_kms)*lam0
                scale = 1/(sigma_lam_arr*np.sqrt(2.*np.pi))
                gaus = scale*np.exp(-(lam0*(1.+V_matrix/c_kms)-wave_ref_matrix)**2/(2.*sigma_lam_arr**2))

                # # When adding gaussian, you are preserving total I(x,y,z) over all V.
                # # So normalize the gaus distribution
                I_conv = I_matrix*gaus

                I_Vxy_oneline = np.sum(I_conv, axis=1)*delt_z
            else:
                I_Vxy_oneline = I_simple_collapse(I_wide, V_wide, wave_arr, nWave,
                                nZ, nY, nX, delt_z, delt_wave, lam0)

            I_Vxy += I_Vxy_oneline*flux_ratio_arr[i]
    elif spec_type == 'velocity':
        # Sigma = sigma_profile(r)
        if np.abs(sigma_wide).max() > 0.:
            scale = 1/(sigma_wide*np.sqrt(2.*np.pi))
            gaus = scale*np.exp(-(V_matrix-wave_ref_matrix)**2/(2.*sigma_wide**2))

            # # When adding gaussian, you are preserving total I(x,y,z) over all V.
            # # So normalize the gaus distribution
            I_conv = I_matrix*gaus

            I_Vxy_oneline = np.sum(I_conv, axis=1)*delt_z
        else:
            I_Vxy_oneline = I_simple_collapse(I_wide, V_wide, wave_arr, nWave,
                            nZ, nY, nX, delt_z, delt_wave, lam0)

        I_Vxy += I_Vxy_oneline

    return I_Vxy

#@jit(nopython=True)
#@jit
@njit
def I_simple_collapse(I_wide, V_wide, wave_arr,
                    nWave, nZ, nY, nX,
                    delt_z, delt_wave, lam0):
    # If no sigma: just V at each (x,y,z). No matrices.
    #   just gather all I,V over z at each (x,y): bin into vel_wave_arr

    I_Vxy = np.zeros((nWave, nY, nX))

    lam_step = delt_wave

    # for k in range(nX):
    #     for j in range(nY):
    #         # At every (x,y): for each z, find closest v bin, and add that to the intensity.
    #         for m in range(nZ):
    for k in six.moves.xrange(np.int(nX)):
        for j in six.moves.xrange(np.int(nY)):
            # At every (x,y): for each z, find closest v bin, and add that to the intensity.
            for m in six.moves.xrange(np.int(nZ)):

                lam = lam0*(1.+V_wide[m,j,k]/c_kms)
                I = I_wide[m,j,k]
                lam_diff = np.abs(lam - wave_arr)

                # Force it all into the closest pixel:
                wh_closest = np.argmin(v_diff)
                I_Vxy[wh_closest,j,k] += I*delt_z


    return I_Vxy


##################################################################
# Get instrument resolution + spatial -> wavelength offsets:
def calculate_effective_inst_res_lam_shift(I_wide_3d=None,
        xarr=None, yarr=None,
        wh_in_slit_x=None,
        R=None, slit_width_R_meas=None,
        tot_slit_width=None, lam0=None,
        PSF=None ):

    # xarr, yarr in ARCSEC
    #   xarr: 0 at center of slit
    #   yarr: 0 at center of slit or at center of object (depending on convention)
    #
    # Should return:
    #   del_velspace: km/s
    #   inst_res_fwhm_array: angstroms
    #
    #       calculated from FWHM_arcsec st

    # FIT:      FWHM_arcsec
    #           del_velspace


    FWHM_arcsec = np.zeros(len(yarr))     # [ARCSEC]
    mu_y = np.zeros(len(yarr))            # [ARCSEC]
    delx_y = np.zeros(len(yarr))          # [ARCSEC]

    # I_wide shape: (nZ, nY, nX)
    I_arr_2D = np.sum(I_wide_3d, axis=0)  # (nY, nX)

    ########################
    # Convolve with PSF:
    xc_psf = np.average(xarr)
    yc_psf = np.average(yarr)
    xarr_trim = xarr[np.abs(xarr-xc_psf) <= PSF.PSF_FWHM]
    yarr_trim = yarr[np.abs(yarr-yc_psf) <= PSF.PSF_FWHM]

    xarr_flat = np.tile(xarr_trim, (len(yarr_trim),1))
    yarr_flat = np.tile(np.array([yarr_trim]).T, (1,len(xarr_trim)))

    # # Convolve with seeing:
    # Should have shape (nWave, nY, nX)
    I_arr_2D_cube = np.array([I_arr_2D])
    I_arr_2D_conv_cube = PSF_convolve(I_arr_2D_cube, PSF)
    I_arr_2D_conv = I_arr_2D_conv_cube[0,:,:]
    ########################



    # Mask:
    mask_outer_vertonly = np.zeros(I_arr_2D_conv.shape)
    mask_outer_vertonly[:,wh_in_slit_x] = 1.
    I_arr_2D_conv = I_arr_2D_conv * mask_outer_vertonly

    # x_arr should be the UNSHIFTED coordinates

    for k in six.moves.xrange(len(yarr)):
        # Do gaussian fit to each row:
        A_tmp, mu_y[k], FWHM_arcsec[k] = lmfit_gaussian(xarr[wh_in_slit_x],
                                        I_arr_2D_conv[k,wh_in_slit_x])

        # # Process: check results:
        # print("k, mu_y[k], FWHM_arcsec[k]=", k, mu_y[k], FWHM_arcsec[k])

        # Need to check if x_l, x_u are w/in slit (defined by tot_slit_width)
        #   If not: truncate FWHM definition to = x_u-x_l
        #   And determine deltx_y = weighted mean position given best-fit profile.

        delx_y[k], FWHM_arcsec[k] = check_slit_illum(xarr, wh_in_slit_x,
                                A_tmp, mu_y[k], FWHM_arcsec[k], tot_slit_width)
        # print("post mods: delx_y[k], FWHM_arcsec[k]=", delx_y[k], FWHM_arcsec[k], "\n")

    Rw = R*slit_width_R_meas    # constant resolution*slit with

    R_eff_inst_res= Rw/FWHM_arcsec
    inst_res_fwhm_array = lam0/R_eff_inst_res  # [Angstroms]

    # Convert to dispersion [Angstroms]
    inst_disp_wave = inst_res_fwhm_array/(2.*np.sqrt(2.*np.log(2.)))


    R_eff_fullslit = Rw/tot_slit_width
    del_lam = lam0/R_eff_fullslit

    wavelength_scale = del_lam/tot_slit_width # [Angstroms / ARCSEC]

    del_velspace = delx_y * wavelength_scale/lam0 * c_kms # [km/s]


    return del_velspace, inst_disp_wave


def check_slit_illum(xarr, wh_in_slit_x, A_tmp, mu_y, FWHM_arcsec, tot_slit_width):
    x_l = mu_y - 0.5*FWHM_arcsec
    x_u = mu_y + 0.5*FWHM_arcsec

    # print("PRECHECK: FWHM_arcsec=", FWHM_arcsec)
    # print("mu_y=", mu_y)
    # print("initial: x_l, x_u=", x_l, x_u)
    FWHM_arcsec_in = FWHM_arcsec

    modified_bounds = False
    if x_l < -0.5*tot_slit_width:
        x_l = -0.5*tot_slit_width
        modified_bounds = True
    if x_u > 0.5*tot_slit_width:
        x_u = 0.5*tot_slit_width
        modified_bounds = True

    # Check that the object doesn't lie completely outside of slit: wrong order of x_l, x_u:
    if x_l > x_u:
        if x_u < -0.5*tot_slit_width:
            x_u = -0.5*tot_slit_width
            modified_bounds = True
        if x_l > 0.5*tot_slit_width:
            x_l = 0.5*tot_slit_width
            modified_bounds = True

    ##############
    delx_y = mu_y


    if modified_bounds:
        # Determine the new mu:
        model = A_tmp*gaus_from_fwhm(xarr, FWHM_arcsec, mu_y)
        model_mask = np.zeros(len(xarr))*np.NaN
        model_mask[wh_in_slit_x] = 1.
        model_orig = model.copy()
        model *= model_mask


        #delx_y = np.sum(xarr[wh_in_slit_x]*model[wh_in_slit_x])/np.sum(model[wh_in_slit_x])

        # Modify FWHM_arcsec as needed:
        # # if FWHM_arcsec[k] > tot_slit_width:
        # #     FWHM_arcsec[k] = tot_slit_width
        FWHM_arcsec = x_u - x_l

        if FWHM_arcsec == 0.:
            valmax = model[wh_in_slit_x].max()
            argmax = np.argmax(model[wh_in_slit_x])
            wh_half = np.where(np.abs(model[wh_in_slit_x]-\
                    0.5*valmax)==np.abs(model[wh_in_slit_x]-0.5*valmax).min())[0]
            FWHM_arcsec = xarr[wh_in_slit_x[argmax]] - xarr[wh_in_slit_x[wh_half[0]]]

        # Check if delx_y is inside the slit:
        if delx_y > 0.5*tot_slit_width:
            delx_y = 0.5*tot_slit_width #- 0.5*FWHM_arcsec

        if delx_y < -0.5*tot_slit_width:
            delx_y = -0.5*tot_slit_width #+ 0.5*FWHM_arcsec


        # ####################
        # # TESTING
        # import matplotlib.pyplot as plt
        # plt.plot(xarr, model_orig, ls='-', color='#1f77b4')
        # plt.gca().axvline(x=mu_y, ls='-', color='#1f77b4')
        # plt.gca().axvline(x=mu_y - 0.5*FWHM_arcsec_in, ls='--', color='#1f77b4')
        # plt.gca().axvline(x=mu_y + 0.5*FWHM_arcsec_in, ls='--', color='#1f77b4')
        # plt.plot(xarr[wh_in_slit_x], model[wh_in_slit_x], ls='-', color='#9467bd')
        # plt.gca().axvline(x=x_l, ls='--', color='#9467bd')
        # plt.gca().axvline(x=x_u, ls='--', color='#9467bd')
        # plt.gca().axvline(x=delx_y, ls='-', color='#ff7f0e')
        # plt.gca().axvline(x=delx_y - 0.5*FWHM_arcsec, ls='--', color='#ff7f0e')
        # plt.gca().axvline(x=delx_y + 0.5*FWHM_arcsec, ls='--', color='#ff7f0e')
        #
        # plt.gca().axvline(x=-0.5*tot_slit_width, ls=':', color='grey')
        # plt.gca().axvline(x=0.5*tot_slit_width, ls=':', color='grey')
        # plt.show()
        # ####################




    # else:
    #     delx_y = mu_y

    # # TEST:
    # delx_y = mu_y


    # print("FWHM_arcsec=", FWHM_arcsec)
    # print("x_l, x_u =", x_l, x_u)

    #raise ValueError


    return delx_y, FWHM_arcsec

## TEST 2019.11.14
#@jit
#@njit
def PSF_convolve(spectra_cube, PSF):
    # Convolve with seeing:
    # Should have shape (nWave, nY, nX)
    #seeing_sigma = PSF_FWHM/(2.*np.sqrt(2.*np.log(2.)))

    if PSF.PSF_FWHM > 0.:
        kern3D = PSF._conv_stamp

        spectra_cube_see_conv = fftconvolve(spectra_cube, kern3D, mode='same')
        # Conserve total flux:
        spectra_cube_see_conv *= np.sum(spectra_cube)/np.sum(spectra_cube_see_conv)

        return spectra_cube_see_conv
    else:
        return spectra_cube

## TEST 2019.11.14
#@jit
#@njit
def PSF_convolve_flat(pstamp, PSF):
    # Convolve with seeing:
    # Should have shape (nWave, nY, nX)
    #seeing_sigma = PSF_FWHM/(2.*np.sqrt(2.*np.log(2.)))

    if PSF.PSF_FWHM > 0.:

        conv_stamp = PSF._conv_stamp


        # FFT
        pstamp_conv = fftconvolve(pstamp, conv_stamp, mode='same')
        # # Conserve total flux:
        # fluxtot = np.sum(pstamp)
        # fluxtotnew = np.sum(pstamp_conv)
        # if fluxtotnew > 0.:
        #     pstamp_conv *= fluxtot/fluxtotnew
        # else:
        #     pstamp_conv *= 0.

        return pstamp_conv
    else:
        return pstamp




########################################################
# From scipy cookbook: http://wiki.scipy.org/Cookbook/Rebinning
def rebin(a, *args):
    shape = a.shape
    lenShape = len(shape)
    factor = np.array(np.asarray(shape)/np.asarray(args), dtype=np.int)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in six.moves.xrange(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in six.moves.xrange(lenShape)]
    return 1.*eval(''.join(evList))


## TEST 2019.11.14
#@jit
@njit
def convol_inst(model_out, xx, lam_cent, fwhm_array):
    # for k in range(model_out.shape[1]):
    for k in six.moves.xrange(model_out.shape[1]):
        wave_gaus = gaus_from_fwhm(xx, fwhm_array[k], lam_cent)
        # Replace non-finite with 0.
        wave_gaus[~np.isfinite(wave_gaus)] = 0.

        # k is each row of the output model (the y direction)

        # conserve total flux:
        fluxtot = np.sum(model_out[:,k])

        #model_out[:,k] = fftconvolve(model_out[:,k], wave_gaus, mode='same')
        #model_out[:,k] = jitfftconvolve(model_out[:,k], wave_gaus, mode='same')
        with objmode(out='float64[:]'):
             out = fftconvolve(model_out[:,k], wave_gaus, mode='same')
        model_out[:,k] = out
        fluxtotnew = np.sum(model_out[:,k])

        if fluxtotnew > 0.:
            model_out[:,k] *= fluxtot/fluxtotnew
        else:
            model_out[:,k] *= 0.


    return model_out

## TEST 2019.11.14
#@jit
#@njit
def convol_instconstfwhm(model_out, xx, lam_cent, fwhm_wave):
    wave_gaus = gaus_from_fwhm(xx, fwhm_wave, lam_cent)
    # Replace non-finite with 0.
    wave_gaus[~np.isfinite(wave_gaus)] = 0.
    wavegaus2D = np.zeros(shape=(wave_gaus.shape[0], 1,))
    wavegaus2D[:,0] = wave_gaus
    fluxtot = np.sum(model_out)
    model_out = fftconvolve(model_out, wavegaus2D, mode='same')
    # Conserve total flux:
    model_out *= fluxtot/np.sum(model_out)

    return model_out



#@jit
#@njit
def inst_convol_complete(model_out, wave_arr, inst_disp_wave):
    # Convert instrument resol dispersion (sigma) to FWHM
    fwhm_wave = inst_disp_wave*(2.*np.sqrt(2.*np.log(2.)))
    # FWHM in angstroms, matches units of xx = obslam_t

    try:
        # Will fail if inst_disp_wave = const wave dispersion / eg, not doing pos wave shift
        if len(fwhm_wave) > 0:
            conv_type = 'variable'
            inst_fwhm_wave_array = fwhm_wave
        else:
            # Parses case of const wave dispersion / float for inst_disp_wave
            conv_type = 'const'
    except:
        conv_type = 'const'

    # Make gaussian for convolution. Want peak centered on a pixel: xx needs to be odd numbered.
    if len(wave_arr) % 2 == 0:
        xx = wave_arr[:-1]
    else:
        xx = wave_arr
    lam_cent = np.average(xx)

    if conv_type == 'variable':
        return convol_inst(model_out, xx, lam_cent, inst_fwhm_wave_array)
    else:
        return convol_instconstfwhm(model_out, xx, lam_cent, fwhm_wave)


#@jit(nopython=True)
#@jit
@njit
def gaus_from_fwhm(x, FWHM, x0):
    """
    Quick function to return gaussian profile given array x,
    with the specified FWHM value.
    """
    sig = FWHM/(2.*np.sqrt(2.*np.log(2.)))
    y = np.exp(-1.*((x-x0)**2)/(2.*(sig**2)))

    # Scale the output as needed.
    return y


#@jit(nopython=True)
@njit
def sin_i(q, q0=None):
    if q < q0:
        sini = 1.
    else:
        sini = np.sqrt((1.-q**2)/(1.-q0**2))

    return sini



#
#
def lmfit_gaussian(x_arr, y_arr, y_err=None):
    params = lmfit.Parameters()
    params.add('mu', value=x_arr[y_arr.argmax()], vary=True)
    params.add('A', value=y_arr.max(), vary=True)
    params.add('FWHM', value=(x_arr.max()-x_arr.min())/10., vary=True, min=0.)

    result = lmfit.minimize(gaus_residual_FWHM, params,
                args=(x_arr, y_arr, y_err))

    return result.params['A'].value, result.params['mu'].value, \
            result.params['FWHM'].value

#
def gaus_residual_FWHM(params, x_arr, y_arr, y_err):
    if y_err is None:
        y_err = np.repeat(0.01*y_arr.max(), len(y_arr))

    model = params['A'].value*gaus_from_fwhm(x_arr,
                    params['FWHM'].value, params['mu'].value)
    resid = (model-y_arr)/y_err
    return resid



########################################################
########################################################

def sigma_e_dispersion(aperModel1DDisp = None, re_arcsec=None, re_mass_arcsec=None, r_outer=None):
    # To integrate to re_arcsec: set r_outer = re_arcsec

    wrapped_sigsq_I = func_wrapper_sigsq_I(re_arcsec, aperModel1DDisp.galaxy.n,
                aperModel1DDisp.Ie, d=aperModel1DDisp.d, r_core=aperModel1DDisp.r_core, re_mass_arcsec=re_mass_arcsec)
    wrapped_I = func_wrapper_I(re_arcsec, aperModel1DDisp.galaxy.n, aperModel1DDisp.Ie,
            r_core=aperModel1DDisp.r_core)
    sigsq_I_int = integrate.quad(wrapped_sigsq_I, 0, r_outer)
    I_int = integrate.quad(wrapped_I, 0, r_outer)

    sigma_e = np.sqrt(sigsq_I_int[0]/I_int[0])
    return sigma_e


###########################################################################
# Useful for intrinsic integration calculation:
def func_wrapper_sigsq_I(*args, **kwargs):
    def func(x):
        return sigsq_I_circ_nosee_integrand(x, *args, **kwargs)
    return func


def func_wrapper_I(*args, **kwargs):
    def func(x):
        return I_circ_nosee_integrand(x, *args, **kwargs)
    return func

def sigsq_I_circ_nosee_integrand(r, re, n, Ie, d=-0.089, r_core=1/300., re_mass_arcsec=None):
    return (sigma_profile(r,re_mass_arcsec, d=d, r_core=r_core)**2)*\
                I_sersic(r, re, n, Ie, r_core=r_core)* 2*np.pi*r

def I_circ_nosee_integrand(r, re, n, Ie, r_core=1/300.):
    return I_sersic(r, re, n, Ie, r_core=r_core)* 2*np.pi*r




########################################################
########################################################
def sigma_aper_dispersion(aperModel1DDisp = None, re_arcsec=None, re_mass_arcsec=None):

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # re_arcsec should already be circularized -- use q=1.
    sig_sq_wide = disp_square(aperModel1DDisp.x_arr_full, aperModel1DDisp.y_arr_full,
                    q=1., delt_PA=0., re_mass_arcsec=re_mass_arcsec,
                    r_core=aperModel1DDisp.r_core)

    I_wide  = Igal(aperModel1DDisp.x_arr_full, aperModel1DDisp.y_arr_full, re_arcsec=re_arcsec,
                        n=aperModel1DDisp.galaxy.n, q=1., delt_PA=0.,r_core=aperModel1DDisp.r_core)

    sigsq_I_wide = sig_sq_wide*I_wide

    ## All pre-seeing!!!


    # -----------------------------------------------
    # Get covoluation array:
    # seeing_wide = seeing_stamp_wide(aperModel1DDisp.nPixels,
    #                         aperModel1DDisp.x_arr_full,
    #                         aperModel1DDisp.y_arr_full,
    #                         aperModel1DDisp.instrument.PSF)
    seeing_wide = aperModel1DDisp.instrument.PSF._conv_stamp

    # plt.imshow(seeing_wide[n_pix+1:3*n_pix+2,n_pix+1:3*n_pix+2], origin='lower',
    #     interpolation='none', aspect=float(delt_y/delt_x))


    # Convolve arrays with seeing:
    if aperModel1DDisp.instrument.PSF.PSF_FWHM > 0.:
        sigsq_I_conv_wide = fftconvolve(sigsq_I_wide, seeing_wide, mode='same')
        I_conv_wide = fftconvolve(I_wide, seeing_wide, mode='same')
    else:
        sigsq_I_conv_wide = sigsq_I_wide
        I_conv_wide = I_wide


    # -----------------------------------------------
    # Now trim these down to the proper size, getting rid of edge effects:

    xtrim = [aperModel1DDisp.nPixels+1, 3*aperModel1DDisp.nPixels+2]
    ytrim = [aperModel1DDisp.nPixels+1, 3*aperModel1DDisp.nPixels+2]

    sigsq_I_conv = sigsq_I_conv_wide[ytrim[0]:ytrim[1],xtrim[0]:xtrim[1]]
    # should have shape (2*n_pix, 2*n_pix)
    I_conv = I_conv_wide[ytrim[0]:ytrim[1],xtrim[0]:xtrim[1]]






    # -----------------------------------------------
    # Sum to find sigma_sq in aperture:
    if aperModel1DDisp.extraction_method == 'optimal':
        prof_sigma = (0.5*aperModel1DDisp.extraction_width/aperModel1DDisp.delt_y)/\
                    (2.*np.sqrt(2.*np.log(2.)))
        # y extent = number of ROWS - shape[0]
        yy_prof = np.array(six.moves.xrange(I_conv.shape[0])) - (I_conv.shape[0]-1.)/2.

        g_y = norm.pdf(yy_prof, 0., prof_sigma)
        g_y = g_y/np.sum(g_y)  # normalize

        G_y = np.tile(np.array([g_y]).transpose(), (1, I_conv.shape[1]))
    elif aperModel1DDisp.extraction_method == 'boxcar':
        G_y = np.ones(I_conv.shape)
    else:
        raise ValueError('extraction method not recognized: '+aperModel1DDisp.extraction_method)

    sigmasq_aper = np.sum(sigsq_I_conv*G_y*aperModel1DDisp.delt_x*aperModel1DDisp.delt_y)/\
                    np.sum(I_conv*G_y*aperModel1DDisp.delt_x*aperModel1DDisp.delt_y)

    sigma_aper = np.sqrt(sigmasq_aper)

    return sigma_aper



#
########################################################
########################################################
def sigma_aper_dispersion_misalign(aperModel1DDisp = None,
            re_arcsec=None, re_mass_arcsec=None):
    # Change 2022-04-05: Do direct calculation, with rotation already included in
    #                    the coordinates, instead of rotating arrays.

    sig_sq_wide = disp_square(aperModel1DDisp.x_arr_full, aperModel1DDisp.y_arr_full,
                    q=aperModel1DDisp.galaxy.q, delt_PA=aperModel1DDisp.galaxy.delt_PA,
                    re_mass_arcsec=re_mass_arcsec,
                    r_core=aperModel1DDisp.r_core)


    I_wide  = Igal(aperModel1DDisp.x_arr_full, aperModel1DDisp.y_arr_full,
                        q=aperModel1DDisp.galaxy.q, delt_PA=aperModel1DDisp.galaxy.delt_PA,
                        re_arcsec=re_arcsec, n=aperModel1DDisp.galaxy.n,
                        r_core=aperModel1DDisp.r_core)

    ## Mask any NaN values with zero:
    #I_wide[~np.isfinite(I_wide)] = 0.
    #sig_sq_wide[~np.isfinite(sig_sq_wide)] = 0.
    sigsq_I_wide = sig_sq_wide*I_wide

    # -----------------------------------------------
    # Convolve arrays with seeing:
    if aperModel1DDisp.instrument.PSF.PSF_FWHM > 0.:
        # Get covoluation array:
        # seeing_wide = PSF_stamp_multid(aperModel1DDisp.xp_arr_full,
        #             aperModel1DDisp.yp_arr_full,
        #             aperModel1DDisp.instrument.PSF)
        seeing_wide = aperModel1DDisp.instrument.PSF._conv_stamp
        sigsq_I_conv_wide = fftconvolve(sigsq_I_wide, seeing_wide, mode='same')
        I_conv_wide = fftconvolve(I_wide, seeing_wide, mode='same')
    else:
        sigsq_I_conv_wide = sigsq_I_wide
        I_conv_wide = I_wide

    # ## -----------------------------------------------
    # interp_order = 5
    #
    # # ROTATE arrays:
    # if np.abs(aperModel1DDisp.galaxy.delt_PA) > 0.:
    #     I_conv_rot = skimagerotate(I_conv_wide, aperModel1DDisp.galaxy.delt_PA,
    #                             order=interp_order,
    #                             cval=0.,preserve_range=True)
    #     sigsq_I_conv_rot = skimagerotate(sigsq_I_conv_wide, aperModel1DDisp.galaxy.delt_PA,
    #                             order=interp_order,
    #                             cval=0.,preserve_range=True)
    #
    # else:
    #     I_conv_rot = I_conv_wide.copy()
    #     sigsq_I_conv_rot = sigsq_I_conv_wide.copy()
    # # has shape nYp, nXp (square, based on y_aper_half)
    #
    #
    # ## -----------------------------------------------
    # # Map coordinates onto gridsize where we can do easy trimming to get rid of padding:
    # # delt_y should be same as delt_yp, as original sizes were chosen based on y_aper_half.
    # # choose new delt_x very close to delt_xp, to avoid oversampling.
    # # But needs to satisfy delt_x = x_aper_half/(n_pix_x+0.5)
    # # Should be n_pix_x+0.5 across x_aper_half
    # aperModel1DDisp.n_pix_x = np.int(np.round((0.5*aperModel1DDisp.x_aper/\
    #                         aperModel1DDisp.delt_xp) - 0.5))
    # aperModel1DDisp.delt_x = (0.5*aperModel1DDisp.x_aper)/(aperModel1DDisp.n_pix_x+0.5)
    # #               calculate the new value of delt_x
    # aperModel1DDisp.padX = aperModel1DDisp.n_pix_x+1
    # aperModel1DDisp.nX = 2*(aperModel1DDisp.n_pix_x + aperModel1DDisp.padX) + 1
    #
    # aperModel1DDisp.n_pix_y = aperModel1DDisp.nPixels
    # aperModel1DDisp.padY = aperModel1DDisp.padYp
    # aperModel1DDisp.delt_y = aperModel1DDisp.delt_yp
    # aperModel1DDisp.nY = aperModel1DDisp.nYp
    #
    # # Need to define new full arrays:
    # # coordinates at each point:
    # # coordinates at each point:
    # aperModel1DDisp.x_arr = np.linspace(-(aperModel1DDisp.n_pix_x+aperModel1DDisp.padX)*\
    #             aperModel1DDisp.delt_x,
    #                 (aperModel1DDisp.n_pix_x+aperModel1DDisp.padX)*aperModel1DDisp.delt_x,
    #                 num=aperModel1DDisp.nX)
    # aperModel1DDisp.y_arr = np.linspace(-(aperModel1DDisp.n_pix_y+aperModel1DDisp.padY)*\
    #             aperModel1DDisp.delt_y,
    #                 (aperModel1DDisp.n_pix_y+aperModel1DDisp.padY)*aperModel1DDisp.delt_y,
    #                 num=aperModel1DDisp.nY)
    # # x_arr, y_arr are in *ARCSECONDS*
    #
    #
    # # Need to make map coords be in terms of index rel to original array:
    # #   ie (float) index of x_arr in xp_arr coordinates.
    # aperModel1DDisp.x_coord = (aperModel1DDisp.x_arr.min()-aperModel1DDisp.xp_arr.min())/\
    #             aperModel1DDisp.delt_xp + \
    #                 np.linspace(0,aperModel1DDisp.nX-1,num=aperModel1DDisp.nX)*\
    #                     aperModel1DDisp.delt_x/aperModel1DDisp.delt_xp
    # aperModel1DDisp.y_coord = np.linspace(0,aperModel1DDisp.nY-1,num=aperModel1DDisp.nY)
    #
    #
    # aperModel1DDisp.x_coord_full = np.tile(aperModel1DDisp.x_coord, (aperModel1DDisp.nY,1))
    # aperModel1DDisp.y_coord_full = np.tile(np.array([aperModel1DDisp.y_coord]).T,
    #                             (1,aperModel1DDisp.nX))
    # aperModel1DDisp.xcoordfull_flat = aperModel1DDisp.x_coord_full.ravel()
    # aperModel1DDisp.ycoordfull_flat = aperModel1DDisp.y_coord_full.ravel()
    #
    # # coordinates = [rowcoords,  colcoords]
    # I_conv_map = scipymapcoords(I_conv_rot, [aperModel1DDisp.ycoordfull_flat,
    #             aperModel1DDisp.xcoordfull_flat],
    #                 order=interp_order)
    # sigsq_I_conv_map = scipymapcoords(sigsq_I_conv_rot, [aperModel1DDisp.ycoordfull_flat,
    #             aperModel1DDisp.xcoordfull_flat],
    #                 order=interp_order)
    #
    # I_conv_map = I_conv_map.reshape((aperModel1DDisp.nY,aperModel1DDisp.nX))
    # sigsq_I_conv_map = sigsq_I_conv_map.reshape((aperModel1DDisp.nY,aperModel1DDisp.nX))





    aperModel1DDisp.n_pix_x = np.int(np.round((0.5*aperModel1DDisp.x_aper/\
                            aperModel1DDisp.delt_x) - 0.5))
    aperModel1DDisp.delt_x = (0.5*aperModel1DDisp.x_aper)/(aperModel1DDisp.n_pix_x+0.5)
    #               calculate the new value of delt_x
    aperModel1DDisp.padX = aperModel1DDisp.n_pix_x+1
    aperModel1DDisp.nX = 2*(aperModel1DDisp.n_pix_x + aperModel1DDisp.padX) + 1

    aperModel1DDisp.n_pix_y = aperModel1DDisp.nPixels
    # aperModel1DDisp.padY = aperModel1DDisp.padYp
    # aperModel1DDisp.delt_y = aperModel1DDisp.delt_yp
    # aperModel1DDisp.nY = aperModel1DDisp.nYp



    ## -----------------------------------------------
    # Now trim these down to the proper size, getting rid of edge effects:
    # Trim in x, y:
    xtrim = [aperModel1DDisp.padX,2*aperModel1DDisp.n_pix_x+aperModel1DDisp.padX+1]
    ytrim = [aperModel1DDisp.padY,2*aperModel1DDisp.n_pix_y+aperModel1DDisp.padY+1]

    sigsq_I_conv = sigsq_I_conv_wide[ytrim[0]:ytrim[1],xtrim[0]:xtrim[1]]
    I_conv = I_conv_wide[ytrim[0]:ytrim[1],xtrim[0]:xtrim[1]]


    # -----------------------------------------------
    # Sum to find sigma_sq in aperture:
    if aperModel1DDisp.extraction_method == 'optimal':
        prof_sigma = (0.5*aperModel1DDisp.extraction_width/aperModel1DDisp.delt_y)/\
                    (2.*np.sqrt(2.*np.log(2.)))
        # y extent = number of ROWS - shape[0]
        yy_prof = np.array(six.moves.xrange(I_conv.shape[0])) - (I_conv.shape[0]-1.)/2.

        g_y = norm.pdf(yy_prof, 0., prof_sigma)
        g_y = g_y/np.sum(g_y)  # normalize

        G_y = np.tile(np.array([g_y]).transpose(), (1, I_conv.shape[1]))
    elif aperModel1DDisp.extraction_method == 'boxcar':
        G_y = np.ones(I_conv.shape)
    else:
        raise ValueError('Extraction method not currently supported: '+aperModel1DDisp.extraction_method)

    sigmasq_aper = np.sum(sigsq_I_conv*G_y*aperModel1DDisp.delt_x*aperModel1DDisp.delt_y)/\
                    np.sum(I_conv*G_y*aperModel1DDisp.delt_x*aperModel1DDisp.delt_y)
    sigma_aper = np.sqrt(sigmasq_aper)


    # -----------------------------------------------
    # Inclination is also included in sigma_e, so sigma_aper/sigma_e
    #       only has the *ROTATION* part depend on inclination

    return sigma_aper



########################################################
########################################################
def sigma_aper_dispersion_misalign_ORIG(aperModel1DDisp = None, re_arcsec=None, re_mass_arcsec=None):
    sig_sq_wide = disp_square(aperModel1DDisp.xp_arr_full, aperModel1DDisp.yp_arr_full,
                    q=aperModel1DDisp.galaxy.q, delt_PA=aperModel1DDisp.delt_PAp,
                    re_mass_arcsec=re_mass_arcsec,
                    r_core=aperModel1DDisp.r_core)


    I_wide  = Igal(aperModel1DDisp.xp_arr_full, aperModel1DDisp.yp_arr_full,
                        q=aperModel1DDisp.galaxy.q, delt_PA=aperModel1DDisp.delt_PAp,
                        re_arcsec=re_arcsec, n=aperModel1DDisp.galaxy.n,
                        r_core=aperModel1DDisp.r_core)

    ## Mask any NaN values with zero:
    #I_wide[~np.isfinite(I_wide)] = 0.
    #sig_sq_wide[~np.isfinite(sig_sq_wide)] = 0.
    sigsq_I_wide = sig_sq_wide*I_wide

    # -----------------------------------------------
    # Convolve arrays with seeing:
    if aperModel1DDisp.instrument.PSF.PSF_FWHM > 0.:
        # Get covoluation array:
        # seeing_wide = PSF_stamp_multid(aperModel1DDisp.xp_arr_full,
        #             aperModel1DDisp.yp_arr_full,
        #             aperModel1DDisp.instrument.PSF)
        seeing_wide = aperModel1DDisp.instrument.PSF._conv_stamp
        sigsq_I_conv_wide = fftconvolve(sigsq_I_wide, seeing_wide, mode='same')
        I_conv_wide = fftconvolve(I_wide, seeing_wide, mode='same')
    else:
        sigsq_I_conv_wide = sigsq_I_wide
        I_conv_wide = I_wide

    ## -----------------------------------------------
    interp_order = 5

    # ROTATE arrays:
    if np.abs(aperModel1DDisp.galaxy.delt_PA) > 0.:
        I_conv_rot = skimagerotate(I_conv_wide, aperModel1DDisp.galaxy.delt_PA,
                                order=interp_order,
                                cval=0.,preserve_range=True)
        sigsq_I_conv_rot = skimagerotate(sigsq_I_conv_wide, aperModel1DDisp.galaxy.delt_PA,
                                order=interp_order,
                                cval=0.,preserve_range=True)

    else:
        I_conv_rot = I_conv_wide.copy()
        sigsq_I_conv_rot = sigsq_I_conv_wide.copy()
    # has shape nYp, nXp (square, based on y_aper_half)


    ## -----------------------------------------------
    # Map coordinates onto gridsize where we can do easy trimming to get rid of padding:
    # delt_y should be same as delt_yp, as original sizes were chosen based on y_aper_half.
    # choose new delt_x very close to delt_xp, to avoid oversampling.
    # But needs to satisfy delt_x = x_aper_half/(n_pix_x+0.5)
    # Should be n_pix_x+0.5 across x_aper_half
    aperModel1DDisp.n_pix_x = np.int(np.round((0.5*aperModel1DDisp.x_aper/\
                            aperModel1DDisp.delt_xp) - 0.5))
    aperModel1DDisp.delt_x = (0.5*aperModel1DDisp.x_aper)/(aperModel1DDisp.n_pix_x+0.5)
    #               calculate the new value of delt_x
    aperModel1DDisp.padX = aperModel1DDisp.n_pix_x+1
    aperModel1DDisp.nX = 2*(aperModel1DDisp.n_pix_x + aperModel1DDisp.padX) + 1

    aperModel1DDisp.n_pix_y = aperModel1DDisp.nPixels
    aperModel1DDisp.padY = aperModel1DDisp.padYp
    aperModel1DDisp.delt_y = aperModel1DDisp.delt_yp
    aperModel1DDisp.nY = aperModel1DDisp.nYp

    # Need to define new full arrays:
    # coordinates at each point:
    # coordinates at each point:
    aperModel1DDisp.x_arr = np.linspace(-(aperModel1DDisp.n_pix_x+aperModel1DDisp.padX)*\
                aperModel1DDisp.delt_x,
                    (aperModel1DDisp.n_pix_x+aperModel1DDisp.padX)*aperModel1DDisp.delt_x,
                    num=aperModel1DDisp.nX)
    aperModel1DDisp.y_arr = np.linspace(-(aperModel1DDisp.n_pix_y+aperModel1DDisp.padY)*\
                aperModel1DDisp.delt_y,
                    (aperModel1DDisp.n_pix_y+aperModel1DDisp.padY)*aperModel1DDisp.delt_y,
                    num=aperModel1DDisp.nY)
    # x_arr, y_arr are in *ARCSECONDS*


    # Need to make map coords be in terms of index rel to original array:
    #   ie (float) index of x_arr in xp_arr coordinates.
    aperModel1DDisp.x_coord = (aperModel1DDisp.x_arr.min()-aperModel1DDisp.xp_arr.min())/\
                aperModel1DDisp.delt_xp + \
                    np.linspace(0,aperModel1DDisp.nX-1,num=aperModel1DDisp.nX)*\
                        aperModel1DDisp.delt_x/aperModel1DDisp.delt_xp
    aperModel1DDisp.y_coord = np.linspace(0,aperModel1DDisp.nY-1,num=aperModel1DDisp.nY)


    aperModel1DDisp.x_coord_full = np.tile(aperModel1DDisp.x_coord, (aperModel1DDisp.nY,1))
    aperModel1DDisp.y_coord_full = np.tile(np.array([aperModel1DDisp.y_coord]).T,
                                (1,aperModel1DDisp.nX))
    aperModel1DDisp.xcoordfull_flat = aperModel1DDisp.x_coord_full.ravel()
    aperModel1DDisp.ycoordfull_flat = aperModel1DDisp.y_coord_full.ravel()

    # coordinates = [rowcoords,  colcoords]
    I_conv_map = scipymapcoords(I_conv_rot, [aperModel1DDisp.ycoordfull_flat,
                aperModel1DDisp.xcoordfull_flat],
                    order=interp_order)
    sigsq_I_conv_map = scipymapcoords(sigsq_I_conv_rot, [aperModel1DDisp.ycoordfull_flat,
                aperModel1DDisp.xcoordfull_flat],
                    order=interp_order)

    I_conv_map = I_conv_map.reshape((aperModel1DDisp.nY,aperModel1DDisp.nX))
    sigsq_I_conv_map = sigsq_I_conv_map.reshape((aperModel1DDisp.nY,aperModel1DDisp.nX))



    ## -----------------------------------------------
    # Now trim these down to the proper size, getting rid of edge effects:
    # Trim in x, y:
    xtrim = [aperModel1DDisp.padX,2*aperModel1DDisp.n_pix_x+aperModel1DDisp.padX+1]
    ytrim = [aperModel1DDisp.padY,2*aperModel1DDisp.n_pix_y+aperModel1DDisp.padY+1]

    sigsq_I_conv = sigsq_I_conv_map[ytrim[0]:ytrim[1],xtrim[0]:xtrim[1]]
    I_conv = I_conv_map[ytrim[0]:ytrim[1],xtrim[0]:xtrim[1]]


    # -----------------------------------------------
    # Sum to find sigma_sq in aperture:
    if aperModel1DDisp.extraction_method == 'optimal':
        prof_sigma = (0.5*aperModel1DDisp.extraction_width/aperModel1DDisp.delt_y)/\
                    (2.*np.sqrt(2.*np.log(2.)))
        # y extent = number of ROWS - shape[0]
        yy_prof = np.array(six.moves.xrange(I_conv.shape[0])) - (I_conv.shape[0]-1.)/2.

        g_y = norm.pdf(yy_prof, 0., prof_sigma)
        g_y = g_y/np.sum(g_y)  # normalize

        G_y = np.tile(np.array([g_y]).transpose(), (1, I_conv.shape[1]))
    elif aperModel1DDisp.extraction_method == 'boxcar':
        G_y = np.ones(I_conv.shape)
    else:
        raise ValueError('Extraction method not currently supported: '+aperModel1DDisp.extraction_method)

    sigmasq_aper = np.sum(sigsq_I_conv*G_y*aperModel1DDisp.delt_x*aperModel1DDisp.delt_y)/\
                    np.sum(I_conv*G_y*aperModel1DDisp.delt_x*aperModel1DDisp.delt_y)
    sigma_aper = np.sqrt(sigmasq_aper)


    # -----------------------------------------------
    # Inclination is also included in sigma_e, so sigma_aper/sigma_e
    #       only has the *ROTATION* part depend on inclination

    return sigma_aper

#@jit
@njit
def disp_square(x, y, q=None, delt_PA=None, re_mass_arcsec=None, r_core=None):

    r = r_int_ellip(x, y, q, delt_PA)

    # r is relative to r_circularized for that given ellipse -- use re_circ.
    # INPUT RE_CIRC!
    sigma_val = sigma_profile(r, re_mass_arcsec, r_core=r_core)

    return sigma_val**2


#@jit(nopython=True)
#@jit
@njit
def r_int_ellip(x, y, q, delt_PA):


    yp = np.sin(delt_PA*deg2rad)*x + np.cos(delt_PA*deg2rad)*y
    xp = np.cos(delt_PA*deg2rad)*x - np.sin(delt_PA*deg2rad)*y

    # Set all r along ellipse to r_circ
    r_int_ell = np.sqrt(q * (yp**2) + 1./q * (xp**2))

    return r_int_ell


#

###########################################################################


#@jit
#@jit(nopython=True)
@njit
def sigma_profile(r, re, d=-0.089, r_core=1/300.):
    # Use values from van de Sande + 2013
    #d = -0.089
    r_core_val = r_core*re #1/30.*re

    # Set constant so that sigma_profile(re) = 1.
    #sigma_cnst = 1.
    sigma_cnst = 1./np.power(((re+r_core_val)/re), d)

    return sigma_cnst * np.power( ((r+r_core_val)/re), d )

@njit
def Igal(x, y, re_arcsec=None, n=None, q=None, delt_PA=None,r_core=None):
    r = r_int_ellip(x, y, q, delt_PA)

    # Make sure re_arcsec is re_arcsec_circ!!
    Ie = 1.  # constant factor placeholder
    return I_sersic(r, re_arcsec, n, Ie, r_core=r_core)

#@jit
#@jit(nopython=True)
@njit
def I_sersic(r, re, n, Ie, r_core=1/300.):
    # Sersic surface intensity profile
    b_n = 2.*n - 0.324      # Ciotti 1991, valid 0.5 <= n <= 10
                            # Ciotti+Bertin99


    r_core_val = r_core*re #1/30.*re

    return Ie*np.exp(-b_n*( (np.power( ((r + r_core_val)/re), (1./np.float(n)) )) - 1.) )
