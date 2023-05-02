# misfit/utils.py
# Utilities for MISFIT
#
# Copyright 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Hidden modules prepended with '_'
from __future__ import print_function

import numpy as np
import os
import sys

try:
    import general_utils as utils
    import io
except:
    from . import general_utils as utils
    from . import io

# import six

import lmfit

def trim_spectrum_1D(spec1D, galaxy, trim_restwave_range=None,
                trim_obswave_range=None,
                param_restwave_filename=None, linename=None):
    """
    Method to trim the 1D spectrum to a given wavelength range.
    Either input: 
    trim_restwave_range (and have galaxy.z set) to do in restframe, or 
    trim_obsframe_range to trim in observed frame, or 
    param_restwave_filename and linename  to read the set from a parameters file.
    """

    if (galaxy.z is None) and (trim_obswave_range is None):
        print("Must set redshift galaxy.z first")
        return None

    if (trim_restwave_range is None) and (trim_obswave_range is None):
        if param_restwave_filename is None:
            d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
            param_restwave_filename = os.path.join(d, 'trim_wavelengths_1D.param')

        # Read in the trimming stuff for this line:
        trim_restwave_range = io.read_wave_range(param_restwave_filename,linename=linename)

    if trim_obswave_range is None:
        trim_obswave_range = np.array(trim_restwave_range)*(1.+galaxy.z).tolist()

    # Find the range there is coverage:
    if spec1D is not None:
        spec1D_trim = spec1D.copy()
        spec1D_trim.trim_obswave_range = trim_obswave_range
        spec1D_trim.trim_restwave_range = trim_restwave_range

        wh = np.where((spec1D.obswave >= trim_obswave_range[0]) & \
                (spec1D.obswave <= trim_obswave_range[1]))[0]

        if len(wh) > 0:
            spec1D_trim.wave = spec1D.wave[wh]
            spec1D_trim.flux = spec1D.flux[wh]
            spec1D_trim.flux_err = spec1D.flux_err[wh]
            spec1D_trim.spec_mask = spec1D.spec_mask[wh]

            spec1D_trim.calculate_restwave(galaxy)
        else:
            spec1D_trim.wave = None
            spec1D_trim.flux = None
            spec1D_trim.flux_err = None
            spec1D_trim.spec_mask = None
    else:
        spec1D_trim = None

    return spec1D_trim

#
def trim_spectrum_2D_wavelength(spec2D, galaxy,
            trim_restwave_range=None,
            trim_obswave_range=None,
            param_restwave_filename=None,
            linename=None, inplace=True):
    """
    Method to trim the 2D spectrum to a given wavelength range.
    Either input: 
    trim_restwave_range (and have galaxy.z set) to do in restframe, or 
    trim_obsframe_range to trim in observed frame, or 
    param_restwave_filename and linename  to read the set from a parameters file.
    """

    if (trim_restwave_range is None) & (trim_obswave_range is None) & ((param_restwave_filename is None) & (linename is None)):
        raise ValueError("Must set one of (a) trim_restwave_range, (b) trim_obswave_range, or (c) param_restwave_filename + linename")


    if (trim_restwave_range is not None) & (trim_obswave_range is not None):
        raise ValueError("Must only call with one of trim_restwave_range or trim_obswave_range!")

    if (trim_restwave_range is not None):
        trim_obswave_range = (np.array(trim_restwave_range)*(1.+galaxy.z)).tolist()
    elif (trim_obswave_range is not None):
        trim_restwave_range = (np.array(trim_obswave_range)/(1.+galaxy.z)).tolist()
    else:
        if param_restwave_filename is None:
            d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
            param_restwave_filename = os.path.join(d, 'line_range_restwave_2D.param')
        # Read in the trimming stuff for this line:
        trim_restwave_range = io.read_wave_range(param_restwave_filename,linename=linename)
        trim_obswave_range = (np.array(trim_restwave_range)*(1.+galaxy.z)).tolist()

    #
    # Find the range there is coverage:
    if inplace:
        if spec2D is not None:
            spec2D.trim_obswave_range = trim_obswave_range
            spec2D.trim_restwave_range = trim_restwave_range

            wh = np.where((spec2D.obswave >= trim_obswave_range[0]) & \
                    (spec2D.obswave <= trim_obswave_range[1]))[0]

            if len(wh) > 0:
                spec2D.wave = spec2D.wave[wh]
                spec2D.flux = spec2D.flux[:,wh]
                spec2D.flux_err = spec2D.flux_err[:,wh]
                spec2D.spec_mask = spec2D.spec_mask[:,wh]
                spec2D.spec_weight = spec2D.spec_weight[:,wh]

                try:
                    spec2D.flux_incl_cont = spec2D.flux_incl_cont[:,wh]
                    spec2D.cont_model = spec2D.cont_model[:,wh]
                except:
                    spec2D.flux_incl_cont = None
                    spec2D.cont_model = None

                spec2D.calculate_restwave(galaxy)
            else:
                spec2D.wave = None
                spec2D.flux = None
                spec2D.flux_err = None
                spec2D.spec_mask = None
                spec2D.spec_weight = None

        return spec2D
    else:

        if spec2D is not None:

            spec2D_trim = spec2D.copy()
            spec2D_trim.trim_obswave_range = trim_obswave_range
            spec2D_trim.trim_restwave_range = trim_restwave_range

            wh = np.where((spec2D_trim.obswave >= trim_obswave_range[0]) & \
                    (spec2D_trim.obswave <= trim_obswave_range[1]))[0]

            if len(wh) > 0:
                spec2D_trim.wave = spec2D.wave[wh]
                spec2D_trim.flux = spec2D.flux[:,wh]
                spec2D_trim.flux_err = spec2D.flux_err[:,wh]
                spec2D_trim.spec_mask = spec2D.spec_mask[:,wh]
                spec2D_trim.spec_weight = spec2D.spec_weight[:,wh]

                spec2D_trim.calculate_restwave(galaxy)
            else:
                spec2D_trim.wave = None
                spec2D_trim.flux = None
                spec2D_trim.flux_err = None
                spec2D_trim.spec_mask = None
                spec2D_trim.spec_weight = None
        else:
            spec2D_trim = None

        return spec2D_trim



#
def trim_spectrum_2D_spatial(spec2D, galaxy, instrument):
    """
    Method to trim the 2D spectrum to a given spatial range.
    Trim vertically: identify yposition and bounding pixel indices:
    """


    #####################
    # First identify peak positions:
    m0, m0_shift_fit, m0_peak_snr, m_min, m_max = vert_position(flux=spec2D.flux.copy()*spec2D.mask_sky.copy(),
                err=spec2D.flux_err.copy()*spec2D.mask_sky.copy(), obswave=spec2D.obswave.copy(),
                obswave_val=spec2D.restwave_arr[0]*(1.+galaxy.z),
                band=spec2D.band, ypos=spec2D.ypos, refit=False,
                wh_sky=spec2D.wh_sky, debug=galaxy.debug,
                skip_check=False, spatial_pixscale=instrument.pixscale,
                dither=galaxy.dither)

    #
    n_spatial_orig = np.shape(spec2D.flux)[0]   # Number of spatial pixels in the original image

    # Where to put top, bottom of trim: modified to cover cases where
    #       obj is near edge of image
    if galaxy.dither:
        l_ind = max([0,m_min])
        r_ind = min([n_spatial_orig-1, m_max])
    else:
        l_ind = 0
        r_ind = n_spatial_orig-1

    # Trim everything
    spec2D.flux = spec2D.flux[l_ind:r_ind,:]
    spec2D.flux_err = spec2D.flux_err[l_ind:r_ind,:]

    spec2D.flux_incl_cont = spec2D.flux_incl_cont[l_ind:r_ind,:]
    spec2D.cont_model = spec2D.cont_model[l_ind:r_ind,:]

    spec2D.spec_weight = spec2D.spec_weight[l_ind:r_ind,:]
    spec2D.spec_mask = spec2D.spec_mask[l_ind:r_ind,:]
    spec2D.mask_qual = spec2D.mask_qual[l_ind:r_ind,:]
    spec2D.mask_sky = spec2D.mask_sky[l_ind:r_ind,:]
    spec2D.mask_low_snr = spec2D.mask_low_snr[l_ind:r_ind,:]

    # Save the new spatial dimensions:
    spec2D.n_spatial = np.shape(spec2D.flux)[0]


    if galaxy.debug:
        print("spec2D.low_snr_row_inds=", spec2D.low_snr_row_inds)
    # Re-figure low S/N row indices directly:
    mask_low_snr_tmp, low_snr_row_inds = mask_low_snr_rows_2D(flux=spec2D.flux,
                        err=spec2D.flux_err, mask=spec2D.mask_qual.copy(),
                        mask_calc=spec2D.mask_sky,
                        snr_cut=spec2D.row_snr_cut, debug=galaxy.debug)
    spec2D.low_snr_row_inds = low_snr_row_inds
    if galaxy.debug:
        print("post trim spec2D.low_snr_row_inds=", spec2D.low_snr_row_inds)


    if (galaxy.spec1D is not None):
        if (galaxy.spec1D.extraction_spatial_profile is not None):
            galaxy.spec1D.extraction_spatial_profile = galaxy.spec1D.extraction_spatial_profile[l_ind:r_ind]


    # And return the bounds of this trimming:
    spec2D.full_m0 = m0
    spec2D.full_l_ind = l_ind
    spec2D.full_r_ind = r_ind


    ################################
    # Get the trimmed positions:
    spec2D.m0 = m0 - l_ind
    spec2D.m0_shift_fit = m0_shift_fit
    spec2D.m0_peak_snr = m0_peak_snr
    # It's already trimmed so the approp m_min and m_max are the edges.
    spec2D.m_lims = np.array([0, np.shape(spec2D.flux)[0]])

    # Offset ypos of the trimmed one:
    spec2D.ypos = spec2D.ypos - l_ind


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if galaxy.debug:
        import matplotlib.pyplot as _plt
        import matplotlib.gridspec as _gridspec

        fig = _plt.figure()
        fig.set_size_inches(8., 4.)
        gs1 = _gridspec.GridSpec(1, 1, height_ratios=[1])
        gs1.update(left=0.05, right=0.98, wspace=0.1)
        ax2 = _plt.subplot(gs1[0,0])

        #
        ax2.imshow(spec2D.flux*spec2D.spec_mask*spec2D.mask_sky, interpolation="None", origin='lower')

        _plt.show()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++


    return spec2D, galaxy

#
def subtract_continuum_2D(spec2D, galaxy, fill_value=-99., no_cont_subtraction=False):
    """
    Subtract continuum from 2D spectrum, using info in spec2D, galaxy (incl. galaxy.spec1D)
    """
    spec2D.spec_mask = spec_mask_missing(spec=spec2D.flux, spec_err=spec2D.flux_err,
                    spec_weight=spec2D.spec_weight, spec_mask=spec2D.spec_mask,
                    nonfinite_only=True)

    if not no_cont_subtraction:
        # Implement nonfinite to also include the *MISSING*, eg err=0 points, if err is not all 0.
        galaxy.spec1D.spec_mask = spec_mask_missing(spec=galaxy.spec1D.flux, spec_err=galaxy.spec1D.flux_err,
                                    spec_mask=galaxy.spec1D.spec_mask)
        galaxy.spec1D.num_mask_edge = 10 #5
        galaxy.spec1D.mask_edges()

        # Check

        spec2D.spec_mask = mask_spec2D_from_1D(mask2D=spec2D.spec_mask, mask1D=galaxy.spec1D.spec_mask)

    spec2D.flux = spec_nan_masking(flux=spec2D.flux, spec_mask=spec2D.spec_mask, fill_value=fill_value)
    spec2D.flux_err = spec_nan_masking(flux=spec2D.flux_err,
                            spec_mask=spec2D.spec_mask, fill_value=fill_value, mask_zero=True)

    ######################################
    # Calculate median sky level:
    sum_over_col_arr = np.sqrt(np.sum((spec2D.flux_err*spec2D.spec_mask)**2, axis=0))
    spec2D.median_flux_error = np.median(sum_over_col_arr)
    if galaxy.debug:
        print("spec2D.median_flux_error=", spec2D.median_flux_error)

    ############################################################################
    ############################################################################

    # Read in restframe ranges for fitting lines + continuum
    if (spec2D.wave_full_range_rest is None):
        if spec2D.wave_full_range_rest_param_filename is None:
            d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
            spec2D.wave_full_range_rest_param_filename = os.path.join(d, 'full_range_restwave_2D.param')
        spec2D.wave_full_range_rest = io.read_wave_range(spec2D.wave_full_range_rest_param_filename,
                        linename=spec2D.linenames_arr[0])

    # Read the wavelength range of just line + close neighbors:
    if (spec2D.wave_line_range_rest is None):
        if spec2D.wave_line_range_rest_param_filename is None:
            d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
            spec2D.wave_line_range_rest_param_filename = os.path.join(d, 'line_range_restwave_2D.param')
        spec2D.wave_line_range_rest = io.read_wave_range(spec2D.wave_line_range_rest_param_filename,
                        linename=spec2D.linenames_arr[0])

    # Read the wavelength range to mask:
    if (spec2D.mask_lines_wave_range_rest is None):
        if spec2D.mask_lines_wave_range_rest_param_filename is None:
            d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
            spec2D.mask_lines_wave_range_rest_param_filename = os.path.join(d, 'mask_lines_range_restwave_2D.param')
        spec2D.mask_lines_wave_range_rest = io.read_wave_range(spec2D.mask_lines_wave_range_rest_param_filename,
                    linename=spec2D.linenames_arr[0])

    ############################################################################
    ############################################################################


    ######################################
    # Prepare for continuum subtraction on spec2D:
    x_fit_plot = spec2D.wave.copy()

    spec2D.flux_incl_cont = spec2D.flux.copy()
    spec2D.cont_model = np.zeros(spec2D.flux.shape)

    ######################################
    # Determine fitting ranges:
    frange = spec2D.wave_full_range_rest*(1.+galaxy.z)
    line_mask_range = spec2D.mask_lines_wave_range_rest*(1.+galaxy.z)

    # Select the left, right portions of continuum; full range
    wh_line_l = np.where((spec2D.obswave >= frange[0])&(spec2D.obswave < line_mask_range[0]))[0]
    wh_line_r = np.where((spec2D.obswave <= frange[1])&(spec2D.obswave > line_mask_range[1]))[0]
    wh_full = np.where((spec2D.obswave >= frange[0])&(spec2D.obswave <= frange[1]))[0]

    if not no_cont_subtraction:
        ############################################################################
        # Do continuum fit of 1D spectrum

        ########## 1-deg poly prep:
        # Lam of where to fit
        ## Current: fits across skylines
        xx_poly = np.append(spec2D.obswave[wh_line_l], spec2D.obswave[wh_line_r])

        if galaxy.debug:
            import matplotlib.pyplot as _plt
            import matplotlib.gridspec as _gridspec

            # Debug:
            # line_mask_range #
            wh_line_mask_range = np.where((spec2D.obswave >= line_mask_range[0])&(spec2D.obswave<= line_mask_range[1]))[0]
            #_plt.plot(spec2D.obswave[wh_line_mask_range], galaxy.spec1D.flux_err[wh_line_mask_range], 'go-')
            wh_not_missing = np.where(galaxy.spec1D.flux_err > fill_value)[0]
            _plt.plot(galaxy.spec1D.obswave[wh_not_missing], galaxy.spec1D.flux_err[wh_not_missing], 'go-')
            _plt.gca().axhline(y=np.median(galaxy.spec1D.flux_err[wh_not_missing]), ls='-', color='teal')
            _plt.gca().axvline(x=line_mask_range[0], ls='--', color='black')
            _plt.gca().axvline(x=line_mask_range[1], ls='--', color='black')

            _plt.show()

        ######################################
        # Fix slope by fitting to continuum of 1D spectrum:

        # From error variance sum
        sum_over_col_arr = np.sqrt((galaxy.spec1D.flux_err*galaxy.spec1D.spec_mask)**2)
        wh_nosky, wh_sky = wh_skylines_1D(sum_over_col_arr,
                            median_err=spec2D.median_flux_error,
                            cutoff=spec2D.band_cutoff, full=True)

        spec_mask_tmp = galaxy.spec1D.spec_mask[:]
        spec_mask_tmp[wh_sky] = 0.


        # Missing other side: only do const cont. subtraction
        if (len(wh_line_l) > 0.) & (len(wh_line_r) > 0.):
            if (spec_mask_tmp[wh_line_l].max() > 0.) & (spec_mask_tmp[wh_line_r].max() > 0.):
                params_1d = lmfit.Parameters()
                params_1d.add('a', value=0.)
                params_1d.add('b', value=0.)
            else:
                params_1d = lmfit.Parameters()
                params_1d.add('a', value=0., vary=False)
                params_1d.add('b', value=0.)
        else:
            params_1d = lmfit.Parameters()
            params_1d.add('a', value=0., vary=False)
            params_1d.add('b', value=0.)



        # Original ranges:
        xx1d = xx_poly.copy()
        yy1d = np.append(galaxy.spec1D.flux[wh_line_l], galaxy.spec1D.flux[wh_line_r])
        yy1d_err = np.append(galaxy.spec1D.flux_err[wh_line_l], galaxy.spec1D.flux_err[wh_line_r])
        yy1d_mask = np.append(spec_mask_tmp[wh_line_l], spec_mask_tmp[wh_line_r])

        result_1d = lmfit.minimize(utils.weighted_linear_residual, params_1d,
                        args=(xx1d, yy1d, yy1d_err, yy1d_mask))
        slope = result_1d.params['a'].value


        if galaxy.debug:
            print("% image.cont_sub():  Doing debug!")

            print("1D continuum fit results:")
            print(lmfit.fit_report(result_1d.params))

            fig = _plt.figure()
            fig.set_size_inches(12.,4.)
            gs1 = _gridspec.GridSpec(1,2, wspace=0.1)
            ax = _plt.subplot(gs1[0,0])
            ax2 = _plt.subplot(gs1[0,1])

            yy1d_range = yy1d.copy()
            yy1d_range.sort()

            ax.axvline(x=spec2D.restwave_arr[0], ls='-', color='red')
            ax.errorbar(galaxy.spec1D.wave[wh_full]/(1.+galaxy.z), galaxy.spec1D.flux[wh_full],
                        yerr=galaxy.spec1D.flux_err[wh_full], ls='-', color='grey')
            wh_notmask = np.where(yy1d_mask == 1.)[0]
            ax.errorbar(xx1d[wh_notmask]/(1.+galaxy.z), yy1d[wh_notmask], yerr=yy1d_err[wh_notmask],
                ls='-', marker='o', color='b')
            ax.errorbar(xx1d/(1.+galaxy.z), utils.linear_fit(result_1d.params, xx1d),
                        ls='-', color='orange')


            ax.set_ylim([yy1d_range[0.01*len(yy1d)], yy1d_range[0.99*len(yy1d)]])
            ax.set_title('Continuum fit, 1D spectrum')


        ######################################
        # Perform continuum subtraction on spec2D:

        for i in range(np.shape(spec2D.flux)[0]):
            params = lmfit.Parameters()
            params.add('a', value=slope, vary=False)
            params.add('b', value=result_1d.params['b'].value)

            yy = np.append(spec2D.flux_incl_cont[i,wh_line_l], spec2D.flux_incl_cont[i,wh_line_r])
            yy_err = np.append(spec2D.flux_err[i,wh_line_l], spec2D.flux_err[i,wh_line_r])
            yy_mask = np.append(spec2D.spec_mask[i,wh_line_l], spec2D.spec_mask[i,wh_line_r])

            result = lmfit.minimize(utils.weighted_linear_residual, params,
                            args=(xx_poly, yy, yy_err, yy_mask))

            continuum_fit = utils.linear_fit(result.params, spec2D.wave)
            spec2D.cont_model[i,:] = continuum_fit

            # Subtract continuum:
            if (np.isfinite(continuum_fit.max())):
                spec2D.flux[i,:] = spec2D.flux_incl_cont[i,:] - continuum_fit
            else:
                errstr = "continuum_fit.max() is not finite!!"
                errstr += "i={:d}".format(i)
                raise ValueError(errstr)

            if galaxy.debug:
                continuum_fit_plot = utils.linear_fit(result.params, x_fit_plot)

                alpha = 0.05
                ax2.errorbar(spec2D.wave/(1.+galaxy.z), spec2D.flux_incl_cont[i,:],
                            yerr=spec2D.flux_err[i,:], ls='-', color='grey', alpha=alpha)
                ax2.errorbar(xx_poly/(1.+galaxy.z), yy, yerr=yy_err, ls='-', color='b', alpha=alpha)
                ax2.errorbar(x_fit_plot/(1.+galaxy.z), continuum_fit_plot, ls='-', color='orange', alpha=alpha)

        if galaxy.debug:
            # Finish this plot:
            yy_range = yy1d.copy()
            yy_range.sort()
            ax2.axvline(x=spec2D.restwave_arr[0], ls='-', color='red')
            ax2.set_xlim([xx_poly.min()/(1.+galaxy.z), xx_poly.max()/(1.+galaxy.z)])
            ax2.set_ylim([-0.2e-17,0.25e-17])
            ax2.set_title('Continuum fit, 2D spectrum slices')

            _plt.show()

    ######################################
    # Trim to line+continuum region
    spec2D.wave = spec2D.wave[wh_full]
    spec2D.flux_incl_cont = spec2D.flux_incl_cont[:, wh_full]
    spec2D.cont_model = spec2D.cont_model[:, wh_full]
    spec2D.flux = spec2D.flux[:, wh_full]
    spec2D.flux_err = spec2D.flux_err[:, wh_full]
    spec2D.spec_mask = spec2D.spec_mask[:, wh_full]
    spec2D.spec_weight = spec2D.spec_weight[:, wh_full]

    if not no_cont_subtraction:
        galaxy.spec1D.wave = galaxy.spec1D.wave[wh_full]
        galaxy.spec1D.flux = galaxy.spec1D.flux[wh_full]
        galaxy.spec1D.flux_err = galaxy.spec1D.flux_err[wh_full]
        galaxy.spec1D.spec_mask = galaxy.spec1D.spec_mask[wh_full]

    # Reset obs, rest wave arrays:
    spec2D.calculate_restwave(galaxy)
    if not no_cont_subtraction:
        galaxy.spec1D.calculate_restwave(galaxy)

    return spec2D





#
def get_skyline_mask_2D(spec2D, full=True):
    mask_sky, wh_sky, wh_nosky = mask_skylines_2D(flux=spec2D.flux,
                    err=spec2D.flux_err,
                    weight=spec2D.spec_weight,
                    mask=spec2D.mask_qual.copy(),
                    band_cutoff=spec2D.band_cutoff,
                    band_cutoff_units=spec2D.band_cutoff_units,
                    median_err=spec2D.median_flux_error,
                    full=full)
    return mask_sky, wh_sky, wh_nosky

#
def get_low_snr_mask_2D(spec2D, snr_cut=None, debug=False):
    mask_low_snr, low_snr_row_inds = mask_low_snr_rows_2D(flux=spec2D.flux,
                        err=spec2D.flux_err, mask=spec2D.mask_qual.copy(),
                        mask_calc=spec2D.mask_sky,
                        snr_cut=snr_cut, debug=debug)
                        # mask=spec2D.mask_qual.copy(),

    return mask_low_snr, low_snr_row_inds


def mask_skylines_2D(flux=None, err=None, weight=None, mask=None,
                    band_cutoff=None, band_cutoff_units='timesMedianErr',
                    median_err=None,
                    full=True):
    #return_sky=False,
    # wh_sky=None,

    if band_cutoff_units != 'timesMedianErr':
        raise ValueError("band_cutoff_units != 'timesMedianErr' not currently supported")

    mask_sky = mask.copy()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ### Mask skylines: just for determining which rows to consider
    #if wh_sky is None:

    # From error variance sum
    sum_over_col_arr = np.sqrt(np.sum((err*mask_sky)**2, axis=0))

    wh_nosky, wh_sky = wh_skylines_1D(sum_over_col_arr,
                        median_err=median_err,
                        cutoff=band_cutoff, full=full)

    mask_sky[:,wh_sky] = 0.

    # if return_sky:
    #     # Also mask skylines
    #     mask = mask_sky.copy()
    # else:
    #     wh_sky = []
    #


    wh_nosky = np.setdiff1d(range(np.shape(flux)[1]), wh_sky)

    return mask_sky, wh_sky, wh_nosky

#
def mask_low_snr_rows_2D(flux=None, err=None, mask=None, mask_calc=None,
                    snr_cut=None, debug=False):

    sum_arr = np.sum(flux*mask_calc, axis=1)
    # Variance error array:
    err_arr = np.sqrt(np.sum((err*mask_calc)**2, axis=1))

    snr_arr = np.abs(sum_arr/err_arr)
    high_snr_inds = np.where(snr_arr >= snr_cut)[0]
    nrows = len(snr_arr)

    # Get continuous
    high_snr_inds_cont = utils.wh_continuous(high_snr_inds)
    if len(high_snr_inds_cont) > 1:
        high_snr_inds_new = []
        for wh in high_snr_inds_cont:
            if len(wh) > 2:
                high_snr_inds_new.extend(wh)
        high_snr_inds = np.array(high_snr_inds_new)

    low_snr_row_inds = np.setdiff1d(range(len(snr_arr)), high_snr_inds)

    # Mask contigious: to outside if we've hit a low S/N row:
    if (len(low_snr_row_inds) > 0) & (len(high_snr_inds) > 0):
        if (low_snr_row_inds.min() < (nrows)/2.) & (high_snr_inds.min() < low_snr_row_inds.min()):
            low_snr_row_inds_orig = low_snr_row_inds[:]
            ii = 0
            minind = low_snr_row_inds.min()
            while ii < minind:
                low_snr_row_inds = np.append(low_snr_row_inds, ii)
                ii += 1

            low_snr_row_inds.sort()
            if debug:
                print("low_snr_row_inds_orig={}".format(low_snr_row_inds_orig))
                print("low_snr_row_inds={}".format(low_snr_row_inds))

        if (low_snr_row_inds.max() > (nrows)/2.) &  (high_snr_inds.max() > low_snr_row_inds.max()):
            low_snr_row_inds_orig = low_snr_row_inds[:]
            ii = nrows-1
            maxind = low_snr_row_inds.max()
            while ii > maxind:
                low_snr_row_inds = np.append(low_snr_row_inds, ii)
                ii -= 1

            low_snr_row_inds.sort()
            if debug:
                print("low_snr_row_inds_orig={}".format(low_snr_row_inds_orig))
                print("low_snr_row_inds={}".format(low_snr_row_inds))


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if debug:
        import matplotlib.pyplot as _plt
        import matplotlib.gridspec as _gridspec

        fig = _plt.figure()
        fig.set_size_inches(8., 4.)
        gs1 = _gridspec.GridSpec(1, 2, height_ratios=[1])
        gs1.update(left=0.05, right=0.98, wspace=0.1)
        ax3 = _plt.subplot(gs1[0,1])
        ax2 = _plt.subplot(gs1[0,0])

        ax2.imshow(mask_calc*flux, interpolation="None", origin='lower')

        ax3.plot(snr_arr, 'mo-')
        ax3.axhline(y=snr_cut, c='DimGrey', ls='-')
        ax3.set_title('S/N, using variance to calc error')

        # Plot low S/N ind bounds
        wh_cont = utils.wh_continuous(low_snr_row_inds)
        n_sp = np.shape(flux)[0]
        for wh in wh_cont:
            low = np.min(wh)
            hi = np.max(wh)
            if low != 0:
                low -= 0.5
                ax3.axvline(x=low, ls='--', color='blue')
            if hi != n_sp-1:
                hi += 0.5
                ax3.axvline(x=hi, ls='--', color='blue')

        _plt.show()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ########### Mask bad rows ##############
    mask_low_snr = mask.copy()
    mask_low_snr[low_snr_row_inds,:] = 0.


    return mask_low_snr, low_snr_row_inds

#
def mask_edges_1D(spec=None, spec_err=None, spec_mask=None,num_edge=10):
    """
    Mask the edges of the spectrum, as these are often very noisy.
    """

    # Find the edges by the first, last non-zero spec_mask value
    unmasked_inds = np.where(spec_mask == 1)[0]

    if len(unmasked_inds) > 0:
        edge_l = unmasked_inds[0]
        edge_r = unmasked_inds[-1]

        # If the edges are partway through the trimmed region:
        #       mask the num_edge pixels closest to the edge
        if edge_l > 0:
            spec_mask[edge_l:edge_l+num_edge] = 0
        if edge_r < len(spec)-1:
            spec_mask[edge_r-num_edge-1:edge_r+1] = 0

    return spec_mask


#
def spec_mask_missing(spec=None, spec_err=None, spec_weight=None,
            spec_mask=None, nonfinite_only=False):
    """
    Mask missing data, eg error=0. and weight = 0. Combined with existing mask.
    """
    if spec_mask is None:
        spec_mask = np.ones(spec.shape, dtype=float)

    # ##########################################################
    # # Mask missing values: eg, where error = 0.
    if (not nonfinite_only) & (np.abs(spec_err).max() > 0.):
        wh_missing = np.where(spec_err == 0.)[0]
        spec_mask[wh_missing] = 0.
    # ##########################################################

    # ##########################################################
    # # Mask non-finite values of data, error:
    spec_mask[np.invert(np.isfinite(spec))] = 0.
    spec_mask[np.invert(np.isfinite(spec_err))] = 0.

    if spec_weight is not None:
        spec_mask[np.where(spec_weight == 0.)] = 0.
    # ##########################################################

    return spec_mask


def mask_spec2D_from_1D(mask2D=None, mask1D=None):
    # Also mask all columns where mask1d is masked:
    wh = np.where(mask1D == 0.)[0]
    for w in wh:
        mask2D[:,w] = 0.

    return mask2D

def spec_nan_masking(flux=None, spec_mask=None, fill_value=-99., mask_zero=False, mask_all_masked=False):
    # Change the non-finite errors so they don't create problems in calculations multiplying with mask

    if not mask_all_masked:
        wh_reset = np.where((np.invert(np.isfinite(flux))) & (spec_mask == 0.))
        flux[wh_reset] = fill_value
    else:
        wh_reset = np.where((spec_mask == 0.))
        flux[wh_reset] = fill_value

    if mask_zero:
        # Eg, set missing errors to non-zero so they don't cause NaN problems later.
        wh_reset = np.where((flux == 0.) & (spec_mask == 0.))
        flux[wh_reset] = fill_value

    return flux


#


def wh_skylines_1D(err_spec, cutoff=None, median_err=None, full=False):
    if median_err is None:
        raise ValueError('median_err is not set')

    wh_sky = np.where(err_spec >= cutoff*median_err)[0]
    wh_nosky = np.where(err_spec < cutoff*median_err)[0]

    if full:
        return wh_nosky, wh_sky

    else:
        wh_cont = utils.wh_continuous(wh_nosky)
        wh_cont_sky = utils.wh_continuous(wh_sky)

        return wh_cont, wh_cont_sky

#

def vert_position(flux=None, err=None, obswave=None, obswave_val=None,
            band=None, ypos=None, refit=False,
            wh_sky=None, debug=False, skip_check=False,
            spatial_pixscale=None, dither=True,
            fill_value=-99.):
    """
    Find vertical (spatial) pixel num of center of (pos) line profile,
    at a given wavelength.

    Output
    ------
        m0 :     
            center pixel
        m_min :  
            lower pixel
        m_max :  
            upper pixel
    """

    # Replace missing errors with large errors:
    err[err == fill_value] = np.median(err)*3.

    # Find the closest pixel to the linecenter:
    wh_exact = np.where(abs(obswave-obswave_val) == abs(obswave-obswave_val).min())[0][0]



    # Straight forward sum over the wavelength direction:
    sp_sum_tmp = np.sum(flux,axis=1)
    err_sum_tmp = np.sqrt(np.sum(err**2,axis=1))

    sn_orig = np.abs(sp_sum_tmp/err_sum_tmp)
    sn_orig[~np.isfinite(sn_orig)] = 0.

    sp_arr = sp_sum_tmp.copy()
    err_arr = err_sum_tmp.copy()
    snr_arr = sn_orig.copy()


    #####
    snr_cut = 0.
    # If max snr at edge, trim it!
    if snr_arr.argmax() == 0:
        sp_arr[0] = 0.
    elif snr_arr.argmax() == len(sp_arr)-1:
        sp_arr[-1] = 0.



    if debug:
        import matplotlib.pyplot as _plt
        import matplotlib.gridspec as _gridspec

        fig = _plt.figure()
        fig.set_size_inches(16., 4.)
        gs1 = _gridspec.GridSpec(1, 3, height_ratios=[1])
        gs1.update(left=0.05, right=0.98, wspace=0.1)
        ax = _plt.subplot(gs1[0,0])
        ax2 = _plt.subplot(gs1[0,1])
        ax3 = _plt.subplot(gs1[0,2])

        ax.imshow(flux, interpolation='Nearest',origin='lower')

        ax2.plot(sp_arr, 'b')
        ax2.plot(err_arr, 'r')
        ax2.set_ylim([sp_arr.min(), sp_arr.max()])
        ax2.set_title("Signal, error collapsed over rows")

        ax3.plot(snr_arr, 'm')
        ax3.axhline(y=snr_cut, color='k', ls='--')
        ax3.set_title("S/N collapsed over rows")

        if ypos is not None:
            ax2.axvline(x=ypos, ls='--', lw=2, color='k')
            ax3.axvline(x=ypos, ls='--', lw=2, color='k')

    ####################################
    # Do peak fitting
    ####################################


    # Set a default
    ypos_orig = ypos
    if ypos is None:
        # Make a first guess??
        ypos_guess = 0.5*(len(sp_arr)-1.)
    else:
        ypos_guess = ypos

    if not refit:
        # Do the peak fitting for the first time:


        params = lmfit.Parameters()

        params.add('mu', value=ypos_guess, min=0, max=len(sp_arr)-1)
        if dither:
            params.add('deltl', value=16., min=1., max=len(sp_arr)-1)
            params.add('mu_l', expr='mu-deltl')
            params.add('deltr', expr='deltl')
            params.add('mu_r', expr='mu+deltr')

        # Need to add conditional relational criteria:
        params.add('A', value=sp_arr.max()) # for pos image
        params.add('C', value=0.)
        params.add('sigma', value=10., min=0.)

        xx = np.array(range(len(sp_arr)))

        if dither:
            result = lmfit.minimize(utils.vert_triple_gaus_residual, params,
                            args=(xx, sp_arr, err_arr))
        else:
            result = lmfit.minimize(utils.gaus_residual, params,
                            args=(xx, sp_arr, err_arr))


        m0 = np.int(np.round(result.params['mu'].value))
        if (m0 > (len(snr_arr) -1)):
            m0_closest = len(snr_arr) -1
        elif (m0 < 0):
            m0_closest = 0
        else:
            m0_closest = m0

        m0_shift_fit = result.params['mu'].value - m0
        m0_peak_snr = snr_arr[m0_closest]
        if dither:
            m1 = np.int(np.round(result.params['mu_l'].value))
            m2 = np.int(np.round(result.params['mu_r'].value))

        if debug:
            print(ypos)
            print(lmfit.fit_report(result.params))

            ax2.axvline(x=m0, ls='--', color='orange')
            ax3.axvline(x=m0, ls='--', color='orange')
            ax2.axvline(x=m0+m0_shift_fit, ls='--', color='magenta')
            ax3.axvline(x=m0+m0_shift_fit, ls='--', color='magenta')
            xx_arr_fine = np.linspace(0,xx.max(), num=len(xx)*10, endpoint=True)
            if dither:
                tmp_fit = utils.vert_triple_gaus_profile(result.params, xx_arr_fine)
            else:
                tmp_fit = utils.gaus_profile(result.params, xx_arr_fine)
            ax2.plot(xx_arr_fine, tmp_fit, color='orange')


        sp_ind0 = m0

        # Cuts it based on where it's halfway between the pos and neg images.
        # Future: cut partway through the vertical profile, to only get the stuff above the
        #  noise?
        if dither:
            sp_ind_neg1 = m1
            sp_ind_neg2 = m2

            diff = (abs(sp_ind0-sp_ind_neg1) + abs(sp_ind0-sp_ind_neg2))/2.
            cut = np.int(np.ceil(diff/2.))
        else:
            cut = np.int(np.ceil(1.5/spatial_pixscale/2.))

        m_min = m0 - cut
        m_max = m0 + cut


        # Set the current ypos -- will vary this to examine lower S/N or failure cases.
        ypos_current = ypos


        if ypos is not None:
        # Check if it's too far off for the refit!
            if (np.abs(ypos-m0) > 2.) & (not skip_check):
                print("m0, ypos are too far apart!")
                print("m0, ypos=", m0, ypos)

                # Just default to the old position!
                m0 = np.int(np.round(ypos))

                params = lmfit.Parameters()

                params.add('mu', value=ypos, vary=False)
                if dither:
                    params.add('deltl', value=16., min=1., max=len(sp_arr)-1)
                    params.add('mu_l', expr='mu-deltl')
                    params.add('deltr', expr='deltl')  #value=8., min=1., max=len(sp_arr)-1)
                    params.add('mu_r', expr='mu+deltr')

                # Need to add conditional relational criteria:
                params.add('A', value=sp_arr.max()) # for pos image
                params.add('C', value=0.)
                params.add('sigma', value=10., min=0.)

                xx = np.array(range(len(sp_arr)))
                if dither:
                    result = lmfit.minimize(utils.vert_triple_gaus_residual, params,
                                args=(xx, sp_arr, err_arr))
                else:
                    result = lmfit.minimize(utils.gaus_residual, params,
                                args=(xx, sp_arr, err_arr))

                m0 = np.int(np.round(result.params['mu'].value))
                if (m0 > (len(snr_arr) -1)):
                    m0_closest = len(snr_arr) -1
                elif (m0 < 0):
                    m0_closest = 0
                else:
                    m0_closest = m0
                m0_shift_fit = result.params['mu'].value - m0
                m0_peak_snr = snr_arr[m0_closest]
                if dither:
                    m1 = np.int(np.round(result.params['mu_l'].value))
                    m2 = np.int(np.round(result.params['mu_r'].value))

                if debug:
                    print(ypos)
                    print(lmfit.fit_report(result.params))
                    ax2.axvline(x=m0, ls='--', color='teal')
                    ax3.axvline(x=m0, ls='--', color='teal')
                    ax2.axvline(x=m0+m0_shift_fit, ls='--', color='violet')
                    ax3.axvline(x=m0+m0_shift_fit, ls='--', color='violet')
                    xx_arr_fine = np.linspace(0,xx.max(), num=len(xx)*10, endpoint=True)
                    if dither:
                        tmp_fit = utils.vert_triple_gaus_profile(result.params, xx_arr_fine)
                    else:
                        tmp_fit = utils.gaus_profile(result.params, xx_arr_fine)
                    ax2.plot(xx_arr_fine, tmp_fit, color='teal')

                    # if dither:
                    #     print("m0, m1, m2=", m0, m1, m2)

                sp_ind0 = m0
                if dither:
                    sp_ind_neg1 = m1
                    sp_ind_neg2 = m2

                    diff = (abs(sp_ind0-sp_ind_neg1) + abs(sp_ind0-sp_ind_neg2))/2.
                    cut = np.int(np.ceil(diff/2.))
                else:
                    cut = np.int(np.ceil(1.5/spatial_pixscale/2.))


                m_min = m0 - cut
                m_max = m0 + cut

                if debug:
                    print("m_max-m_min=", m_max-m_min)
                    print("len(sp_arr)-5=", len(sp_arr)-5)

            # If it's failed to do a decent fit job: ie too short length from m_max-m_min:
              # ((m_max-m_min) > len(sp_arr)-5)
            if ( ((m_max-m_min)*spatial_pixscale > 4.) | (m_max-m_min < 5)):
                result_orig = result
                print("trimming edges to do new fit:")

                # print "ypos=", ypos
                # print "len(sp_arr)=", len(sp_arr)
                # print "sp_arr.max()=", sp_arr.max()

                trimval = 5
                trim = [trimval,-trimval]
                sp_arr = sp_arr[trim[0]:trim[1]]
                err_arr = err_arr[trim[0]:trim[1]]
                snr_arr = snr_arr[trim[0]:trim[1]]

                # print "ypos-trimval=", ypos-trimval
                # print "len(sp_arr)=", len(sp_arr)
                # print "sp_arr.max()=", sp_arr.max()
                # print "min=0., max=10*sp_arr.max()", 0., 10.*sp_arr.max()

                # Trim edges, refit:
                params = lmfit.Parameters()

                params.add('mu', value=ypos-trimval, vary=False)
                if dither:
                    params.add('deltl', value=16., min=1., max=(len(sp_arr)-1)/2.)
                    params.add('mu_l', expr='mu-deltl')
                    params.add('deltr', expr='deltl')
                    params.add('mu_r', expr='mu+deltr')

                # Need to add conditional relational criteria:
                print("% misfit.galaxyutils: 10.*sp_arr.max()=", 10.*sp_arr.max())
                try:
                    params.add('A', value=sp_arr.max(), min=0., max=10.*sp_arr.max())  # for pos image
                except:
                    params.add('A', value=sp_arr.max(), min=0.)
                params.add('C', value=0.)
                params.add('sigma', value=5., min=0.)

                xx = np.array(range(len(sp_arr)))

                if dither:
                    result = lmfit.minimize(utils.vert_triple_gaus_residual, params,
                                args=(xx, sp_arr, err_arr))
                else:
                    result = lmfit.minimize(utils.gaus_residual, params,
                                    args=(xx, sp_arr, err_arr))

                m0 = np.int(np.round(result.params['mu'].value)) + trimval
                if (m0 > (len(snr_arr) -1)):
                    m0_closest = len(snr_arr) -1
                elif (m0 < 0):
                    m0_closest = 0
                else:
                    m0_closest = m0
                m0_shift_fit = result.params['mu'].value + trimval - m0
                m0_peak_snr = snr_arr[m0_closest-trimval]
                if dither:
                    m1 = np.int(np.round(result.params['mu_l'].value)) + trimval
                    m2 = np.int(np.round(result.params['mu_r'].value)) + trimval

                if debug:
                    print(ypos)
                    print(lmfit.fit_report(result.params))

                    ax2.axvline(x=m0, ls='--', color='magenta',lw=2)
                    ax3.axvline(x=m0, ls='--', color='magenta',lw=2)
                    ax2.axvline(x=m0+m0_shift_fit, ls='--', color='pink')
                    ax3.axvline(x=m0+m0_shift_fit, ls='--', color='pink')
                    xx_arr_fine = np.linspace(0,xx.max(), num=len(xx)*10, endpoint=True)
                    if dither:
                        tmp_fit = utils.vert_triple_gaus_profile(result.params, xx_arr_fine)
                    else:
                        tmp_fit = utils.gaus_profile(result.params, xx_arr_fine)
                    ax2.plot(xx_arr_fine+trimval, tmp_fit, color='magenta',lw=2)

                sp_ind0 = m0
                if dither:
                    sp_ind_neg1 = m1
                    sp_ind_neg2 = m2

                    diff = (abs(sp_ind0-sp_ind_neg1) + abs(sp_ind0-sp_ind_neg2))/2.
                    cut = np.int(np.ceil(diff/2.))
                else:
                    cut = np.int(np.ceil(1.5/spatial_pixscale/2.))

                m_min = m0 - cut
                m_max = m0 + cut

        # If it's failed to do a decent fit job: ie too short length from m_max-m_min:
        elif ((m_max-m_min > len(sp_arr)-5) | (m_max-m_min < 5)):
            result_orig = result
            print("trimming edges to do new fit:")
            trimval = 5
            trim = [trimval,-trimval]
            sp_arr = sp_arr[trim[0]:trim[1]]
            err_arr = err_arr[trim[0]:trim[1]]
            snr_arr = snr_arr[trim[0]:trim[1]]

            # Trim edges, refit:
            params = lmfit.Parameters()

            ypos_current -= trimval

            params.add('mu', value=ypos_current, min=0, max=len(sp_arr)-1)
            if dither:
                params.add('deltl', value=16., min=1., max=(len(sp_arr)-1)/2.)
                params.add('mu_l', expr='mu-deltl')
                params.add('deltr', expr='deltl')
                params.add('mu_r', expr='mu+deltr')
            # Need to add conditional relational criteria:
            params.add('A', value=sp_arr.max(), min=0., max=10*sp_arr.max())
            params.add('C', value=0.)
            params.add('sigma', value=5., min=0.)

            xx = np.array(range(len(sp_arr)))

            if dither:
                result = lmfit.minimize(utils.vert_triple_gaus_residual, params,
                            args=(xx, sp_arr, err_arr))
            else:
                result = lmfit.minimize(utils.gaus_residual, params,
                            args=(xx, sp_arr, err_arr))

            m0 = np.int(np.round(result.params['mu'].value)) + trimval
            m0_shift_fit = result.params['mu'].value + trimval - m0
            m0_peak_snr = snr_arr[m0-trimval]
            if dither:
                m1 = np.int(np.round(result.params['mu_l'].value)) + trimval
                m2 = np.int(np.round(result.params['mu_r'].value)) + trimval

            if debug:
                print(ypos            )
                print(lmfit.fit_report(result.params))

                ax2.axvline(x=m0, ls='--', color='purple',lw=2)
                ax3.axvline(x=m0, ls='--', color='purple',lw=2)
                ax2.axvline(x=m0+m0_shift_fit, ls='--', color='gold')
                ax3.axvline(x=m0+m0_shift_fit, ls='--', color='gold')
                xx_arr_fine = np.linspace(0,xx.max(), num=len(xx)*10, endpoint=True)
                if dither:
                    tmp_fit = utils.vert_triple_gaus_profile(result.params, xx_arr_fine)
                else:
                    tmp_fit = utils.gaus_profile(result.params, xx_arr_fine)
                ax2.plot(xx_arr_fine+trimval, tmp_fit, color='purple',lw=2)

            sp_ind0 = m0
            if dither:
                sp_ind_neg1 = m1
                sp_ind_neg2 = m2

                diff = (abs(sp_ind0-sp_ind_neg1) + abs(sp_ind0-sp_ind_neg2))/2.
                cut = np.int(np.ceil(diff/2.))
            else:
                cut = np.int(np.ceil(1.5/spatial_pixscale/2.))

            m_min = m0 - cut
            m_max = m0 + cut

    else:
        # REFIT -- already did trimming, just need to find center.
        params = lmfit.Parameters()
        params.add('mu', value=ypos, min=0, max=len(sp_arr)-1)
        params.add('A', value=sp_arr.max())  # for pos image
        params.add('C', value=0.)
        params.add('sigma', value=5., min=0.)

        xx = np.array(range(len(sp_arr)))

        result = lmfit.minimize(utils.gaus_residual, params,
                        args=(xx, sp_arr, err_arr))

        m0 = np.int(np.round(result.params['mu'].value))
        m0_shift_fit = result.params['mu'].value - m0
        m0_peak_snr = snr_arr[m0]

        if debug:
            print("ypos=", ypos)
            print("sp_arr=", sp_arr)

            print(lmfit.fit_report(result.params))

            ax2.axvline(x=m0, ls='--', color='orange')
            ax3.axvline(x=m0, ls='--', color='orange')
            ax2.axvline(x=m0+m0_shift_fit, ls='--', color='gold')
            ax3.axvline(x=m0+m0_shift_fit, ls='--', color='gold')
            xx_arr_fine = np.linspace(0,xx.max(), num=len(xx)*10, endpoint=True)
            tmp_fit = utils.gaus_profile(result.params, xx_arr_fine)
            ax2.plot(xx_arr_fine, tmp_fit, color='orange')



        sp_ind0 = m0
        m_min = 0
        m_max = len(sp_arr)
        cut = None #-99


        # Check if it's too far off for the refit!
        if ypos is not None:
            if ((np.abs(ypos-m0) > 2.) & (not skip_check)):
                # Just default to the old position!
                m0 = np.int(np.round(ypos))
                m0_shift_fit = ypos - m0
                m0_peak_snr = snr_arr[m0]



    if debug:
        print("m0, m0_shift_fit, m0_peak_snr,m_min, m_max=", \
                m0, m0_shift_fit, m0_peak_snr, m_min, m_max)
        print( "cut=", cut)


    return m0, m0_shift_fit, m0_peak_snr, m_min, m_max


def calculate_2D_fit_weights(spec2D):
    ### *******************************************************
    ########## Weighting ############
    ## ++++++++++++++++++++++++++++++++++++++++++++
    # ##      linearly weighting the direct spectrum.
    if spec2D.weighting_type == 'up-edges':
        ## Mask the input images, to avoid noisy rows.
        sum_arr = np.sum(spec2D.flux*spec2D.mask_sky, axis=1)
        ## Variance error arr
        err_arr = np.sqrt(np.sum((spec2D.flux_err*spec2D.mask_sky)**2, axis=1))
        snr_arr = np.abs(sum_arr/err_arr)

        weights_nonnorm = 1./snr_arr

        # set low S/N row weights to 0:
        weights_nonnorm[spec2D.low_snr_row_inds] = 0.

    ## ++++++++++++++++++++++++++++++++++++++++++++
    # ## *No* weighting test:
    elif spec2D.weighting_type == 'none':
        weights_nonnorm = np.ones(spec2D.flux.shape[0])
    ## ++++++++++++++++++++++++++++++++++++++++++++


    ## ++++++++++++++++++++++++++++++++++++++++++++
    ## *OPTIMAL* weighting test, esp for dispersion-only objects:
    elif spec2D.weighting_type == 'optimal':
        sum_arr = np.abs(np.sum(spec2D.flux_incl_cont*spec2D.mask_sky, axis=1))
        # Need to fit this to a gaussian:
        err_arr = np.abs(np.sum(spec2D.flux_err*spec2D.mask_sky, axis=1))
        xx_arr = np.linspace(0,len(sum_arr), num=len(sum_arr), endpoint=False)

        params = lmfit.Parameters()
        params.add('mu', value=len(sum_arr)/2.)
        params.add('A', value=sum_arr.max())
        params.add('sigma', value=len(sum_arr)/4., min=0.)

        result = lmfit.minimize(utils.gaus_residual, params,
                        args=(xx_arr, sum_arr, err_arr))

        gauss_fit = utils.gaus_profile(result.params, xx_arr)

        weights_nonnorm = gauss_fit

        weights_nonnorm[spec2D.low_snr_row_inds] = 0.
    ## ++++++++++++++++++++++++++++++++++++++++++++
    else:
        print("weighting type '{}' not supported".format(spec2D.weighting_type))

    ### *******************************************************

    # Normalize:
    w_tot = np.sum(weights_nonnorm)
    weights = weights_nonnorm/w_tot

    # If the normalization fails because w_tot is = 0.:
    if w_tot == 0.:
        weights[:] = 0.


    spec2D.fitting_weight_profile = weights
    spec2D.fitting_weight_matrix = np.array([spec2D.fitting_weight_profile for i in range(spec2D.flux.shape[1])]).T

    return spec2D




def calculate_slit_offset_angles(galaxy):
    """Calculate misalignment of galaxy within slit: store in galaxy"""

    if not galaxy.generate_model:
        slit_PA_on_img = galaxy.spec2D.slit_PA - galaxy.pstamp.img_PA
        delt_PA = galaxy.sersicPA - slit_PA_on_img

        delt_PA = np.arctan(np.tan(delt_PA*np.pi/180.))*180./np.pi

        galaxy.delt_PA = delt_PA

    galaxy.delt_PA_abs = np.abs(galaxy.delt_PA)

    return galaxy


#
def get_m0_lam0_pos_2D(spec2D, galaxy):

    #######
    # m0_shift, lam0 = utils.m0_shift_lam0_gausfits(self.spec2D)

    # Already have m0_shift from the triple profile fit:
    m0_shift = spec2D.m0_shift_fit


    # Mask sky and low S/N rows
    spec = spec2D.flux*spec2D.mask_sky*spec2D.mask_low_snr
    err = spec2D.flux_err*spec2D.mask_sky*spec2D.mask_low_snr

    # Straight forward sum over the wavelength direction:
    y_prof = np.sum(spec, axis=1)
    y_prof_err = np.sqrt(np.sum(err**2, axis=1))
    yy_arr = np.linspace(spec2D.m_lims[0], spec2D.m_lims[1],
                    num=(spec2D.m_lims[1]-spec2D.m_lims[0]), endpoint=False)

    wh_not_mask_y = np.setdiff1d(range(len(yy_arr)), spec2D.low_snr_row_inds)


    # Sum over spatial direction
    lam_prof = np.sum(spec, axis=0)
    lam_prof_err = np.sqrt(np.sum(err**2, axis=0))
    lam_arr = spec2D.wave.copy()

    wh_not_mask_lam = spec2D.wh_nosky.copy()
    n_spatial = spec2D.flux.shape[0]

    if (len(wh_not_mask_y) > 0) & (len(wh_not_mask_lam) > 0):

        #############################
        #############################
        #############################
        #if galaxy.debug:
        # Setup gaus fit for y:
        params = lmfit.Parameters()
        params.add('mu', value=spec2D.ypos, min=yy_arr.min(), max=yy_arr.max())
        params.add('A', value=y_prof.max())  # for pos image
        params.add('sigma', value=n_spatial/4., min=0.)

        try:
            result = lmfit.minimize(utils.gaus_residual_mask, params,
                            args=(yy_arr, y_prof, y_prof_err, wh_not_mask_y))

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if galaxy.debug:
                import matplotlib.pyplot as _plt
                import matplotlib.gridspec as _gridspec

                fig = _plt.figure()
                fig.set_size_inches(8., 4.)
                # gs1 = _gridspec.GridSpec(1, 1, height_ratios=[1])
                # gs1.update(left=0.05, right=0.98, wspace=0.1)
                # ax = _plt.subplot(gs1[0,0])

                ax = fig.add_subplot(1,1,1)

                ax.errorbar(yy_arr, y_prof, yerr=y_prof_err, ls='-', color='grey')

                ax.errorbar(yy_arr[wh_not_mask_y], y_prof[wh_not_mask_y],
                    yerr=y_prof_err[wh_not_mask_y], ls='-', marker='o', color='b')

                #
                prof_fit = utils.gaus_profile(result.params, yy_arr)
                ax.plot(yy_arr, prof_fit, ls='-',color='orange')

                ax.axvline(x=result.params['mu'].value,ls='--',color='orange')
                ax.axvline(x=spec2D.ypos,ls='--',color='blue')
                ax.axvline(x=yy_arr.min(),ls='--',color='grey')
                ax.axvline(x=yy_arr.max(),ls='--',color='grey')


                ax.axvline(x=spec2D.m0, ls='--',color='teal')
                ax.axvline(x=spec2D.m0+spec2D.m0_shift_fit, ls='--',color='magenta')

                _plt.show()

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if ((result.params['mu'].value >= yy_arr.min()) & \
                 (result.params['mu'].value <= yy_arr.max())):
                if np.abs(spec2D.ypos -result.params['mu'].value ) <= 2.:
                     m0_shift = result.params['mu'].value - spec2D.m0
                elif (np.abs(spec2D.m0 -result.params['mu'].value ) <= 2.) & \
                    (np.abs(spec2D.m0_shift_fit) <= 1.) & (spec2D.m0_peak_snr >= 3.):
                    # It was high enough S/N, and should use the fit value.
                    m0_shift = result.params['mu'].value - spec2D.m0
                    if galaxy.debug:
                        print( "m0, m0_shift, m0_shift_fit_first=", \
                                spec2D.m0, m0_shift, spec2D.m0_shift_fit)
                else:
                    print( 'Warning in m0_shift_lam0_gausfits (too far from ypos)')
                    m0_shift = 0.
            else:
                print( 'Warning in m0_shift_lam0_gausfits (too far from ypos)')
                m0_shift = 0.
        except:
            print( "Exception m0_shift_lam0_gausfits (yprof)")
            m0_shift = 0.

        #############################
        #############################
        #############################

        # Setup gaus fit for lam0:
        params = lmfit.Parameters()

        wave_range = lam_arr.max()-lam_arr.min()

        lam0_init = spec2D.lam0 #spec2D.restwave_arr[0]*(1.+galaxy.z)
        spec2D.lam0_init = lam0_init

        params.add('mu', value=lam0_init, min=lam_arr.min() + wave_range/4.,
                                    max=lam_arr.max() - wave_range/4.)
        params.add('A', value=lam_prof.max())  # for pos image
        params.add('sigma', value=len(lam_arr)/4., min=0.)

        try:
            result = lmfit.minimize(utils.gaus_residual_mask, params,
                            args=(lam_arr, lam_prof, lam_prof_err, wh_not_mask_lam))

            if ((result.params['mu'].value >= lam_arr.min() + wave_range/4.) & \
                 (result.params['mu'].value <= lam_arr.max() - wave_range/4.)):
                if np.abs(lam0_init -result.params['mu'].value ) <= wave_range/4.:
                     lam0 = result.params['mu'].value
                else:
                    print( 'Warning in m0_shfit_lam0_gausfits (too far from lam0)')
                    lam0 = lam0_init
            else:
                print( 'Warning in m0_shfit_lam0_gausfits (lam mu not in range)')
                lam0 = lam0_init
        except:
            print( "Exception m0_shift_lam0_gausfits (lamprof)")
            lam0 = lam0_init
    else:
        m0_shift = 0.
        lam0_init = spec2D.lam0 #spec2D.restwave_arr[0]*(1.+galaxy.z)
        lam0 = lam0_init


    spec2D.m0_shift = m0_shift
    spec2D.lam0 = lam0

    return spec2D

#
def fit_emission_y_profile(spec2D, gal, inst, filename_plot=None, plot=False, num_MC=500):
    if filename_plot is not None:
        plot = True

    # Mask sky:  don't zero out low S/N rows -- masking later
    spec = spec2D.flux*spec2D.mask_sky#*spec2D.mask_low_snr
    err = spec2D.flux_err*spec2D.mask_sky#*spec2D.mask_low_snr

    # Straight forward sum over the wavelength direction:
    pixscale = inst.pixscale
    n_spatial = spec2D.flux.shape[0]
    y_prof = np.sum(spec, axis=1)
    y_prof_err = np.sqrt(np.sum(err**2, axis=1))
    yy_arr = np.linspace(spec2D.m_lims[0]*pixscale, spec2D.m_lims[1]*pixscale,
                    num=(spec2D.m_lims[1]-spec2D.m_lims[0]), endpoint=False)

    print("spec2D.m_lims={}".format(spec2D.m_lims))

    wh_not_mask_y = np.setdiff1d(range(len(yy_arr)), spec2D.low_snr_row_inds)
    #wh_not_mask_y = range(len(yy_arr))

    # if (len(wh_not_mask_y) == 0):
    #     raise ValueError


    if (len(wh_not_mask_y) > 0):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if (filename_plot is not None) | (plot is True):
            import matplotlib.pyplot as _plt
            import matplotlib.gridspec as _gridspec

            fig = _plt.figure()
            fig.set_size_inches(8., 4.)

            ax = fig.add_subplot(1,1,1)

        #############################
        #############################
        #############################
        #if galaxy.debug:
        # Setup gaus fit for y:
        std2FWHM = (2.*np.sqrt(2.*np.log(2.)))
        params = lmfit.Parameters()
        maxsigma = (yy_arr.max()-yy_arr.min())/std2FWHM # FWHM can't be greater than the window
        params.add('mu', value=spec2D.ypos*pixscale, min=yy_arr.min(), max=yy_arr.max())
        params.add('A', value=y_prof.max())  # for pos image
        params.add('sigma', value=n_spatial/4., min=0.)
        #params.add('C', value=0.)   # constant offset

        try:

            # Check there's even more than PSF FWHM in unmasked rows: otherwise, fits will really not nec. work.
            if len(wh_not_mask_y)*pixscale/inst.PSF.PSF_FWHM < 0.8:
                raise ValueError

            #############################
            result = lmfit.minimize(utils.gaus_residual_mask, params,
                            args=(yy_arr, y_prof, y_prof_err, wh_not_mask_y))

            #
            if result.params['sigma'].value > maxsigma:
                params = lmfit.Parameters()
                params.add('mu', value=spec2D.ypos*pixscale, min=yy_arr.min(), max=yy_arr.max())
                params.add('A', value=y_prof.max())  # for pos image
                params.add('sigma', value=n_spatial/4., min=0., max=maxsigma)
                #params.add('C', value=0.)   # constant offset
                #############################
                result = lmfit.minimize(utils.gaus_residual_mask, params,
                                args=(yy_arr, y_prof, y_prof_err, wh_not_mask_y))
            #
            if (filename_plot is not None) | (plot is True):

                ax.errorbar(yy_arr, y_prof, yerr=y_prof_err, ls='-', color='grey', zorder=101.)

                ax.errorbar(yy_arr[wh_not_mask_y], y_prof[wh_not_mask_y],
                    yerr=y_prof_err[wh_not_mask_y], ls='-', marker='o', color='b', zorder=102.)


                ax.axvline(x=result.params['mu'].value,ls='--',color='magenta', zorder=100.)
                ax.axvline(x=yy_arr.min(),ls='--',color='grey')
                ax.axvline(x=yy_arr.max(),ls='--',color='grey')

                ax.axhline(y=0.,ls='--',color='grey')

                prof_fit = utils.gaus_profile(result.params, yy_arr)
                ax.plot(yy_arr, prof_fit, ls='-',color='magenta', zorder=100.)
                ylim = ax.get_ylim()

            spec2D.sigma_y_emis_obs = result.params['sigma'].value
            spec2D.sigma_y_emis_int = np.sqrt(spec2D.sigma_y_emis_obs**2 - (inst.PSF.PSF_FWHM/std2FWHM)**2)
            if (not np.isfinite(spec2D.sigma_y_emis_int)):
                spec2D.sigma_y_emis_int = 0.
            spec2D.D50 = spec2D.sigma_y_emis_int * std2FWHM

            values_true = np.array([spec2D.sigma_y_emis_obs, spec2D.sigma_y_emis_int, spec2D.D50])
            # MC: perturb each spec pt by gaussian random number with 1sig=error
            # then do lmfit for each realization:
            value_matrix = np.zeros((num_MC, len(values_true)))

            # Structure of value matrix: columns:
            #  z_fit   vel_disp   flux_line0 ... flux_linen-1  cont_coeff0 .. cont_coeffn-1

            for i in range(num_MC):
                #print "MC error iter %i/%i" % (i+1,self.num_MC)
                if ( ((i+1) % 50 == 0)):
                    print("MC error iter {:d}/{:d}".format(i+1,num_MC))

                spec_perturb = y_prof.copy()
                # Now perturb randomly, using normal distribution
                rand_vals = np.random.randn(len(spec_perturb))
                spec_perturb += y_prof_err*rand_vals

                params_best = lmfit.Parameters()
                params_best.add('mu', value=spec2D.ypos*pixscale, min=yy_arr.min(), max=yy_arr.max())
                params_best.add('A', value=y_prof.max())  # for pos image
                params_best.add('sigma', value=n_spatial/4., min=0.)#, max=maxsigma)

                #now fit the perturbed spectrum:
                result_mc = lmfit.minimize(utils.gaus_residual_mask, params_best,
                                args=(yy_arr, spec_perturb, y_prof_err, wh_not_mask_y))

                if result_mc.params['sigma'].value > maxsigma:
                    params_best = lmfit.Parameters()
                    params_best.add('mu', value=spec2D.ypos*pixscale, min=yy_arr.min(), max=yy_arr.max())
                    params_best.add('A', value=y_prof.max())  # for pos image
                    params_best.add('sigma', value=n_spatial/4., min=0., max=maxsigma)

                    #now fit the perturbed spectrum:
                    result_mc = lmfit.minimize(utils.gaus_residual_mask, params_best,
                                    args=(yy_arr, spec_perturb, y_prof_err, wh_not_mask_y))

                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if (filename_plot is not None) | (plot is True):
                    prof_fit_mc = utils.gaus_profile(result_mc.params, yy_arr)
                    ax.plot(yy_arr, prof_fit_mc,
                            ls='-', lw=1, c='red', alpha=0.025, zorder=-1.)

                value_matrix[i,0] = result_mc.params['sigma'].value
                value_matrix[i,1] = np.sqrt(value_matrix[i,0]**2 - (inst.PSF.PSF_FWHM/std2FWHM)**2)
                if (not np.isfinite(value_matrix[i,1])):
                    value_matrix[i,1] = 0.
                value_matrix[i,2] = value_matrix[i,1] * std2FWHM


            # Get lower, upper 1 sig values for each param
            values_err = np.zeros((len(values_true),2))
            limits = np.percentile(value_matrix, [15.865, 84.135], axis=0).T
            values_err[:,0] = values_true[:] - limits[:,0]
            values_err[:,1] = limits[:,1] - values_true[:]


            spec2D.sigma_y_emis_obs_err = values_err[0,:]
            spec2D.sigma_y_emis_int_err = values_err[1,:]
            spec2D.D50_err = values_err[2,:]


            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if (filename_plot is not None) | (plot is True):

                ax.axvline(x=result.params['mu'].value-0.5*2.35*spec2D.sigma_y_emis_obs,ls=':',color='magenta')
                ax.axvline(x=result.params['mu'].value+0.5*2.35*spec2D.sigma_y_emis_obs,ls=':',color='magenta')

                #print("% fit_emission_y_profile: spec2D.D50={}".format(spec2D.D50))

                ax.axvline(x=result.params['mu'].value-0.5*spec2D.D50,ls=':',color='blue', zorder=102.)
                ax.axvline(x=result.params['mu'].value+0.5*spec2D.D50,ls=':',color='blue', zorder=102.)

                ax.axvline(x=result.params['mu'].value-0.5*inst.PSF.PSF_FWHM,ls=':',color='grey')
                ax.axvline(x=result.params['mu'].value+0.5*inst.PSF.PSF_FWHM,ls=':',color='grey')

                print("D50/FWHM={}".format(spec2D.D50/inst.PSF.PSF_FWHM))
                print("sigmaobs/maxsigma={}".format(spec2D.sigma_y_emis_obs/maxsigma))
                print("D50/maxsigma={}".format(spec2D.D50/maxsigma))
                print("nfree/FWHM={}".format(len(wh_not_mask_y)*pixscale/inst.PSF.PSF_FWHM))

                ax.set_ylim(ylim)
                ax.set_xlabel('Spatial extent [arcsec]')
                ax.set_ylabel('Collapsed emission-line profile [arbitrary flux]')
                try:
                    f_or_m = "-".join(gal.maskname.split("_"))
                except:
                    f_or_m = gal.field
                ax.set_title(r'{}.{}.{} (SLITOBJNAME: {})'.format(f_or_m, spec2D.band, np.int64(gal.ID), spec2D.slitobjname))

                if (filename_plot is not None):
                    _plt.savefig(filename_plot, bbox_inches='tight', dpi=300)
                    _plt.close(fig)
                else:
                    _plt.show()

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
        except:
        #else:
            missing_val = -99.
            spec2D.D50 = missing_val
            spec2D.D50_err = np.array([missing_val,missing_val])
            spec2D.sigma_y_emis_obs = missing_val
            spec2D.sigma_y_emis_obs_err = np.array([missing_val,missing_val])
            spec2D.sigma_y_emis_int = missing_val
            spec2D.sigma_y_emis_int_err = np.array([missing_val,missing_val])
    else:

        missing_val = -99.
        spec2D.D50 = missing_val
        spec2D.D50_err = np.array([missing_val,missing_val])
        spec2D.sigma_y_emis_obs = missing_val
        spec2D.sigma_y_emis_obs_err = np.array([missing_val,missing_val])
        spec2D.sigma_y_emis_int = missing_val
        spec2D.sigma_y_emis_int_err = np.array([missing_val,missing_val])


    return spec2D
