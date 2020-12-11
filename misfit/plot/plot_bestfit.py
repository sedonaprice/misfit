# misfit/plot/plot_bestfit.py
# Module to contain plotting functions for MCMC analysis: best fits + residuals
#
# Copyright 2014-2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Hidden modules prepended with '_'
from __future__ import print_function

import numpy as np
import os
import matplotlib

#matplotlib.rcParams['text.usetex'] = False

import six

from matplotlib.colors import colorConverter
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
try:
    from plot_image_slit import plot_image_slit
except:
    from .plot_image_slit import plot_image_slit

cmap = cm.gray_r

from misfit.general import general_utils as utils




def plot_bestfit_model(fitEmis2D, fileout=None, noTitles=False, verbose=False, usetex=True):
    """
    Input:
        FitEmis2D object: contains
            galaxy
            instrument (spectrum)
            instrument_img
            model           fitting model + results

    Optional:
        ax:             axes instance. If set, must reside in pre-defined figure,
                          and already be created (using proper gridspec positioning, etc)
        saveToFile:     (True/False) Save current axis instance to file. 'fileout' must be set.
    """
    sh_spec_trim = np.shape(fitEmis2D.galaxy.spec2D.flux)
    sp_rat = sh_spec_trim[1]/np.float(sh_spec_trim[0])  # width/height


    fig = plt.figure()

    if fitEmis2D.kinModel is not None:
        fig.set_size_inches(20., 5.)
        n_cols = 5
        gs1 = gridspec.GridSpec(1, n_cols, width_ratios=[1, 1, sp_rat, sp_rat, sp_rat], wspace=0.1)
    else:
        fig.set_size_inches(12., 5.)
        n_cols = 3
        gs1 = gridspec.GridSpec(1, n_cols, width_ratios=[1, 1, sp_rat], wspace=0.1)

    axes = []
    for i in six.moves.xrange(n_cols):
        axes.append(plt.subplot(gs1[0,i]))

    #
    # Pstamp + slit plot, image convolved to spectrum resolution:
    if fitEmis2D.kinModel.bestfit is not None:
        m0shift = fitEmis2D.kinModel.bestfit.theta[-2]
    else:
        m0shift = 0.


    if fitEmis2D.galaxy.pstamp is not None:
        # Align center of pstamp with center of model:
        plot_image_slit(galaxy=fitEmis2D.galaxy, instrument=fitEmis2D.instrument,
                        instrument_img=fitEmis2D.instrument_img,
                        ax=axes[0], n_spatial_pixels=sh_spec_trim[0],
                        m0=fitEmis2D.galaxy.spec2D.m0 + m0shift, verbose=verbose)

        plot_image_slit(galaxy=fitEmis2D.galaxy, instrument=fitEmis2D.instrument,
                        instrument_img=fitEmis2D.instrument_img,
                        convolve=True, downsample=True, conv_sigma=fitEmis2D.instrument.conv_sigma,
                        ax=axes[1], n_spatial_pixels=sh_spec_trim[0],
                        m0=fitEmis2D.galaxy.spec2D.m0 + m0shift, verbose=verbose)
    else:
        # Turn off axes:
        axes[0].axis('off')
        axes[1].axis('off')

    ## 2D plot of spectra
    int_mode = "nearest"  # 'None'

    # IGNORE SKYLINES for scaling!
    if len(fitEmis2D.galaxy.spec2D.wh_nosky) > 0:
        vmin = fitEmis2D.galaxy.spec2D.flux[:,fitEmis2D.galaxy.spec2D.wh_nosky].min()
        vmax = fitEmis2D.galaxy.spec2D.flux[:,fitEmis2D.galaxy.spec2D.wh_nosky].max()
    else:
        vmin = fitEmis2D.galaxy.spec2D.flux.min()
        vmax = fitEmis2D.galaxy.spec2D.flux.max()


    origin = 'lower'       # Origin in lower left
    spec2d = axes[2].imshow(fitEmis2D.galaxy.spec2D.flux,
                    cmap=cmap, interpolation=int_mode,
                    vmin=vmin, vmax=vmax, origin=origin)

    if fitEmis2D.kinModel is not None:
        spec2dc = axes[3].imshow(fitEmis2D.kinModel.model,
                        cmap=cmap, interpolation=int_mode,
                        vmin=vmin, vmax=vmax, origin=origin)

        spec2d2 = axes[4].imshow(fitEmis2D.galaxy.spec2D.flux - fitEmis2D.kinModel.model,
                    cmap=cmap, interpolation=int_mode,
                    vmin=vmin, vmax=vmax, origin=origin)


    re_arcsec_int = fitEmis2D.galaxy.re_arcsec  # HWHM, intrinsic, arcsec
    re_proj = utils.y_proj_major(re_arcsec_int, fitEmis2D.galaxy.delt_PA)

    # if (fitEmis2D.instrument.conv_sigma is None):
    #     # Define conv_sigma if not set:
    #     conv_FWHM = np.sqrt(fitEmis2D.instrument.PSF_FWHM**2 - fitEmis2D.instrument_img.PSF_FWHM**2)
    #     conv_sigma = conv_FWHM/(2.*np.sqrt(2.*np.log(2.)))

    # print "FIX THIS!"
    # raise ValueError

    conv_sigma_galfit = fitEmis2D.instrument.PSF.PSF_FWHM/(2.*np.sqrt(2.*np.log(2.)))

    # re_sigma = re_proj/(np.sqrt(2.*np.log(2.)))
    # re_sigma_conv = np.sqrt(conv_sigma_galfit**2 + re_sigma**2)
    # re_proj_arcsec = re_sigma_conv*(np.sqrt(2.*np.log(2.)))

    re_sigma = re_arcsec_int/(np.sqrt(2.*np.log(2.)))
    re_sigma_conv = np.sqrt(conv_sigma_galfit**2 + re_sigma**2)
    re_major_arcsec = re_sigma_conv*(np.sqrt(2.*np.log(2.)))
    re_proj_arcsec = utils.y_proj_major(re_major_arcsec, fitEmis2D.galaxy.delt_PA)


    if verbose:
        print("% plot_bestfit:")
        try:
            print(fitEmis2D.galaxy.maskname, fitEmis2D.galaxy.ID)
        except:
            print(fitEmis2D.galaxy.field, fitEmis2D.galaxy.ID)
        print("\t Plotting convolved, projected R_E")
        print("\t R_E int = {:0.3f}".format(re_arcsec_int) )
        print("\t R_E proj = {:0.3f}".format(re_proj) )
        print("\t R_E conv+proj = {:0.3f}".format(re_proj_arcsec) )

    row_color = 'orange'
    ls = ':'
    n_wave = np.shape(fitEmis2D.galaxy.spec2D.flux)[1]
    n_sp = np.shape(fitEmis2D.galaxy.spec2D.flux)[0]

    low = fitEmis2D.galaxy.spec2D.m0 + m0shift - \
            re_proj_arcsec/fitEmis2D.instrument.pixscale
    hi = fitEmis2D.galaxy.spec2D.m0 + m0shift + \
            re_proj_arcsec/fitEmis2D.instrument.pixscale
    axes[2].axhline(y=low, xmin=0., xmax=n_wave, ls=ls, color=row_color)
    axes[2].axhline(y=hi, xmin=0., xmax=n_wave, ls=ls, color=row_color)

    if fitEmis2D.kinModel is not None:
        axes[3].axhline(y=low, xmin=0., xmax=n_wave, ls=ls, color=row_color)
        axes[3].axhline(y=hi, xmin=0., xmax=n_wave, ls=ls, color=row_color)
        axes[4].axhline(y=low, xmin=0., xmax=n_wave, ls=ls, color=row_color)
        axes[4].axhline(y=hi, xmin=0., xmax=n_wave, ls=ls, color=row_color)

    # Plot m0+m0_shift center:
    mid = fitEmis2D.galaxy.spec2D.m0  + m0shift
    ls = '-'
    row_color = 'orange'
    axes[2].axhline(y=mid, xmin=0., xmax=n_wave, ls=ls, color=row_color)
    if fitEmis2D.kinModel is not None:
        axes[3].axhline(y=mid, xmin=0., xmax=n_wave, ls=ls, color=row_color)
        axes[4].axhline(y=mid, xmin=0., xmax=n_wave, ls=ls, color=row_color)

    ######################
    ## Alternate: use continuous to only draw lines bounding cont regions.
    row_color = 'yellow'
    ls = '--'
    wh_cont = utils.wh_continuous(fitEmis2D.galaxy.spec2D.low_snr_row_inds)
    for wh in wh_cont:
        low = np.min(wh)
        hi = np.max(wh)
        if low != 0:
            low -= 0.5  # bottom of pixel

            axes[2].axhline(y=low, xmin=0., xmax=n_wave, ls=ls, color=row_color)
            if fitEmis2D.kinModel is not None:
                axes[3].axhline(y=low, xmin=0., xmax=n_wave, ls=ls, color=row_color)
                axes[4].axhline(y=low, xmin=0., xmax=n_wave, ls=ls, color=row_color)
        if hi != n_sp-1:
            hi += 0.5 # top of pixel
            axes[2].axhline(y=hi, xmin=0., xmax=n_wave, ls=ls, color=row_color)
            if fitEmis2D.kinModel is not None:
                axes[3].axhline(y=hi, xmin=0., xmax=n_wave, ls=ls, color=row_color)
                axes[4].axhline(y=hi, xmin=0., xmax=n_wave, ls=ls, color=row_color)

    # Fill with hatching:
    color_hatch = 'yellow'
    lw_hatch = 0.5 #1.

    matplotlib.rcParams['hatch.color'] = color_hatch
    matplotlib.rcParams['hatch.linewidth'] = lw_hatch

    # find the contiguous low_snr_inds:
    low_snr_inds_cont = wh_cont

    pad = 0.5
    hatch_pattern = '////' #'//'
    lw_edge_hatch = 0
    for low_snr in low_snr_inds_cont:
        if len(low_snr) == 1:
            low_snr = [low_snr[0], low_snr[0]]
        axes[2].add_patch(Polygon([[-pad,low_snr[0]-pad],
                        [n_wave-pad, low_snr[0]-pad],
                        [n_wave-pad, low_snr[-1]+1-pad],
                        [-pad, low_snr[-1]+1-pad]],
                        closed=True,
                        lw=lw_edge_hatch, hatch=hatch_pattern,
                        fill=False, color = color_hatch))
        if fitEmis2D.kinModel is not None:
            axes[3].add_patch(Polygon([[-pad,low_snr[0]-pad],
                            [n_wave-pad, low_snr[0]-pad],
                            [n_wave-pad, low_snr[-1]+1-pad],
                            [-pad, low_snr[-1]+1-pad]],
                            closed=True,
                            lw=lw_edge_hatch, hatch=hatch_pattern,
                            fill=False, color=color_hatch))
            axes[4].add_patch(Polygon([[-pad,low_snr[0]-pad],
                            [n_wave-pad, low_snr[0]-pad],
                            [n_wave-pad, low_snr[-1]+1-pad],
                            [-pad, low_snr[-1]+1-pad]],
                            closed=True,
                            lw=lw_edge_hatch, hatch=hatch_pattern,
                            fill=False, color=color_hatch))

    ## Alternate: use continuous to only draw lines bounding cont regions.
    column_color = 'yellow'
    ls = '--'
    wh_cont = utils.wh_continuous(fitEmis2D.galaxy.spec2D.wh_sky_fit)
    for wh in wh_cont:
        low = np.min(wh)
        hi = np.max(wh)
        if low != 0:
            low -= 0.5  # bottom of pixel
            axes[2].axvline(x=low, ls=ls, color=column_color)
            if fitEmis2D.kinModel is not None:
                axes[3].axvline(x=low, ls=ls, color=column_color)
                axes[4].axvline(x=low, ls=ls, color=column_color)
        if hi != n_sp-1:
            hi += 0.5 # top of pixel
            axes[2].axvline(x=hi, ls=ls, color=column_color)
            if fitEmis2D.kinModel is not None:
                axes[3].axvline(x=hi, ls=ls, color=column_color)
                axes[4].axvline(x=hi, ls=ls, color=column_color)
    #################
    # Fill with hatching: skyline masking for fitting:
    color_hatch = 'yellow'
    lw_hatch = 1.

    pad = 0.5
    hatch_pattern = '//'
    lw_edge_hatch = 0
    for wh in wh_cont:
        if len(wh_cont) == 1:
            wh_cont = [wh_cont[0], wh_cont[0]]
        axes[2].add_patch(Polygon([[-pad,wh_cont[0]-pad],
                        [n_wave-pad, wh_cont[0]-pad],
                        [n_wave-pad, wh_cont[-1]+1-pad],
                        [-pad, wh_cont[-1]+1-pad]],
                        closed=True,
                        lw=lw_edge_hatch,
                        fill=False, color = color_hatch))
        if fitEmis2D.kinModel is not None:
            axes[3].add_patch(Polygon([[-pad,wh_cont[0]-pad],
                            [n_wave-pad, wh_cont[0]-pad],
                            [n_wave-pad, wh_cont[-1]+1-pad],
                            [-pad, wh_cont[-1]+1-pad]],
                            closed=True,
                            lw=lw_edge_hatch,
                            fill=False, hatch=hatch_pattern, color=color_hatch))
            axes[4].add_patch(Polygon([[-pad,wh_cont[0]-pad],
                            [n_wave-pad, wh_cont[0]-pad],
                            [n_wave-pad, wh_cont[-1]+1-pad],
                            [-pad, wh_cont[-1]+1-pad]],
                            closed=True,
                            lw=lw_edge_hatch,
                            fill=False, hatch=hatch_pattern, color=color_hatch))
    #################

    # Turn off axes boxes, tick marks:
    axes[2].set_axis_off()
    if fitEmis2D.kinModel is not None:
        axes[3].set_axis_off()
        axes[4].set_axis_off()




    if not noTitles:
        fontsize=11
        axes[0].set_title('High-res. image', fontsize=fontsize)
        axes[1].set_title('Image, spectrum spatial res.', fontsize=fontsize)
        axes[2].set_title('Spec', fontsize=fontsize)
        if fitEmis2D.kinModel is not None:
            axes[3].set_title('Model', fontsize=fontsize)
            axes[4].set_title('Residual', fontsize=fontsize)


    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()
