# misfit/utils.py
# Utilities for MISFIT
#
# Copyright 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Some handling of MCMC / posterior distribution analysis inspired by speclens:
#    https://github.com/mrgeorge/speclens/blob/master/speclens/fit.py

from __future__ import print_function

import numpy as np
import os
import sys

from scipy.stats import gaussian_kde
from scipy.optimize import fmin

from astropy.io import ascii

import six

from contextlib import contextmanager

deg2rad = np.pi/180.

#
def weighted_linear_residual(params, xx, data, err, mask):
    wh_bad = np.where(mask == 0)[0]
    line_fit = linear_fit(params, xx)
    resid = (line_fit-data)/err
    resid[wh_bad] = 0.
    resid[~np.isfinite(resid)] = 0.

    return resid

#
def linear_fit(params, xx):
    # Just a line
    a = params['a'].value # slope
    b = params['b'].value # intercept

    line_fit = a*xx + b

    return line_fit

#
def gaus_simple(xx, A, mu, sigma, C=None):

    gaus = np.exp((-(xx-mu)**2)/(2.*sigma**2))/np.sqrt(2.*np.pi*sigma)

    #gaus = gaus/gaus.max()*A

    gaus *= A

    if C is not None:
        gaus += C

    return gaus

#
def gaus_profile(params, xx):
    A = params['A'].value
    mu = params['mu'].value
    sigma = params['sigma'].value

    #if params.has_key('C'):
    if 'C' in params.keys():
        C = params['C'].value
    else:
        C = None

    gaus = gaus_simple(xx, A, mu, sigma, C=C)

    return gaus

def gaus_residual(params, xx, data, err):
    gaus = gaus_profile(params, xx)
    resid = (gaus-data)/err
    resid[~np.isfinite(resid)] = 0.

    return resid

#
def gaus_residual_mask(params, xx, data, err, wh_not_mask):
    gaus = gaus_profile(params, xx)
    resid = (gaus[wh_not_mask]-data[wh_not_mask])/err[wh_not_mask]
    resid[~np.isfinite(resid)] = 0.
    return resid


def vert_triple_gaus_profile(params,xx):
    A = params['A'].value
    sigma = params['sigma'].value
    mu = params['mu'].value
    mu_l = params['mu_l'].value
    mu_r = params['mu_r'].value
    C = params['C'].value

    prof = np.zeros(len(xx))

    prof += gaus_simple(xx, A, mu, sigma)
    prof += gaus_simple(xx, -A, mu_l, sigma)
    prof += gaus_simple(xx, -A, mu_r, sigma)
    prof += C

    return prof

def vert_triple_gaus_residual(params, xx, data, err):
    trip_prof = vert_triple_gaus_profile(params, xx)
    resid = (trip_prof-data)/err
    resid[~np.isfinite(resid)] = 0.

    return resid


def wh_continuous(wh_arr):
    wh_arrs = []
    arr = []
    for i in six.moves.xrange(len(wh_arr)):
        if i < len(wh_arr)-1:
            if wh_arr[i+1] - wh_arr[i] == 1:
                arr.append(wh_arr[i])
            else:
                arr.append(wh_arr[i])
                wh_arrs.append(arr)
                arr = []
        else:
            arr.append(wh_arr[i])
            wh_arrs.append(arr)

    return wh_arrs

#

def y_proj_phys(re, delt_PA, q):
    return q*re/np.sqrt((q*np.cos(delt_PA*deg2rad))**2 + (np.sin(delt_PA*deg2rad)**2))

def x_proj_phys(re, delt_PA, q):
    return q*re/np.sqrt((q*np.cos((90-delt_PA)*deg2rad))**2 + (np.sin((90-delt_PA)*deg2rad)**2))

def x_proj_major(re, delt_PA, q):
    # Ignores trimming from slit
    x_proj_major = re*np.sin(delt_PA*deg2rad)

    return np.abs(x_proj_major)

def y_proj_major(re, delt_PA):
    # Ignores trimming from slit
    y_proj_major = re*np.cos(delt_PA*deg2rad)

    return np.abs(y_proj_major)

def r_unproj_major(y_proj, delt_PA, q):
    r_unproj = y_proj/np.cos(delt_PA*deg2rad)

    return np.abs(r_unproj)

def y_proj_major_inslit(re, delt_PA, slit_width_arcsec=None):
    slit_HW = 0.5 * slit_width_arcsec  #0.35  # slit HW in [arcsec]
    a_slit = np.abs(slit_HW/np.sin(delt_PA*deg2rad))

    x_proj = np.abs(re*np.sin(delt_PA*deg2rad))
    if x_proj <= slit_HW:
        # whole RE lies within slit:
        y_proj_major_inslit = np.abs(re*np.cos(delt_PA*deg2rad))
    else:
        # part of object falls outside of slit:
        y_proj_major_inslit = np.abs(a_slit*np.cos(delt_PA*deg2rad))

    return y_proj_major_inslit

def inclination_angle(q, q0=None):
    return np.arccos(np.sqrt(((q)**2 - q0**2)/(1.-q0**2)))



@contextmanager
def file_or_stdout(file_name):
    if file_name is None:
        yield sys.stdout
    else:
        with open(file_name, 'w') as out_file:
            yield out_file



def range_arrs(fitEmis2D):
    param_range = []
    param_lower = []
    for j in six.moves.xrange(len(fitEmis2D.kinModel.theta_names)):
        if fitEmis2D.kinModel.theta_vary[j]:
            param_range.append(max(fitEmis2D.thetaPrior.theta_bounds[j])-\
                        min(fitEmis2D.thetaPrior.theta_bounds[j]))
            param_lower.append(min(fitEmis2D.thetaPrior.theta_bounds[j]))

    return param_range, param_lower


def find_peak_gaussian_KDE(flatchain, initval):
    """
    Return chain parameters that give peak of the posterior PDF, using KDE.
    """
    try:
        nparams = flatchain.shape[1]
    except:
        nparams = 1

    if nparams > 1:
        peakvals = np.zeros(nparams)
        for i in six.moves.xrange(nparams):
            kern = gaussian_kde(flatchain[:,i])
            peakvals[i] = fmin(lambda x: -kern(x), initval[i], disp=False)
        return peakvals
    else:
        neg_kernel = gaussian_KDE_kernel_func(flatchain)
        peakval = fmin(neg_kernel, initval, disp=False)

        return peakval


def find_peak_gaussian_KDE_multiD(flatchain, linked_inds, initval):
    """
    Return chain parameters that give peak of the posterior PDF *FOR LINKED PARAMETERS, using KDE.
    """

    nparams = len(linked_inds)
    kern = gaussian_kde(flatchain[:,linked_inds].T)
    peakvals = fmin(lambda x: -kern(x), initval, disp=False)

    return peakvals



#
def get_bestfit_values_linked(fitEmis2D, err_theta, mcmc_lims_zip,
            theta_linked_posteriors=None, bins=50):
    #
    flatchain = fitEmis2D.sampler_dict['flatchain'].copy()


    try:
        if theta_linked_posteriors.strip().lower() == 'all':
            theta_linked_posteriors = range(len(fitEmis2D.mcmc_results.bestfit_theta)-4)
    except:
        pass

    # Reset output values:
    bestfit_theta = fitEmis2D.mcmc_results.bestfit_theta.copy()

    bestfit_theta_linked = find_peak_gaussian_KDE_multiD(flatchain, theta_linked_posteriors,
            fitEmis2D.mcmc_results.bestfit_theta[theta_linked_posteriors])

    for k in six.moves.xrange(len(theta_linked_posteriors)):
        bestfit_theta[theta_linked_posteriors[k]] = bestfit_theta_linked[k]
        err_theta[theta_linked_posteriors[k]] = np.array([bestfit_theta_linked[k]-mcmc_lims_zip[theta_linked_posteriors[k]][0],
                                mcmc_lims_zip[theta_linked_posteriors[k]][1] - bestfit_theta_linked[k]])

    return bestfit_theta, err_theta
