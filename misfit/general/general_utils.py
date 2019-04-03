# misfit/utils.py
# Utilities for MISFIT
# 
# Copyright 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

# Hidden modules prepended with '_'
from __future__ import print_function

import numpy as _np
import os as _os
import sys as _sys

from scipy.stats import gaussian_kde as _gaussian_kde
from scipy.optimize import fmin as _fmin

from astropy.io import ascii as _ascii


from astropy.extern import six as _six

from contextlib import contextmanager

deg2rad = _np.pi/180.
    
#
def weighted_linear_residual(params, xx, data, err, mask):
    wh_bad = _np.where(mask == 0)[0]
    line_fit = linear_fit(params, xx)
    resid = (line_fit-data)/err
    resid[wh_bad] = 0.
    resid[~_np.isfinite(resid)] = 0.
    
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
    
    gaus = _np.exp((-(xx-mu)**2)/(2.*sigma**2))/_np.sqrt(2.*_np.pi*sigma)
    
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
    
    if params.has_key('C'):
        C = params['C'].value
    else:
        C = None
    
    gaus = gaus_simple(xx, A, mu, sigma, C=C)
    
    return gaus
    
def gaus_residual(params, xx, data, err):
    gaus = gaus_profile(params, xx)
    resid = (gaus-data)/err
    resid[~_np.isfinite(resid)] = 0.
    
    return resid
    
#
def gaus_residual_mask(params, xx, data, err, wh_not_mask):
    gaus = gaus_profile(params, xx)
    resid = (gaus[wh_not_mask]-data[wh_not_mask])/err[wh_not_mask]
    resid[~_np.isfinite(resid)] = 0.
    return resid
    
    
def vert_triple_gaus_profile(params,xx):
    A = params['A'].value
    sigma = params['sigma'].value
    mu = params['mu'].value
    mu_l = params['mu_l'].value
    mu_r = params['mu_r'].value
    C = params['C'].value
    
    prof = _np.zeros(len(xx))
    
    prof += gaus_simple(xx, A, mu, sigma)
    prof += gaus_simple(xx, -A, mu_l, sigma)
    prof += gaus_simple(xx, -A, mu_r, sigma)
    prof += C
    
    return prof
    
def vert_triple_gaus_residual(params, xx, data, err):
    trip_prof = vert_triple_gaus_profile(params, xx)
    resid = (trip_prof-data)/err
    resid[~_np.isfinite(resid)] = 0.
    
    return resid


def wh_continuous(wh_arr):
    wh_arrs = []
    arr = []
    for i in _six.moves.xrange(len(wh_arr)):
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
    return q*re/_np.sqrt((q*_np.cos(delt_PA*deg2rad))**2 + (_np.sin(delt_PA*deg2rad)**2))

def x_proj_phys(re, delt_PA, q):
    return q*re/_np.sqrt((q*_np.cos((90-delt_PA)*deg2rad))**2 + (_np.sin((90-delt_PA)*deg2rad)**2))

def x_proj_major(re, delt_PA, q):
    # Ignores trimming from slit
    x_proj_major = re*_np.sin(delt_PA*deg2rad)

    return _np.abs(x_proj_major)

def y_proj_major(re, delt_PA):
    # Ignores trimming from slit
    y_proj_major = re*_np.cos(delt_PA*deg2rad)
    
    return _np.abs(y_proj_major)
    
def r_unproj_major(y_proj, delt_PA, q):
    r_unproj = y_proj/_np.cos(delt_PA*deg2rad)
    
    return _np.abs(r_unproj)
    
def y_proj_major_inslit(re, delt_PA, slit_width_arcsec=None):
    slit_HW = 0.5 * slit_width_arcsec  #0.35  # slit HW in [arcsec]
    a_slit = _np.abs(slit_HW/_np.sin(delt_PA*deg2rad))
    
    x_proj = _np.abs(re*_np.sin(delt_PA*deg2rad))
    if x_proj <= slit_HW:
        # whole RE lies within slit:
        y_proj_major_inslit = _np.abs(re*_np.cos(delt_PA*deg2rad))
    else:
        # part of object falls outside of slit:
        y_proj_major_inslit = _np.abs(a_slit*_np.cos(delt_PA*deg2rad))
        
    return y_proj_major_inslit
    
def inclination_angle(q, q0=None):
    return _np.arccos(_np.sqrt(((q)**2 - q0**2)/(1.-q0**2)))
    


@contextmanager
def file_or_stdout(file_name):
    if file_name is None:
        yield _sys.stdout
    else:
        with open(file_name, 'w') as out_file:
            yield out_file



def range_arrs(fitEmis2D):
    param_range = []
    param_lower = []
    for j in _six.moves.xrange(len(fitEmis2D.kinModel.theta_names)):
        if fitEmis2D.kinModel.theta_vary[j]:
            param_range.append(max(fitEmis2D.thetaPrior.theta_bounds[j])-\
                        min(fitEmis2D.thetaPrior.theta_bounds[j]))
            param_lower.append(min(fitEmis2D.thetaPrior.theta_bounds[j]))
            
    return param_range, param_lower 
    
    
    
#
def getPeakKDE(flatchain, guess):
    # Taken from speclens: 
    #    https://github.com/mrgeorge/speclens/blob/master/speclens/fit.py
    # and modified as necessary
    """
    Return chain pars that give peak of posterior PDF, using KDE.
    From speclens: https://github.com/mrgeorge/speclens/blob/master/speclens/fit.py
    By Matt George
    """
    if(len(flatchain.shape)==1):
        nPars=1
        kern=_gaussian_kde(flatchain)
        peakKDE=_fmin(lambda x: -kern(x), guess,disp=False)
        return peakKDE
    else:
        nPars=flatchain.shape[1]
        peakKDE=_np.zeros(nPars)
        for ii in _six.moves.xrange(nPars):
            kern=_gaussian_kde(flatchain[:,ii])
            peakKDE[ii]=_fmin(lambda x: -kern(x), guess[ii],disp=False)
        return peakKDE
        
#
def getPeakKDEmultiD(flatchain, inds, guess):
    """
    Return chain pars that give peak of posterior PDF *FOR LINKED PARAMETERS, using KDE.
    From speclens: https://github.com/mrgeorge/speclens/blob/master/speclens/fit.py
    By Matt George
    """
    nPars = len(inds)
    
    kern = _gaussian_kde(flatchain[:,inds].T)
    peakKDE = _fmin(lambda x: -kern(x), guess, disp=False)
    
    return peakKDE
    
        
        
#
def get_bestfit_values_linked(fitEmis2D, err_theta, mcmc_lims_zip,
            theta_linked_posteriors=None, bins=50):
    #
    flatchain = fitEmis2D.sampler_dict['flatchain'].copy()
    
    # samples = fitEmis2D.sampler_dict['flatchain'].copy()
    # if len(theta_linked_posteriors) == 2:
    #     x = samples[:,theta_linked_posteriors[0]]
    #     y = samples[:,theta_linked_posteriors[1]]
    #     
    #     extent = [[x.min(), x.max()], [y.min(), y.max()]]
    #     X = _np.linspace(extent[0][0], extent[0][1], bins+1)
    #     Y = _np.linspace(extent[1][0], extent[1][1], bins+1)
    #     
    #     H, X, Y = _np.histogram2d(x.flatten(), y.flatten(), bins=(X, Y))
    #     
    #     #############################
    #     # use H peak to get "best fit values"
    #     wh_H_peak = _np.column_stack(_np.where(H == H.max()))[0]
    #     
    #     col1_best = _np.average([X[wh_H_peak[0]],X[wh_H_peak[0]+1]])
    #     col2_best = _np.average([Y[wh_H_peak[1]], Y[wh_H_peak[1]+1]])
    #     
    #     # Reset output values:
    #     bestfit_theta = fitEmis2D.mcmc_results.bestfit_theta.copy()
    #     
    #     bestfit_theta[theta_linked_posteriors[0]] = col1_best
    #     bestfit_theta[theta_linked_posteriors[1]] = col2_best
    #     
    #     err_theta[theta_linked_posteriors[0]] = _np.array([col1_best-mcmc_lims_zip[theta_linked_posteriors[0]][0], 
    #                             mcmc_lims_zip[theta_linked_posteriors[0]][1] - col1_best])
    #     #
    #     err_theta[theta_linked_posteriors[1]] = _np.array([col2_best-mcmc_lims_zip[theta_linked_posteriors[1]][0], 
    #                             mcmc_lims_zip[theta_linked_posteriors[1]][1] - col2_best])
    #     
    #     
    # elif len(theta_linked_posteriors) > 2:
    #     raise ValueError("Linked posteriors for more than 2 parameters not implemented!")
    # elif len(theta_linked_posteriors) == 1:
    #     raise ValueError("Must specify 2+ parameters for linked posteriors")
    
    try:
        if theta_linked_posteriors.strip().lower() == 'all':
            theta_linked_posteriors = range(len(fitEmis2D.mcmc_results.bestfit_theta)-4)
    except:
        pass
    
    # Reset output values:
    bestfit_theta = fitEmis2D.mcmc_results.bestfit_theta.copy()
    
    bestfit_theta_linked = getPeakKDEmultiD(flatchain, theta_linked_posteriors, 
            fitEmis2D.mcmc_results.bestfit_theta[theta_linked_posteriors])
            
    for k in _six.moves.xrange(len(theta_linked_posteriors)):
        bestfit_theta[theta_linked_posteriors[k]] = bestfit_theta_linked[k]
        err_theta[theta_linked_posteriors[k]] = _np.array([bestfit_theta_linked[k]-mcmc_lims_zip[theta_linked_posteriors[k]][0], 
                                mcmc_lims_zip[theta_linked_posteriors[k]][1] - bestfit_theta_linked[k]])
    
    return bestfit_theta, err_theta
    
    