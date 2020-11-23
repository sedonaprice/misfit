# misfit/plot/plot_bestfit.py
# Module to contain plotting functions for MCMC analysis: best fits + residuals
#
# Copyright 2014-2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function

import numpy as np
import os

import matplotlib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

import six

def plot_trace(sampler_dict, fitEmis2D, fileout=None):
    names = []

    for i in six.moves.xrange(len(fitEmis2D.kinModel.theta)):
        if fitEmis2D.kinModel.theta_vary[i]:
            names.append(fitEmis2D.kinModel.theta_names_nice[i])

    ######################################
    # Setup plot:
    f = plt.figure()
    f.set_size_inches(8.5,11.)


    nRows = len(names)
    gs = gridspec.GridSpec(nRows, 1, hspace=0.2)

    axes = []

    nWalkers = sampler_dict['nWalkers']

    alpha = max(0.01, 1./np.float(nWalkers))


    # Define random color inds for tracking some walkers:
    nTraceWalkers = 5
    cmap = cm.viridis
    alphaTrace = 0.8
    lwTrace = 1.5
    trace_inds = np.random.randint(0,nWalkers, size=nTraceWalkers)
    trace_colors = []
    for i in six.moves.xrange(nTraceWalkers):
        trace_colors.append(cmap(1./np.float(nTraceWalkers)*i))

    norm_inds = np.setdiff1d(range(nWalkers), trace_inds)



    for k in six.moves.xrange(nRows):
        axes.append(plt.subplot(gs[k,0]))

        axes[k].plot(sampler_dict['chain'][norm_inds,:,k].T, '-', color='black', alpha=alpha)

        for j in six.moves.xrange(nTraceWalkers):
            axes[k].plot(sampler_dict['chain'][trace_inds[j],:,k].T, '-',
                    color=trace_colors[j], lw=lwTrace, alpha=alphaTrace)


        axes[k].set_ylabel(names[k])

        if k == nRows-1:
            axes[k].set_xlabel('Step number')


    #############################################################
    # Save to file:
    plt.savefig(fileout, bbox_inches='tight', dpi=300)
    plt.close()


    return None
