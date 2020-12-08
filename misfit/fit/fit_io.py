# Copyright 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

from __future__ import print_function

import pickle
import numpy as np

import copy

def make_emcee_sampler_dict(sampler,fitEmis2D=None, nBurn=0):
    """
    Save chain + key results from emcee sampler instance to a dict,
    as the emcee samplers aren't pickleable.
    """
    # Cut first nBurn steps, to avoid the edge cases that are rarely explored.
    samples = sampler.chain[:, nBurn:, :].reshape((-1, sampler.dim))
    # Walkers, iterations
    probs = sampler.lnprobability[:, nBurn:].reshape((-1))

    fitEmis2D_fit = sampler.args[0].copy()

    # Drop the large space hogs:

    # Copy the key stuff:
    whitelist_keys = ['galaxy', 'instrument', 'intensityProfile', 'kinProfile']
    for key in whitelist_keys:
        fitEmis2D_fit.__dict__[key] = copy.deepcopy(fitEmis2D_fit.kinModel.aperModel.__dict__[key])


    whitelist_others = ['mcmcOptions', 'kinModelOptions', 'thetaSettings']
    for key in whitelist_others:
        fitEmis2D_fit.__dict__[key] = copy.deepcopy(fitEmis2D.__dict__[key])


    del fitEmis2D_fit.kinModel.aperModel

    #  'fitEmis2D': sampler.args,

    try:
        acor_time = sampler.get_autocorr_time(tol=10, quiet=True)
    except:
        try:
            acor_time = sampler.get_autocorr_time(self, low=5, c=10)
        except:
            acor_time = np.NaN

    # Make a dictionary:
    df = { 'fitEmis2D': fitEmis2D_fit,
           'chain': sampler.chain[:, nBurn:, :],
           'lnprobability': sampler.lnprobability[:, nBurn:],
           'flatchain': samples,
           'flatlnprobability': probs,
           'nIter': sampler.iterations,
           'nParam': sampler.dim,
           'nCPU': sampler.threads,
           'nWalkers': len(sampler.chain),
           'acceptance_fraction': sampler.acceptance_fraction,
           'acor_time': acor_time }

    return df

#
# def load_sampler_info(filename_sampler):
#
#     fitEmis2D.sampler_dict = pickle.load(open(fitEmis2D.filename_sampler, "rb"))
#
#     return fitEmis2D

def loadpickle(filename):
    data = pickle.load(open(filename, "rb"))
    return data

def dumppickle(data, filename=None):
    pickle.dump(data, open(filename, "wb") )
    return None
