# Copyright 2016-2019 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

# Some handling of MCMC / posterior distribution analysis inspired by speclens:
#    https://github.com/mrgeorge/speclens/blob/master/speclens/fit.py
#    Many thanks to Matt George for guidance/help in learning to implement MCMC

from __future__ import print_function

import numpy as np

import pickle
import json
import six

import emcee
import psutil
# try:
#     import acor
# except:
#     pass



import copy

import time, datetime

import misfit.general.general_utils as utils
import misfit.plot as misfit_plot
from misfit import GalaxyBasic, ObsSpectrum2DBasic
from misfit.model import KinModel2DOptions, KinModel2D
from misfit.model.kin_classes import KinProfileFiducial, IntensityProfileFiducial, ThetaPriorFlat

try:
    import fit_io
except:
    from . import fit_io
#import sys
#from emcee.utils import MPIPool

class FitEmissionLines2DResults(object):
    """
    Pared-down class with very basic fitEmis2D result attributes.
    """
    # def __init__(self, galaxy, instrument, **kwargs):
    def __init__(self, galaxy=None, instrument=None, instrument_img=None, **kwargs):
        #
        self.galaxy = galaxy
        self.instrument = instrument
        self.instrument_img = instrument_img

        # Store stuff about the lines that will be fit:
        self.linegroup_name = None
        self.linenames_arr = None   # All lines to be included in 2D model
        self.restwave_arr = None    # Rest wavelengths of these lines
        self.flux_ratio_arr = None  # Relative flux strengths

        # Kinematic profile function. Fiducial takes 5 parameters
        self.kinProfile = KinProfileFiducial()

        # Intensity profile: Information about how the model light profile is generated
        self.intensityProfile = IntensityProfileFiducial(galaxy=self.galaxy)

        # Options for model, to pass when kinModel is created:
        self.kinModelOptions = KinModel2DOptions()

        # Priors for model:
        self.thetaPrior = None

        # Parameter info:
        self.thetaSettings = None

        # Options for MCMC fitting, including output filenames
        self.mcmcOptions = MCMC2DOptions()

        # If set, an array of arrays giving indices of theta_fitting (ie, theta which vary)
        #       for which best-fit should be calculated in multiD space.
        self.theta_linked_posteriors = None
        self.mcmc_results = MCMCResults()      # Class for holding MCMC resuls


        self.setAttr(**kwargs)

    def setAttr(self,**kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))

    def copy(self):
        return copy.deepcopy(self)




class FitEmissionLines2DBasic(object):
    """
    Pared-down class with very basic fitEmis2D attributes.
    """
    def __init__(self, **kwargs):

        # Kinematic model class
        self.kinModel = None

        # Options for model, to pass when kinModel is created:
        self.kinModelOptions = KinModel2DOptions()

        # Priors for model:
        self.thetaPrior = None

        self.thetaSettings = None

        self.setAttr(**kwargs)

    def setAttr(self,**kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))

    def copy(self):
        return copy.deepcopy(self)

    def make_model(self, galaxy=None, instrument=None, theta_fitting=None):
        # Use methods in self.kinModel
        self.kinModel.make_model(galaxy=galaxy, instrument=instrument,
                    theta_fitting=theta_fitting)

    def update_model(self, theta_fitting=None):
        self.kinModel.update_model(theta_fitting=theta_fitting)

def initialize_fitting_theta(theta, theta_vary):
    # Initialize model:
    # Get intial fitting values
    j = 0
    theta_fitting_init = np.array([])
    for i in six.moves.xrange(len(theta)):
        if theta_vary[i]:
            theta_fitting_init = np.append(theta_fitting_init, theta[i])

    return theta_fitting_init


def lnlike(theta_fitting, fitEmis2D):
    """
    Define log likelihood for MCMC parameter exploration
    Defined as lnlike = -chi^2_{reduced}

    Input:
    theta_fitting, fitEmis2D object (has spectrum and model class)
        taking model from fitEmis2D.kinModel.model
        data from fitEmis2D.kinModel.aperModel.galaxy.spec2D

    """
    # Make model:
    fitEmis2D.update_model(theta_fitting=theta_fitting)

    llike = fitEmis2D.kinModel.model_llike()

    return llike

def lnprob(theta_fitting, fitEmis2D):
    # Prior: defined in fitEmis2D.

    ln_prior = fitEmis2D.thetaPrior.log_prior(theta_fitting=theta_fitting)

    if not np.isfinite(ln_prior):
        return -np.inf

    ln_like = lnlike(theta_fitting, fitEmis2D)
    ln_prob = ln_prior + ln_like

    if not np.isfinite(ln_prob):
        # Make sure the non-finite ln_prob is -Inf, for emcee handling
        ln_prob = -np.inf

    return ln_prob

#

def run_mcmc(fitEmis2D, fitEmis2D_fit=None):
    """
    Run emcee to do 2D kin fitting using fitEmis2D. Option to pass fitEmis2D_fit,
    a version of fitEmis2D that is very pared down, for faster copying+calculation.
    Otherwise will crudely get rid of most things in these classes before calculation.

    However, if the fitEmis2D, galaxy, instrument class have lots of specially-defined attributes,
        may want to fully setup model (eg, do fitEmis2D.setup_model(thetaSettings=thetaSettings))
        and then delete attributes
    """
    if np.int(emcee.__version__[0]) >= 3:
        fitEmis2D = _run_mcmc_emcee_3(fitEmis2D, fitEmis2D_fit=fitEmis2D_fit)
    else:
        fitEmis2D = _run_mcmc_emcee_221(fitEmis2D, fitEmis2D_fit=fitEmis2D_fit)

    return fitEmis2D


def _run_mcmc_emcee_221(fitEmis2D, fitEmis2D_fit=None):
    """
    Run emcee to do 2D kin fitting using fitEmis2D. Option to pass fitEmis2D_fit,
    a version of fitEmis2D that is very pared down, for faster copying+calculation.
    Otherwise will crudely get rid of most things in these classes before calculation.

    However, if the fitEmis2D, galaxy, instrument class have lots of specially-defined attributes,
        may want to fully setup model (eg, do fitEmis2D.setup_model(thetaSettings=thetaSettings))
        and then delete attributes
    """
    # --------------------------------
    # Setup for threading for emcee, if desired
    if fitEmis2D.mcmcOptions.nCPUs is None:
        fitEmis2D.mcmcOptions.nCPUs = np.int(np.floor(psutil.cpu_count()*fitEmis2D.mcmcOptions.cpuFrac))
    if fitEmis2D.mcmcOptions.NoThread:
        fitEmis2D.mcmcOptions.nCPUs = 1


    # --------------------------------
    if fitEmis2D_fit is None:

        fitEmis2D_fit = make_pruned_fitEmis2D_class(fitEmis2D)



    # --------------------------------
    # Initialize walker starting positions
    keys = []
    for j in six.moves.xrange(len(fitEmis2D.kinModel.theta_names)):
        if fitEmis2D.kinModel.theta_vary[j]:
            keys.append(fitEmis2D.kinModel.theta_names[j])

    initial_pos = init_walker_pos(fitEmis2D, ndim=fitEmis2D.kinModel.n_free_param,
                    nwalkers=fitEmis2D.mcmcOptions.nWalkers)


    # --------------------------------
    # Initialize emcee sampler

    sampler = emcee.EnsembleSampler(fitEmis2D.mcmcOptions.nWalkers, fitEmis2D.kinModel.n_free_param,
                                    lnprob,
                                    args=(fitEmis2D_fit,),
                                    a = fitEmis2D.mcmcOptions.scale_param_a,
                                    threads = fitEmis2D.mcmcOptions.nCPUs)#,
                                    #pool=pool)


    # --------------------------------
    # Start log file
    with utils.file_or_stdout(fitEmis2D.mcmcOptions.filename_log) as f_log:
        f_log.write('NoThread = {} \n'.format(fitEmis2D.mcmcOptions.NoThread))
        f_log.write('nCPUs: {} \n'.format(fitEmis2D.mcmcOptions.nCPUs))
        f_log.write('nSubpixels = {} \n'.format(fitEmis2D.kinModel.kinModelOptions.nSubpixels))
        f_log.write('PSF_type = {} \n'.format(fitEmis2D.instrument.PSF.PSF_type))
        f_log.write('PSF yspace_dither_arc = {} \n'.format(fitEmis2D.instrument.PSF.yspace_dither_arc))


        print('NoThread = {}'.format(fitEmis2D.mcmcOptions.NoThread))
        print('nCPUs: {}'.format(fitEmis2D.mcmcOptions.nCPUs))
        print('nSubpixels = {}'.format(fitEmis2D.kinModel.kinModelOptions.nSubpixels))
        print('PSF_type = {}'.format(fitEmis2D.instrument.PSF.PSF_type))
        print('PSF yspace_dither_arc = {}'.format(fitEmis2D.instrument.PSF.yspace_dither_arc))
        print("")

        ################################################################
        # --------------------------------
        # Run burn-in
        if fitEmis2D.mcmcOptions.nBurn > 0:
            f_log.write('Burn-in: \n')
            f_log.write('Start: {} \n'.format(datetime.datetime.now()))

            start = time.time()

            ####
            pos = initial_pos
            prob = None
            state = None
            for k in six.moves.xrange(fitEmis2D.mcmcOptions.nBurn):
                print( "k={:3d}, time: {}".format(k, datetime.datetime.now()) )
                pos, prob, state = sampler.run_mcmc(pos, 1, lnprob0=prob, rstate0=state)


            end = time.time()
            elapsed = end-start

            # try:
            #     acor_time = [acor.acor(sampler.chain[:,:,jj])[0] for jj in six.moves.xrange(sampler.dim)]
            # except:
            #     acor_time = "Undefined, chain did not converge"


            try:
                acor_time = sampler.get_autocorr_time(self, low=5, c=10)
            except:
                acor_time = "Undefined, chain did not converge"





            #######################################################################################
            # Return Burn-in info
            # ****

            f_log.write('End: '+str(datetime.datetime.now())+'\n')
            f_log.write('\n')
            f_log.write('******************'+'\n')
            f_log.write('nCPU, nParam, nWalker, nBurn = {}, {}, {}, {} \n'.format(fitEmis2D.mcmcOptions.nCPUs,
                            fitEmis2D.kinModel.n_free_param,
                            fitEmis2D.mcmcOptions.nWalkers,
                            fitEmis2D.mcmcOptions.nBurn))
            f_log.write("keys={} \n".format(keys))
            f_log.write('Scale param a= {} \n' .format(fitEmis2D.mcmcOptions.scale_param_a))
            f_log.write('Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s) \n'.format(elapsed, np.floor(elapsed/60.),
                                (elapsed/60.-np.floor(elapsed/60.))*60.))
            f_log.write("Mean acceptance fraction: {:0.3f} \n ".format(np.mean(sampler.acceptance_fraction)))

            f_log.write("Ideal acceptance frac: 0.2 - 0.5 \n")
            # Autocorrelation time:
            f_log.write("Acor est: ")
            f_log.write(str(acor_time))
            f_log.write('\n')
            f_log.write('******************\n')
            f_log.write('\n')

            nBurn_nEff = 2
            try:
                if fitEmis2D.mcmcOptions.nBurn < np.max(acor_time) * nBurn_nEff:
                    f_log.write('#################\n')
                    f_log.write('nBurn is less than {}*acorr time \n'.format(nBurn_nEff))
                    f_log.write('#################\n')
                    # Give warning if the burn-in is less than say 2-3 times the autocorr time
            except:
                f_log.write('#################\n')
                f_log.write("acorr time undefined -> can't check convergence\n")
                f_log.write('#################\n')


            ##########################################
            ##########################################
            ##########################################
            # --------------------------------
            # Plot burn-in trace, if output file set
            if fitEmis2D.mcmcOptions.filename_plot_trace_burnin is not None:
                sampler_dict_burnin = fit_io.make_emcee_sampler_dict(sampler, fitEmis2D=fitEmis2D, nBurn=0)
                misfit_plot.plot_trace(sampler_dict_burnin, fitEmis2D,
                                fileout=fitEmis2D.mcmcOptions.filename_plot_trace_burnin)
            ##########################################
            ##########################################
            ##########################################

            # Reset sampler after burn-in:
            sampler.reset()

        else:
            # --------------------------------
            # No burn-in: set initial position:
            pos = np.array(initial_pos)
            prob = None
            state = None

        #######################################################################################
        # ****
        # --------------------------------
        # Run sampler: Get start time
        f_log.write('Ensemble sampling:\n')
        f_log.write('Start: {} \n'.format(datetime.datetime.now()))
        start = time.time()

        # --------------------------------
        # Run sampler: output info at each step
        for ii in six.moves.xrange(fitEmis2D.mcmcOptions.nSteps):
            pos_cur = pos.copy()    # copy just in case things are set strangely

            # --------------------------------
            # Only do one step at a time.
            pos, prob, state = sampler.run_mcmc(pos_cur, 1, lnprob0=prob, rstate0=state)
            # --------------------------------

            # --------------------------------
            # Give output info about this step:
            print( "ii={:3d}, a_frac={:0.4f}, time: {}".format(ii, np.mean(sampler.acceptance_fraction),
                            datetime.datetime.now()) )

            # try:
            #     acor_time = [acor.acor(sampler.chain[:,:,jj])[0] for jj in six.moves.xrange(sampler.dim)]
            #     f_log.write("{:d}: acor_time = {}".format(ii,  np.array(acor_time) ) +"\n")
            # except RuntimeError:
            #     f_log.write(" {}: Chain too short for acor to run".format(ii) +"\n")
            #     acor_time = None
            try:
                acor_time = sampler.get_autocorr_time(self, low=5, c=10)
                f_log.write("{:d}: acor_time = {}".format(ii,  np.array(acor_time) ) +"\n")
            except RuntimeError:
                f_log.write(" {}: Chain too short for acor to run".format(ii) +"\n")
                acor_time = None

            # --------------------------------
            # Case: test for convergence and truncate early:
            # Criteria checked: whether acceptance fraction within (minAF, maxAF),
            #                   and whether total number of steps > nEff * average autocorrelation time:
            #                   to make sure the paramter space is well explored.
            if ( (fitEmis2D.mcmcOptions.minAF is not None) & \
                    (fitEmis2D.mcmcOptions.maxAF is not None) & \
                    (fitEmis2D.mcmcOptions.nEff is not None) & \
                    (acor_time is not None)):
                if ((fitEmis2D.mcmcOptions.minAF < np.mean(sampler.acceptance_fraction) < fitEmis2D.mcmcOptions.maxAF) & \
                    (not fitEmis2D.mcmcOptions.runAllSteps) & \
                    (ii > np.max(acor_time) * fitEmis2D.mcmcOptions.nEff) ):
                    if ii == acor_force_min:
                        f_log.write(" Enforced min step limit: {}.".format(ii+1))
                    if ii >= acor_force_min:
                        f_log.write(" Finishing calculations early at step {}.".format(ii+1))
                        break


        # --------------------------------
        # Check if it converged before the max number of steps
        finishedSteps= ii+1
        if (finishedSteps == fitEmis2D.mcmcOptions.nSteps) & \
            ( (fitEmis2D.mcmcOptions.minAF is not None) & \
                (fitEmis2D.mcmcOptions.maxAF is not None) & \
                (fitEmis2D.mcmcOptions.nEff is not None) ) & \
               (not fitEmis2D.mcmcOptions.runAllSteps):
            f_log.write("Caution: no convergence within nSteps."+'\n')


        # --------------------------------
        # Finishing info for fitting:
        end = time.time()
        elapsed = end-start
        f_log.write("Finished {} steps".format(finishedSteps)+"\n")
        try:
            #acor_time = [acor.acor(sampler.chain[:,:,jj])[0] for jj in six.moves.xrange(sampler.dim)]
            acor_time = sampler.get_autocorr_time(self, low=5, c=10)
        except:
            acor_time = "Undefined, chain not converged"

        #######################################################################################
        # ***********
        # Consider overall acceptance fraction
        f_log.write('End: '+str(datetime.datetime.now())+'\n')
        f_log.write('\n')
        f_log.write('******************'+'\n')
        f_log.write('nCPU, nParam, nWalker, nSteps = {}, {}, {}, {} \n'.format(fitEmis2D.mcmcOptions.nCPUs,
                        fitEmis2D.kinModel.n_free_param,
                        fitEmis2D.mcmcOptions.nWalkers,
                        fitEmis2D.mcmcOptions.nSteps))
        f_log.write("keys={} \n".format(keys))
        f_log.write('Scale param a= {} \n'.format(fitEmis2D.mcmcOptions.scale_param_a))
        f_log.write('Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s) \n'.format(elapsed, np.floor(elapsed/60.),
                                        (elapsed/60.-np.floor(elapsed/60.))*60.))
        f_log.write("Mean acceptance fraction: {:0.3f} \n".format(np.mean(sampler.acceptance_fraction)))

        f_log.write("Ideal acceptance frac: 0.2 - 0.5 \n")\
        # Autocorrelation time:
        f_log.write("Acor est: ")
        f_log.write(str(acor_time))
        f_log.write('\n')
        f_log.write('******************\n')


        # --------------------------------
        # Save sampler, if output file set:
        #   Burn-in is already cut by resetting the sampler at the beginning.

        # Get pickleable format:
        sampler_dict = fit_io.make_emcee_sampler_dict(sampler, fitEmis2D=fitEmis2D, nBurn=0)

        #pool.close()
        sampler.pool.close()

        if fitEmis2D.mcmcOptions.filename_sampler is not None:
            # Save stuff to file, for future use:
            fit_io.dumppickle(sampler_dict, filename=fitEmis2D.mcmcOptions.filename_sampler)



    ##########################################
    ##########################################
    ##########################################

    # --------------------------------
    # Plot trace, if output file set
    if fitEmis2D.mcmcOptions.filename_plot_trace is not None:
        misfit_plot.plot_trace(sampler_dict, fitEmis2D,
                        fileout=fitEmis2D.mcmcOptions.filename_plot_trace)

    fitEmis2D.sampler_dict = sampler_dict

    return fitEmis2D

#
def _run_mcmc_emcee_3(fitEmis2D, fitEmis2D_fit=None):
    """
    Run emcee to do 2D kin fitting using fitEmis2D. Option to pass fitEmis2D_fit,
    a version of fitEmis2D that is very pared down, for faster copying+calculation.
    Otherwise will crudely get rid of most things in these classes before calculation.

    However, if the fitEmis2D, galaxy, instrument class have lots of specially-defined attributes,
        may want to fully setup model (eg, do fitEmis2D.setup_model(thetaSettings=thetaSettings))
        and then delete attributes

    Updated for emcee v3
    """
    # --------------------------------
    # Setup for threading for emcee, if desired
    if fitEmis2D.mcmcOptions.nCPUs is None:
        fitEmis2D.mcmcOptions.nCPUs = np.int(np.floor(psutil.cpu_count()*fitEmis2D.mcmcOptions.cpuFrac))
    if fitEmis2D.mcmcOptions.NoThread:
        fitEmis2D.mcmcOptions.nCPUs = 1


    # --------------------------------
    if fitEmis2D_fit is None:

        fitEmis2D_fit = make_pruned_fitEmis2D_class(fitEmis2D)


    # --------------------------------
    # Start pool, moves, backend:
    if (fitEmis2D.mcmcOptions.nCPUs > 1):
        pool = Pool(fitEmis2D.mcmcOptions.nCPUs)
    else:
        pool = None

    moves = emcee.moves.StretchMove(a=fitEmis2D.mcmcOptions.scale_param_a)

    backend_burn = emcee.backends.HDFBackend(fitEmis2D.mcmcOptions.filename_sampler_h5, name="burnin_mcmc")

    if overwrite:
        backend_burn.reset(nWalkers, nDim)

    # sampler_burn = emcee.EnsembleSampler(nWalkers, nDim, lnprob,
    #             backend=backend_burn, pool=pool, moves=moves,
    #             args=[gal], kwargs=kwargs_dict)
    #
    # nBurnCur = sampler_burn.iteration
    #
    # nBurn = nBurn_orig - nBurnCur


    # --------------------------------
    # Initialize walker starting positions
    keys = []
    for j in six.moves.xrange(len(fitEmis2D.kinModel.theta_names)):
        if fitEmis2D.kinModel.theta_vary[j]:
            keys.append(fitEmis2D.kinModel.theta_names[j])

    initial_pos = init_walker_pos(fitEmis2D, ndim=fitEmis2D.kinModel.n_free_param,
                    nwalkers=fitEmis2D.mcmcOptions.nWalkers)


    # --------------------------------
    # Initialize emcee sampler

    sampler_burn = emcee.EnsembleSampler(fitEmis2D.mcmcOptions.nWalkers, fitEmis2D.kinModel.n_free_param,
                                    lnprob,
                                    backend=backend_burn, pool=pool, moves=moves,
                                    args=(fitEmis2D_fit,))


    # --------------------------------
    # Start log file
    with utils.file_or_stdout(fitEmis2D.mcmcOptions.filename_log) as f_log:
        f_log.write('NoThread = {} \n'.format(fitEmis2D.mcmcOptions.NoThread))
        f_log.write('nCPUs: {} \n'.format(fitEmis2D.mcmcOptions.nCPUs))
        f_log.write('nSubpixels = {} \n'.format(fitEmis2D.kinModel.kinModelOptions.nSubpixels))
        f_log.write('PSF_type = {} \n'.format(fitEmis2D.instrument.PSF.PSF_type))
        f_log.write('PSF yspace_dither_arc = {} \n'.format(fitEmis2D.instrument.PSF.yspace_dither_arc))


        print('NoThread = {}'.format(fitEmis2D.mcmcOptions.NoThread))
        print('nCPUs: {}'.format(fitEmis2D.mcmcOptions.nCPUs))
        print('nSubpixels = {}'.format(fitEmis2D.kinModel.kinModelOptions.nSubpixels))
        print('PSF_type = {}'.format(fitEmis2D.instrument.PSF.PSF_type))
        print('PSF yspace_dither_arc = {}'.format(fitEmis2D.instrument.PSF.yspace_dither_arc))
        print("")

        ################################################################
        # --------------------------------
        # Run burn-in
        if fitEmis2D.mcmcOptions.nBurn > 0:
            f_log.write('Burn-in: \n')
            f_log.write('Start: {} \n'.format(datetime.datetime.now()))

            start = time.time()

            ####
            pos = initial_pos
            prob = None
            state = None
            for k in six.moves.xrange(fitEmis2D.mcmcOptions.nBurn):
                print( "k={:3d}, time: {}".format(k, datetime.datetime.now()) )
                pos = sampler_burn.run_mcmc(pos, 1)


            end = time.time()
            elapsed = end-start

            # try:
            #     acor_time = [acor.acor(sampler_burn.chain[:,:,jj])[0] for jj in six.moves.xrange(sampler.dim)]
            # except:
            #     acor_time = "Undefined, chain did not converge"


            acor_time = sampler_burn.get_autocorr_time(tol=10, quiet=True)

            #######################################################################################
            # Return Burn-in info
            # ****

            f_log.write('End: '+str(datetime.datetime.now())+'\n')
            f_log.write('\n')
            f_log.write('******************'+'\n')
            f_log.write('nCPU, nParam, nWalker, nBurn = {}, {}, {}, {} \n'.format(fitEmis2D.mcmcOptions.nCPUs,
                            fitEmis2D.kinModel.n_free_param,
                            fitEmis2D.mcmcOptions.nWalkers,
                            fitEmis2D.mcmcOptions.nBurn))
            f_log.write("keys={} \n".format(keys))
            f_log.write('Scale param a= {} \n' .format(fitEmis2D.mcmcOptions.scale_param_a))
            f_log.write('Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s) \n'.format(elapsed, np.floor(elapsed/60.),
                                (elapsed/60.-np.floor(elapsed/60.))*60.))
            f_log.write("Mean acceptance fraction: {:0.3f} \n ".format(np.mean(sampler.acceptance_fraction)))

            f_log.write("Ideal acceptance frac: 0.2 - 0.5 \n")
            # Autocorrelation time:
            f_log.write("Acor est: ")
            f_log.write(str(acor_time))
            f_log.write('\n')
            f_log.write('******************\n')
            f_log.write('\n')

            nBurn_nEff = 2
            try:
                if fitEmis2D.mcmcOptions.nBurn < np.max(acor_time) * nBurn_nEff:
                    f_log.write('#################\n')
                    f_log.write('nBurn is less than {}*acorr time \n'.format(nBurn_nEff))
                    f_log.write('#################\n')
                    # Give warning if the burn-in is less than say 2-3 times the autocorr time
            except:
                f_log.write('#################\n')
                f_log.write("acorr time undefined -> can't check convergence\n")
                f_log.write('#################\n')


            ##########################################
            ##########################################
            ##########################################
            # --------------------------------
            # Plot burn-in trace, if output file set
            if fitEmis2D.mcmcOptions.filename_plot_trace_burnin is not None:
                sampler_dict_burnin = fit_io.make_emcee_sampler_dict(sampler_burn, fitEmis2D=fitEmis2D, nBurn=0)
                misfit_plot.plot_trace(sampler_dict_burnin, fitEmis2D,
                                fileout=fitEmis2D.mcmcOptions.filename_plot_trace_burnin)
            ##########################################
            ##########################################
            ##########################################

            # Reset sampler after burn-in:
            #sampler.reset()

        else:
            # --------------------------------
            # No burn-in: set initial position:
            pos = np.array(initial_pos)
            prob = None
            state = None

        #######################################################################################
        # ****
        #######################################################################################
        # Setup sampler:
        # --------------------------------
        # Start backend:
        backend = emcee.backends.HDFBackend(fitEmis2D.mcmcOptions.filename_sampler_h5, name="mcmc")

        if overwrite:
            backend.reset(nWalkers, nDim)

        # sampler = emcee.EnsembleSampler(nWalkers, nDim, log_prob,
        #             backend=backend, pool=pool, moves=moves,
        #             args=[gal], kwargs=kwargs_dict)
        sampler = emcee.EnsembleSampler(fitEmis2D.mcmcOptions.nWalkers, fitEmis2D.kinModel.n_free_param,
                                    lnprob,
                                    backend=backend, pool=pool, moves=moves,
                                    args=(fitEmis2D_fit,))

        # --------------------------------
        # Run sampler: Get start time
        f_log.write('Ensemble sampling:\n')
        f_log.write('Start: {} \n'.format(datetime.datetime.now()))
        start = time.time()

        # --------------------------------
        # Run sampler: output info at each step
        for ii in six.moves.xrange(fitEmis2D.mcmcOptions.nSteps):
            pos_cur = pos.copy()    # copy just in case things are set strangely

            # --------------------------------
            # Only do one step at a time.
            pos = sampler.run_mcmc(pos, 1)
            # --------------------------------

            # --------------------------------
            # Give output info about this step:
            print( "ii={:3d}, a_frac={:0.4f}, time: {}".format(ii, np.mean(sampler.acceptance_fraction),
                            datetime.datetime.now()) )

            # try:
            #     acor_time = [acor.acor(sampler.chain[:,:,jj])[0] for jj in six.moves.xrange(sampler.dim)]
            #     f_log.write("{:d}: acor_time = {}".format(ii,  np.array(acor_time) ) +"\n")
            # except RuntimeError:
            #     f_log.write(" {}: Chain too short for acor to run".format(ii) +"\n")
            #     acor_time = None
            try:
                acor_time = sampler.get_autocorr_time(tol=10, quiet=True)
                f_log.write("{:d}: acor_time = {}".format(ii,  np.array(acor_time) ) +"\n")
            except RuntimeError:
                f_log.write(" {}: Chain too short for acor to run".format(ii) +"\n")
                acor_time = None

            # --------------------------------
            # Case: test for convergence and truncate early:
            # Criteria checked: whether acceptance fraction within (minAF, maxAF),
            #                   and whether total number of steps > nEff * average autocorrelation time:
            #                   to make sure the paramter space is well explored.
            if ( (fitEmis2D.mcmcOptions.minAF is not None) & \
                    (fitEmis2D.mcmcOptions.maxAF is not None) & \
                    (fitEmis2D.mcmcOptions.nEff is not None) & \
                    (acor_time is not None)):
                if ((fitEmis2D.mcmcOptions.minAF < np.mean(sampler.acceptance_fraction) < fitEmis2D.mcmcOptions.maxAF) & \
                    (not fitEmis2D.mcmcOptions.runAllSteps) & \
                    (ii > np.max(acor_time) * fitEmis2D.mcmcOptions.nEff) ):
                    if ii == acor_force_min:
                        f_log.write(" Enforced min step limit: {}.".format(ii+1))
                    if ii >= acor_force_min:
                        f_log.write(" Finishing calculations early at step {}.".format(ii+1))
                        break


        # --------------------------------
        # Check if it converged before the max number of steps
        finishedSteps= ii+1
        if (finishedSteps == fitEmis2D.mcmcOptions.nSteps) & \
            ( (fitEmis2D.mcmcOptions.minAF is not None) & \
                (fitEmis2D.mcmcOptions.maxAF is not None) & \
                (fitEmis2D.mcmcOptions.nEff is not None) ) & \
               (not fitEmis2D.mcmcOptions.runAllSteps):
            f_log.write("Caution: no convergence within nSteps."+'\n')


        # --------------------------------
        # Finishing info for fitting:
        end = time.time()
        elapsed = end-start
        f_log.write("Finished {} steps".format(finishedSteps)+"\n")
        try:
            #acor_time = [acor.acor(sampler.chain[:,:,jj])[0] for jj in six.moves.xrange(sampler.dim)]
            acor_time = sampler.get_autocorr_time(tol=10, quiet=True)
        except:
            acor_time = "Undefined, chain not converged"

        #######################################################################################
        # ***********
        # Consider overall acceptance fraction
        f_log.write('End: '+str(datetime.datetime.now())+'\n')
        f_log.write('\n')
        f_log.write('******************'+'\n')
        f_log.write('nCPU, nParam, nWalker, nSteps = {}, {}, {}, {} \n'.format(fitEmis2D.mcmcOptions.nCPUs,
                        fitEmis2D.kinModel.n_free_param,
                        fitEmis2D.mcmcOptions.nWalkers,
                        fitEmis2D.mcmcOptions.nSteps))
        f_log.write("keys={} \n".format(keys))
        f_log.write('Scale param a= {} \n'.format(fitEmis2D.mcmcOptions.scale_param_a))
        f_log.write('Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s) \n'.format(elapsed, np.floor(elapsed/60.),
                                        (elapsed/60.-np.floor(elapsed/60.))*60.))
        f_log.write("Mean acceptance fraction: {:0.3f} \n".format(np.mean(sampler.acceptance_fraction)))

        f_log.write("Ideal acceptance frac: 0.2 - 0.5 \n")\
        # Autocorrelation time:
        f_log.write("Acor est: ")
        f_log.write(str(acor_time))
        f_log.write('\n')
        f_log.write('******************\n')


        # --------------------------------
        # Save sampler, if output file set:
        #   Burn-in is already cut by resetting the sampler at the beginning.

        # Get pickleable format:
        sampler_dict = fit_io.make_emcee_sampler_dict(sampler, fitEmis2D=fitEmis2D, nBurn=0)

        # #pool.close()
        # sampler.pool.close()
        if fitEmis2D.mcmcOptions.nCPUs > 1:
            pool.close()
            sampler.pool.close()
            sampler_burn.pool.close()

        if fitEmis2D.mcmcOptions.filename_sampler is not None:
            # Save stuff to file, for future use:
            fit_io.dumppickle(sampler_dict, filename=fitEmis2D.mcmcOptions.filename_sampler)



    ##########################################
    ##########################################
    ##########################################

    # --------------------------------
    # Plot trace, if output file set
    if fitEmis2D.mcmcOptions.filename_plot_trace is not None:
        misfit_plot.plot_trace(sampler_dict, fitEmis2D,
                        fileout=fitEmis2D.mcmcOptions.filename_plot_trace)

    fitEmis2D.sampler_dict = sampler_dict

    return fitEmis2D


def make_pruned_fitEmis2D_class(fitEmis2D):
    galaxy = GalaxyBasic()
    spec2D = ObsSpectrum2DBasic()

    # Initialize some needed keys:
    spec2D.flux_ratio_arr = None
    spec2D.lam0 = None
    spec2D.m0 = None
    spec2D.wh_nosky = None
    spec2D.restwave_arr = None
    spec2D.wave = None
    spec2D.fitting_weight_matrix = None

    # Delete a few unneeded attribs
    del spec2D.units_flux
    del spec2D.units_wave
    del spec2D.band
    del spec2D.slit_PA

    galaxy.set_spectrum_2D(spec2D)

    instrument = fitEmis2D.instrument.copy()
    del instrument.instrument_name
    del instrument.band


    fitEmis2D_fit = FitEmissionLines2DBasic()

    # Copy over relevent attributes of galaxy:
    for key in galaxy.__dict__.keys():
        if key != 'spec2D':
            galaxy.__dict__[key] = copy.deepcopy(fitEmis2D.galaxy.__dict__[key])

    # Copy over relevent attributes of galaxy.spec2D:
    for key in galaxy.spec2D.__dict__.keys():
        galaxy.spec2D.__dict__[key] = copy.deepcopy(fitEmis2D.galaxy.spec2D.__dict__[key])

    # Copy over relevent attributes of fitEmis2D_fit:
    whitelist_keys = ['kinModelOptions', 'thetaPrior','thetaSettings']
    for key in whitelist_keys:
        fitEmis2D_fit.__dict__[key] = copy.deepcopy(fitEmis2D.__dict__[key])

    # Initialize kinModel:
    fitEmis2D_fit.kinModel = KinModel2D(galaxy,
                    thetaSettings=fitEmis2D_fit.thetaSettings,
                    kinProfile=copy.deepcopy(fitEmis2D.kinProfile),
                    intensityProfile=copy.deepcopy(fitEmis2D.intensityProfile),
                    kinModelOptions=fitEmis2D_fit.kinModelOptions)
    #
    ## Initialize parameters to be fit:
    theta_fitting_init = initialize_fitting_theta(fitEmis2D_fit.kinModel.theta_init,
                            fitEmis2D_fit.kinModel.theta_vary)

    ## Make an initial model:
    fitEmis2D_fit.make_model(galaxy=galaxy, instrument=instrument,
                                theta_fitting=theta_fitting_init)

    # Delete a few unnecessary things after
    del fitEmis2D_fit.thetaSettings
    del fitEmis2D_fit.kinModelOptions
    del fitEmis2D_fit.kinModel.aperModel.kinProfile.theta_names_nice
    del fitEmis2D_fit.kinModel.aperModel.kinProfile.theta_names
    del fitEmis2D_fit.thetaPrior.name

    keys_kinmodel = ['kinModelOptions', 'kinProfile', 'intensityProfile',
                        'bestfit', 'theta_names_nice', 'theta_names',
                        'theta_init','do_position_wave_shift']

    for key in keys_kinmodel:
        del fitEmis2D_fit.kinModel.__dict__[key]


    return fitEmis2D_fit




def init_walker_pos(fitEmis2D, ndim=None, nwalkers=200):
    # Initialize walker positions randomly within rough bounds: need theta_bounds to be set
    #   even if it's infinite -- to define *something* to start.

    param_range, param_lower = utils.range_arrs(fitEmis2D)

    pos = [np.random.random_sample(ndim)*param_range + param_lower for i in six.moves.xrange(nwalkers)]

    return pos


#
def get_chain_results(fitEmis2D):
    # Get V(R_E), V_2.2:
    fitEmis2D = add_v_re_22(fitEmis2D)

    fitEmis2D = get_param_posterior_bestfits(fitEmis2D)

    return fitEmis2D

#
def add_v_re_22(fitEmis2D):
    """
    Add velocity, dispersion at R_E and r_2.2 (eg, 2.2*r_s, or 2.2/1.67*R_E assuming n=1)
    """

    if fitEmis2D.kinModelOptions.absvalsigma:
        # Take abs of dispersion chain values before continuing
        if fitEmis2D.kinModel.kinProfile.dispProfile.n_params == 1:
            ind_all = fitEmis2D.kinModel.kinProfile.velProfile.n_params
            j = 0
            free_inds = []
            for i in six.moves.xrange(len(fitEmis2D.kinModel.theta)):
                if fitEmis2D.kinModel.theta_vary[i]:
                    free_inds.append(j)
                    j += 1
                else:
                    free_inds.append(None)
            ind = free_inds[ind_all]
            fitEmis2D.sampler_dict['flatchain'][:, ind] = \
                   np.abs(fitEmis2D.sampler_dict['flatchain'][:, ind])
            fitEmis2D.sampler_dict['chain'][:, :, ind] = \
                   np.abs(fitEmis2D.sampler_dict['chain'][:, :, ind])
        else:
            raise ValueError

    i_free = 0
    len_chain = fitEmis2D.sampler_dict['flatchain'].shape[0]
    for i in six.moves.xrange(len(fitEmis2D.kinModel.theta)):
        if fitEmis2D.kinModel.theta_vary[i]:
            if i > 0:
                theta_chain = np.append(theta_chain,
                                np.array([fitEmis2D.sampler_dict['flatchain'].T[i_free]]), axis=0)
            else:
                theta_chain = np.array([fitEmis2D.sampler_dict['flatchain'].T[i_free]])
            i_free += 1
        else:
            if i > 0:
                theta_chain = np.append(theta_chain,
                            np.array([np.repeat(fitEmis2D.kinModel.theta[i], len_chain)]), axis=0)
            else:
                theta_chain = np.array([np.repeat(fitEmis2D.kinModel.theta[i], len_chain)])


    fitEmis2D.kinProfile.update_theta(theta_chain)

    # Sample at R_E, and at z=0.
    V_re_arr = np.array([fitEmis2D.kinProfile.vel(fitEmis2D.galaxy.re_mass_arcsec, 0.)])
    V_22_arr = np.array([fitEmis2D.kinProfile.vel(2.2/1.676 * fitEmis2D.galaxy.re_mass_arcsec, 0.)])

    flatchain = np.append(fitEmis2D.sampler_dict['flatchain'].T,V_re_arr, axis=0).T
    flatchain = np.append(flatchain.T,V_22_arr, axis=0).T


    # Also add stuff about sigma: # Sample at R_E, and at z=0.
    sigma_re_arr = np.array([fitEmis2D.kinProfile.sigma(fitEmis2D.galaxy.re_mass_arcsec, 0.)])
    sigma_22_arr = np.array([fitEmis2D.kinProfile.sigma(2.2/1.676 * fitEmis2D.galaxy.re_mass_arcsec, 0.)])

    flatchain = np.append(flatchain.T, sigma_re_arr, axis=0).T
    flatchain = np.append(flatchain.T, sigma_22_arr, axis=0).T



    fitEmis2D.sampler_dict['flatchain'] = flatchain

    theta_names = []
    theta_names_nice = []

    for i in six.moves.xrange(len(fitEmis2D.kinModel.theta_names)):
        if fitEmis2D.kinModel.theta_vary[i]:
            theta_names.append(fitEmis2D.kinModel.theta_names[i])
            theta_names_nice.append(fitEmis2D.kinModel.theta_names_nice[i])

    # Add the added stuff:
    theta_names.append('V_RE')
    theta_names_nice.append(r'$V(R_E)$')
    theta_names.append('V_22')
    theta_names_nice.append(r'$V_{2.2}$')

    theta_names.append('sigma_RE')
    theta_names_nice.append(r'$\sigma(R_E)$')
    theta_names.append('sigma_22')
    theta_names_nice.append(r'$\sigma_{2.2}$')


    fitEmis2D.mcmc_results.theta_names = theta_names
    fitEmis2D.mcmc_results.theta_names_nice = theta_names_nice


    return fitEmis2D

#
def get_param_posterior_bestfits(fitEmis2D):
    samples = fitEmis2D.sampler_dict['flatchain'].copy()

    # Unpack MCMC samples: lower, upper 1, 2 sigma
    mcmc_limits = np.percentile(samples, [15.865, 84.135], axis=0)
    mcmc_limits_2sig = np.percentile(samples, [2.275, 97.725], axis=0)

    ## location of peaks of *marginalized histograms*
    ##      for each parameter
    mcmc_peak_hist = np.zeros(samples.shape[1])
    for i in six.moves.xrange(samples.shape[1]):
        yb, xb = np.histogram(samples[:,i], bins=50)
        wh_pk = np.where(yb == yb.max())[0][0]
        mcmc_peak_hist[i] = np.average([xb[wh_pk], xb[wh_pk+1]])
        # Compare w/ median of histogram: case of narrow peaks near edges:
        q50 = np.percentile(samples[:,i], 50.)
        if np.abs(mcmc_peak_hist[i]-q50)/(samples[:,i].max()-samples[:,i].min()) > 0.15:
            # if off by > 15%: seed with q50 instead:
            mcmc_peak_hist[i] = q50

    ## Use max prob as guess to get peak value of gaussian KDE, to find 'best-fit' of posterior:
    mcmc_peak_KDE = utils.find_peak_gaussian_KDE(fitEmis2D.sampler_dict['flatchain'], mcmc_peak_hist)

    # Set best-fit values
    fitEmis2D.mcmc_results.bestfit_theta = mcmc_peak_KDE

    mcmc_stack = np.concatenate(([mcmc_peak_KDE], mcmc_limits), axis=0)
    # Order: best fit value, lower 1sig bound, upper 1sig bound

    mcmc_uncertainties_1sig = np.array(list(map(lambda v: (v[0]-v[1], v[2]-v[0]),
                        list(zip(*mcmc_stack)))))
    # 1sig lower, upper uncertainty

    mcmc_stack_2sig = np.concatenate(([mcmc_peak_KDE], mcmc_limits_2sig), axis=0)
    mcmc_uncertainties_2sig = np.array(list(map(lambda v: (v[0]-v[1], v[2]-v[0]),
                        list(zip(*mcmc_stack_2sig)))))

    # Set +- 1 sig uncertainties: lower, upper
    fitEmis2D.mcmc_results.err_theta_1sig = mcmc_uncertainties_1sig
    fitEmis2D.mcmc_results.err_theta_2sig = mcmc_uncertainties_2sig

    #############################################################################
    # Recalculate some best-fit values for objects which should be analyzed together:
    #       eg, theta_linked_posteriors.
    # eg, for V_a and r_t mcmc based on best-fit value for V(R_E):
    if fitEmis2D.theta_linked_posteriors is not None:
        mcmc_lims_zip = list(zip(*mcmc_limits))

        for i in six.moves.xrange(len(fitEmis2D.theta_linked_posteriors)):
            mcmc_peak, mcmc_uncertainties_1sig = \
                    utils.get_bestfit_values_linked(fitEmis2D,
                    mcmc_uncertainties_1sig, mcmc_lims_zip,
                    theta_linked_posteriors=fitEmis2D.theta_linked_posteriors[i]) #, debug=True)
            fitEmis2D.mcmc_results.bestfit_theta = mcmc_peak
            fitEmis2D.mcmc_results.err_theta_1sig = mcmc_uncertainties_1sig

            mcmc_lims_zip = list(zip(*mcmc_limits_2sig))
            mcmc_peak, mcmc_uncertainties_2sig = \
                    utils.get_bestfit_values_linked(fitEmis2D,
                    mcmc_uncertainties_2sig, mcmc_lims_zip,
                    theta_linked_posteriors=fitEmis2D.theta_linked_posteriors[i]) #, debug=True)
            fitEmis2D.mcmc_results.err_theta_2sig = mcmc_uncertainties_2sig

    #############################################################################
    ## Get percent other sign:
    # v_re_chain = fitEmis2D.sampler_dict['flatchain'][:,-2].copy()
    # vre_best = fitEmis2D.mcmc_results.bestfit_theta[-2]
    # v_re_chain *= np.sign(vre_best)
    # percent_oppsign = len(np.where(v_re_chain < 0.)[0])/np.float(len(v_re_chain)) * 100.
    # fitEmis2D.mcmc_results.percent_oppsign = percent_oppsign

    # For all params.
    opp_flatchain = np.sign(fitEmis2D.mcmc_results.bestfit_theta)*\
                fitEmis2D.sampler_dict['flatchain'].copy()
    percent_oppsign = np.array([])
    for i in six.moves.xrange(len(fitEmis2D.mcmc_results.bestfit_theta)):
        perct_opp = len(np.where(opp_flatchain[:,i] < 0.)[0])/np.float(len(opp_flatchain[:,i])) * 100.
        percent_oppsign = np.append(percent_oppsign, perct_opp)

    fitEmis2D.mcmc_results.percent_oppsign = percent_oppsign


    # Clarify which were the free-parameter values:
    fitEmis2D.mcmc_results.bestfit_theta_fitting = \
                    fitEmis2D.mcmc_results.bestfit_theta[:fitEmis2D.kinModel.n_free_param]

    return fitEmis2D

class MCMC2DOptions(object):
    """
    Class to hold options for 2D MCMC fitting, including output filenames.
    """
    def __init__(self, **kwargs):

        # Options for running MCMC:
        self.cpuFrac = 0.75     # Fraction of CPUs to use
        self.nCPUs = None       # Number of CPUs to use
        self.nWalkers=2000      #
        self.nSteps=100         # fixed number of steps, or highest num of steps if using nEff
        self.nBurn=50           # Number of burn-in steps to run
        self.scale_param_a=3.0  # emcee scale paramter a
        self.nEff=10            # number of Acorr times to run, at minimum
        self.minAF=0.2          # Minimum acceptance fraction
        self.maxAF=0.5          # Maximum   "     "
        self.NoThread=False     # Option to not multithread
        self.runAllSteps=False  # Switch on to force run of all steps even if min,maxAF are specified


        # Optional output filenames for MCMC:
        self.filename_log = None
        self.filename_sampler = None
        self.filename_bestfit_model = None
        self.filename_plot_trace_burnin = None
        self.filename_plot_trace = None
        self.filename_plot_param_corner = None
        self.filename_plot_bestfit = None

        self.setAttr(**kwargs)

    def setAttr(self,**kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))

    def copy(self):
        return copy.deepcopy(self)

class MCMCResults(object):
    """
    Class to hold results from 2D MCMC fitting:
        bestfit parameters, 1sigma bounds, 2 sigma bounds, etc
    """
    def __init__(self, **kwargs):
        self.bestfit_theta = None

        # Order: lower, upper uncertainty
        self.err_theta_1sig = None
        self.err_theta_2sig = None

        self.percent_oppsign = None

        self.setAttr(**kwargs)

    def setAttr(self,**kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))
