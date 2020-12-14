# misfit/fit1D.py
# Fit 1D emission lines for MISFIT
#
# Copyright 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

# Hidden modules prepended with '_'

from __future__ import print_function

import matplotlib as mpl
mpl.use('agg')
#mpl.rcParams['text.usetex'] = True #False

import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import copy
import os
import sys

import pickle
import six

from scipy.stats import norm

try:
    import misfit.general.general_utils as utils
    import misfit.general.io as io
    from misfit.model.emission_lines_model import EmissionLinesSpectrum1DModel
except:
    from ..general import general_utils as utils
    from ..general import io as io
    from ..model.emission_lines_model import EmissionLinesSpectrum1DModel


import lmfit

import astropy.constants as _constants
c_cgs = _constants.c.cgs.value
c_kms = c_cgs * 1.e-5 # from cm/s -> km/s
c_AA = c_cgs * 1.e8  # go from cm/s -> AA/s


class FitEmissionLines1D(object):
    """
    Fit the flux and velocity dispersion of a 1D spectrum, using a set of lines.



    Define a spectrum made of multiple linesets set of emission lines and profiles for 1D spectra of a single lineset
        (eg, Ha, OIII doublet, ...)

    Input:
        galaxy              galaxy class
            spec1D          1D spectrum for the galaxy
        instrument.instrument_resolution should be in km/s,
                    and is needed for using FitEmissionLines1D.calculate_inst_corr_vel_disp()

        names_arr:          eg: ['Halpha', 'NII'], ['OIII'], ['Hbeta']

        Fit parameters: set to initial value before fitting

        flux_arr:           flux in flam of brightest line for each in the set.
        z:                  redshift
        vel_disp:           dispersion of the line [km/s]

        Fixed fit params
        cont_order:    order of the continuum fit
        shape1D:            currently only 'gaussian' is supported

    Optional:
        linenames_arr =         set of linenames for each set of lines
        restwave_arr =          set of restwaves for each set of lines
        trim_restwave_range     restframe wavelength to trim the spectra for fitting
        trim_wave_paramfile     file with wavelength to trim the spectrum for fitting
        del_obs_lam:            +- obs angstroms for optimizing z ; default = 20.

    """
    def __init__(self, **kwargs):

        # self.galaxy = galaxy
        # self.instrument = instrument
        self.galaxy = None
        self.instrument = None

        self.z = None

        self.names_arr = None
        self.flux_arr = None
        self.vel_disp = None
        self.cont_coeff = None

        # Fitting options:
        self.shape1D = 'gaussian'
        self.cont_order = 1
        self.del_obs_lam = 20.  # +- obs angstroms for optimizing redshift
        self.trim_wave_paramfile = None
        self.trim_restwave_range = None  #

        # Number of realizations for error measurements:
        self.num_MC = 500


        # Store stuff about the lines that will be fit:
        self.linenames_arr = None   # only store the first one of every lineset
        self.restwave_arr = None

        self.setAttr(**kwargs)
        self.setup_spectrum()

    def setAttr(self,**kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))

        self.z = self.galaxy.z

        if self.shape1D != 'gaussian':
            raise ValueError('Non-gaussian line profiles are currently not supported')


        #
        if (self.linenames_arr is None):

            linenames_arr = []

            # set this from lib if not set:
            d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
            names_file = os.path.join(d, 'line_names_cat.dat')
            for name in self.names_arr:
                linename_arr_tmp = io.read_line_names(names_file, name=name)
                linenames_arr.append(linename_arr_tmp)
            self.linenames_arr = linenames_arr

        if (self.linenames_arr is not None) & (self.restwave_arr is None):
            d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
            wave_file = os.path.join(d, 'line_wavelengths_ratios.dat')
            waves_arr = []
            for linenames in self.linenames_arr:
                wave_arr = []
                for linename in linenames:
                    wave_arr.append(io.read_restwave(wave_file, linename=linename))
                waves_arr.append(wave_arr)

            self.restwave_arr = np.array(waves_arr)

    def setup_spectrum(self):
        if (self.trim_restwave_range is None) and (self.trim_wave_paramfile is None):
            # set this from lib if not set:
            d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
            trim_file = os.path.join(d, 'trim_wavelengths_1D.param')
        else:
            trim_file = self.trim_wave_paramfile

        self.galaxy.spec1D_trim = self.galaxy.spec1D.trim_spectrum_wavelength(self.galaxy,
                param_restwave_filename=trim_file,
                trim_restwave_range=self.trim_restwave_range,
                linename=self.names_arr[0])

        # and mask the edges:
        self.galaxy.spec1D_trim.mask_edges()

    #
    def make_params(self, flux_bound=False, vel_disp_bound=False):
        """
        Initialize the LMFIT fitting parameters class with values and fit ranges.
        """
        params = lmfit.Parameters()
        params.add('z', value=self.z, min=self.galaxy.z-self.del_obs_lam/self.restwave_arr[0][0],
                    max=self.galaxy.z+self.del_obs_lam/self.restwave_arr[0][0])

        if vel_disp_bound:
            params.add('vel_disp', value=self.vel_disp, min=0.)
        else:
            params.add('vel_disp', value=self.vel_disp)#, min=0.)

        if flux_bound:
            for i in six.moves.xrange(len(self.names_arr)):
                params.add('flux'+str(i), value=self.flux_arr[i], min=0.)
        else:
            for i in six.moves.xrange(len(self.names_arr)):
                params.add('flux'+str(i), value=self.flux_arr[i])

        for i in six.moves.xrange(self.cont_order+1):
            params.add('cont_coeff'+str(i), value=self.cont_coeff[i])

        return params

    def fit(self, reload_errors=False,
            noErrors=False, noPlot=False,
            plot_filename=None, err_filename=None,
            reload = False):
        """
        Use LMFIT to find best-fit parameters for the 1D spectra.

        Usage:
            fitEmis1D = FitEmissionLines1D(....)
            fitEmis1D.fit()
            z = fitEmis1D.z
            ...

        Options:
            noErrors:           don't calculate errors (Monte Carlo perturb.)
            noPlot:             don't plot fit diagnostic plot
            err_filename:       specify absolute output filename for MC value matrix
            plot_filename:      specify absolute output plot filename

        Returns:
            unpack by reading from FitEmissionLines1D parameters below

            FitEmissionLines1D.
                    z
                    vel_disp
                    flux_arr
                    cont_coeff
                    z_err
                    vel_disp_err
                    flux_arr_err
                    cont_coeff_err
        """

        if self.instrument.instrument_resolution < 0.:
            raise ValueError

        ########################################################
        # Initialize the model + parameters:

        # Initialize cont_coeff if it's missing:
        if self.cont_coeff is None:
            cont_coeff_tmp = np.array([])
            for i in six.moves.xrange(self.cont_order+1):
                cont_coeff_tmp = np.append(cont_coeff_tmp, 0.)
            self.cont_coeff = cont_coeff_tmp

        params = self.make_params()

        emisModel = EmissionLinesSpectrum1DModel(names_arr=self.names_arr,
                        cont_order=self.cont_order, shape1D=self.shape1D)

        # Do the fitting:
        result = lmfit.minimize(emisModel.residual1DProfile, params,
                args=(self.galaxy.spec1D_trim.obswave, self.galaxy.spec1D_trim.flux,
                    self.galaxy.spec1D_trim.flux_err, self.galaxy.spec1D_trim.spec_mask))
        if (result.params['vel_disp'].value < 0.) & (result.params['flux0'].value < 0.):
            params = self.make_params(vel_disp_bound=True, flux_bound=True)
            result = lmfit.minimize(emisModel.residual1DProfile, params,
                args=(self.galaxy.spec1D_trim.obswave, self.galaxy.spec1D_trim.flux,
                    self.galaxy.spec1D_trim.flux_err, self.galaxy.spec1D_trim.spec_mask))
        elif (result.params['vel_disp'].value < 0.):
            params = self.make_params(vel_disp_bound=True)
            result = lmfit.minimize(emisModel.residual1DProfile, params,
                args=(self.galaxy.spec1D_trim.obswave, self.galaxy.spec1D_trim.flux,
                    self.galaxy.spec1D_trim.flux_err, self.galaxy.spec1D_trim.spec_mask))
        elif (result.params['flux0'].value < 0.):
            params = self.make_params(flux_bound=True)
            result = lmfit.minimize(emisModel.residual1DProfile, params,
                args=(self.galaxy.spec1D_trim.obswave, self.galaxy.spec1D_trim.flux,
                    self.galaxy.spec1D_trim.flux_err, self.galaxy.spec1D_trim.spec_mask))

        ########################################################
        # Save the restuls:
        self.lmfit_result = result

        # save the model result for quick plotting:
        self.final_model = emisModel.make1DProfile(self.lmfit_result.params, self.galaxy.spec1D_trim.obswave)

        # Unpack the final values to return:
        self.z = self.lmfit_result.params['z'].value
        self.vel_disp = self.lmfit_result.params['vel_disp'].value


        cont_coeff = np.array([])
        for i in six.moves.xrange(self.cont_order+1):
            cont_coeff = np.append(cont_coeff, self.lmfit_result.params['cont_coeff'+str(i)].value)
        self.cont_coeff = cont_coeff

        flux_arr = np.array([])
        for i in six.moves.xrange(len(self.names_arr)):
            flux_arr = np.append(flux_arr, self.lmfit_result.params['flux'+str(i)].value)
        self.flux_arr = flux_arr

        values_true = np.array([self.z, self.vel_disp])
        values_true = np.append(values_true, self.flux_arr)
        values_true = np.append(values_true, self.cont_coeff)

        print( "-----------------------------------------------------------------------------")
        print( "{} {:d}".format(self.galaxy.field, self.galaxy.ID) )
        print( "Message: \t\t {}".format(self.lmfit_result.message) )
        print( "Success: \t\t {}".format(self.lmfit_result.success) )
        print( "Errorbars estimated?: \t {}".format( self.lmfit_result.errorbars) )
        print( "fit report:")
        #print( lmfit.report_fit(self.lmfit_result.params))
        print(lmfit.fit_report(self.lmfit_result))
        print( "-----------------------------------------------------------------------------")

        ########################################################
        # Setup plot for bestfit+errors: start by plotting error fits over data
        if not noPlot:
            # import matplotlib as mpl
            # #mpl.use('Agg')
            # mpl.rcParams['text.usetex'] = True #False
            #
            # import matplotlib.pyplot as plt

            if plot_filename is None:
                raise ValueError("Must set plot_filename")


            f, ax = plt.subplots()
            f.set_size_inches(5.,4.)


            ###########

            for i in six.moves.xrange(len(self.names_arr)):
                for j in six.moves.xrange(len(self.restwave_arr[i])):
                    ax.axvline(x=(1.+self.z)*self.restwave_arr[i][j],
                                ls='-', color='black', alpha=0.8,zorder=-15)
                    ax.axvline(x=(1.+self.galaxy.z)*self.restwave_arr[i][j],
                                ls='--', color='lightgrey', alpha=0.8,zorder=-10)

            wh_notmask = np.where(self.galaxy.spec1D_trim.spec_mask == 1.)[0]
            wh_mask =  np.where(self.galaxy.spec1D_trim.spec_mask == 0.)[0]

            ms = 3
            elinewidth = 0.5 #0.75
            capsize = 1.5 #3
            capthick = elinewidth
            p_err = ax.errorbar(self.galaxy.spec1D_trim.obswave[wh_notmask],
                    self.galaxy.spec1D_trim.flux[wh_notmask],
                    yerr=self.galaxy.spec1D_trim.flux_err[wh_notmask],
                    elinewidth=elinewidth, capsize=capsize, capthick=capthick,
                    ms=ms, marker='o', ls='None', color='k', ecolor='k', zorder=-1.)
            p_err_mask = ax.errorbar(self.galaxy.spec1D_trim.obswave[wh_mask],
                    self.galaxy.spec1D_trim.flux[wh_mask],
                    yerr=self.galaxy.spec1D_trim.flux_err[wh_mask],
                    elinewidth=elinewidth, capsize=capsize, capthick=capthick,
                    ms=ms, marker='o', ls='None', color='grey', ecolor='grey', zorder=-1.5)

        ########################################################
        if not noErrors:

            if reload_errors:
                value_matrix = self.load_mc_sim_matrix(line=self.names_arr[0],
                                        err_filename=err_filename)


            else:
                # MC: perturb each spec pt by gaussian random number with 1sig=error
                # then do lmfit for each realization:
                value_matrix = np.zeros((self.num_MC, len(self.lmfit_result.params)))

                # Structure of value matrix: columns:
                #  z_fit   vel_disp   flux_line0 ... flux_linen-1  cont_coeff0 .. cont_coeffn-1

                for i in six.moves.xrange(self.num_MC):
                    #print "MC error iter %i/%i" % (i+1,self.num_MC)
                    if ( ((i+1) % 50 == 0)):
                        print("MC error iter {:3d}/{:3d}".format(i+1,self.num_MC))

                    spec_perturb = self.galaxy.spec1D_trim.flux.copy()
                    # Now perturb randomly, using normal distribution
                    rand_vals = np.random.randn(len(spec_perturb))
                    spec_perturb += self.galaxy.spec1D_trim.flux_err*rand_vals

                    params_best = lmfit.Parameters()
                    params_best.add('z', value=self.z, min=self.galaxy.z-self.del_obs_lam/self.restwave_arr[0][0],
                                max=self.galaxy.z+self.del_obs_lam/self.restwave_arr[0][0])
                    params_best.add('vel_disp', value=self.vel_disp)#, min=0.)

                    for jj in six.moves.xrange(len(self.names_arr)):
                        params_best.add('flux'+str(jj), value=self.flux_arr[jj])

                    for jj in six.moves.xrange(self.cont_order+1):
                        params_best.add('cont_coeff'+str(jj), value=self.cont_coeff[jj])

                    #now fit the perturbed spectrum:
                    result_mc = lmfit.minimize(emisModel.residual1DProfile, params_best,
                            args=(self.galaxy.spec1D_trim.obswave, spec_perturb,
                                self.galaxy.spec1D_trim.flux_err, self.galaxy.spec1D_trim.spec_mask))


                    if not noPlot:
                        model_mc = emisModel.make1DProfile(result_mc.params, self.galaxy.spec1D_trim.obswave)
                        ax.plot(self.galaxy.spec1D_trim.obswave, model_mc,
                                ls='-', lw=1, c='red', alpha=0.025)

                    value_matrix[i,0] = result_mc.params['z'].value
                    value_matrix[i,1] = result_mc.params['vel_disp'].value
                    jj = 1
                    for j in six.moves.xrange(len(self.names_arr)):
                        jj += 1
                        value_matrix[i,jj] = result_mc.params['flux'+str(j)].value
                    for j in six.moves.xrange(self.cont_order+1):
                        jj += 1
                        value_matrix[i,jj] = result_mc.params['cont_coeff'+str(j)].value

                    ###

                    # Red chisq value: result_mc.redchi
                #
                self.save_mc_sim_matrix(value_matrix, line=self.names_arr[0],
                                        err_filename=err_filename)

            # Get lower, upper 1 sig values for each param
            values_err = np.zeros((len(self.lmfit_result.params),2))
            limits = np.percentile(value_matrix, [15.865, 84.135], axis=0).T
            values_err[:,0] = values_true[:] - limits[:,0]
            values_err[:,1] = limits[:,1] - values_true[:]

            # Lower, upper errors
            self.z_err =            values_err[0,:]
            self.vel_disp_err =     values_err[1,:]
            self.flux_arr_err =     values_err[2:len(self.names_arr)+2,:]
            self.cont_coeff_err =   \
                values_err[len(self.names_arr)+2:len(self.names_arr)+2+self.cont_order+1,:]



        else:
            print( "Not doing errors")
            self.z_err =            np.zeros(2)
            self.vel_disp_err =     np.zeros(2)
            self.flux_arr_err =     np.zeros((len(self.names_arr),2))
            self.cont_coeff_err =   np.zeros((self.cont_order+1,2))


        #############################################################
        #  Finish plot
        if not noPlot:

            cont_model = np.zeros(len(self.galaxy.spec1D_trim.obswave))
            for i in six.moves.xrange(self.cont_order+1):
                cont_model += self.lmfit_result.params['cont_coeff'+str(i)].value*1.e-18*\
                            np.power(self.galaxy.spec1D_trim.obswave,i)

            ax.plot(self.galaxy.spec1D_trim.obswave, cont_model, lw=1, color='green')


            ax.plot(self.galaxy.spec1D_trim.obswave, self.final_model, ls='-', lw=1, c='blue')

            ax.set_xlabel(r'$\lambda$')
            ax.set_ylabel(r'$F_{\lambda}$')
            names_cat = ''
            for j in six.moves.xrange(len(self.names_arr)):
                if j > 0:
                    names_cat += '+'
                names_cat += self.names_arr[j]


            #
            xpos = 0.035
            ypos = 0.915
            ydel = 0.07#5
            if not noErrors:
                string = r'$z = %3.3f_{-%3.3f}^{+%3.3f}$' % \
                        (self.z, self.z_err[0], self.z_err[1])
            else:
                string = r'$z = %3.3f$' % (self.z)
            ax.annotate(string, (xpos,ypos), xycoords='axes fraction',
                    fontsize=9., backgroundcolor='white')
            ypos -= ydel
            if not noErrors:
                string = r'$\sigma_{\mathrm{vel\ disp}} = %3.3f_{-%3.3f}^{+%3.3f}$' % \
                    (self.vel_disp, self.vel_disp_err[0], self.vel_disp_err[1])
            else:
                string = r'$\sigma_{\mathrm{vel\ disp}} = %3.3f$' % (self.vel_disp)
            ax.annotate(string, (xpos,ypos), xycoords='axes fraction',
                    fontsize=9., backgroundcolor='white')


            xlim = np.array([self.galaxy.spec1D_trim.obswave.min()-20,
                        self.galaxy.spec1D_trim.obswave.max()+20])
            yrange = np.abs(self.final_model.max()-self.final_model.min())
            pad_frac=0.15
            ylim = np.array([self.final_model.min()-yrange*pad_frac,
                             self.final_model.max()+yrange*pad_frac])


            if self.galaxy.maskname is not None:
                title = '\_'.join(self.galaxy.maskname.split('_'))+'-'+str(self.galaxy.ID)+' '+names_cat
            else:
                title = self.galaxy.field+'-'+str(self.galaxy.ID)+' '+names_cat

            ax.set_title(title)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            print("Saving fitting plot to {}".format(plot_filename) )
            plt.savefig(plot_filename, dpi=600)  #bbox_inches='tight',
            plt.close(f)


            ######


    #
    def calculate_inst_corr_vel_disp(self,
                noErrors=False,
                err_filename=None):
        """
            Use instrument resoultion to calculate corrected velocity dispersion
            Requries FitEmissionLines1D.instrument.instrument_resolution to be
                set to instrument resolution (in km/s)
        """
        if self.instrument.instrument_resolution is None:
            print("Must set FitEmissionLines1D.instrument.instrument_resolution to use calculate_inst_corr_vel_disp()")
            return

        # Correct the vel_disp from inst resolution:
        if np.abs(self.vel_disp) > np.abs(self.instrument.instrument_resolution):
            self.vel_disp_inst_corr = np.sqrt(self.vel_disp**2 - self.instrument.instrument_resolution**2)
        else:
            self.vel_disp_inst_corr = 0.


        if not noErrors:
            if err_filename is None:
                raise ValueError("Must set err_filename")

            # Load in best-fit matrix, if set and it exists:
            value_matrix = self.load_mc_sim_matrix(line=self.names_arr[0],
                                    err_filename=err_filename)

            # Correct all in value matrix
            vel_disp_mc = value_matrix[:,1]
            vel_disp_mc_inst_corr = np.sqrt(vel_disp_mc**2 - self.galaxy.spec1D.instrument_resolution**2)

            # Make non-finite cases: these are where inst res > vel disp
            vel_disp_mc_inst_corr[~np.isfinite(vel_disp_mc_inst_corr)] = 0.

            # Get lower, upper 1 sig values for each param
            limits = np.percentile(vel_disp_mc_inst_corr, [15.865, 84.135])

            # Lower, upper errors
            vel_disp_inst_corr_err = np.zeros(2)
            vel_disp_inst_corr_err[0] = self.vel_disp_inst_corr - limits[0]
            vel_disp_inst_corr_err[1] = limits[1] - self.vel_disp_inst_corr

            if ~np.isfinite(np.average(limits)):
                print( "self.vel_disp_inst_corr={}".format(self.vel_disp_inst_corr) )
                raise ValueError('Limits on inst corr vel disp are not finite')

            self.vel_disp_inst_corr_err =     vel_disp_inst_corr_err

        else:
            self.vel_disp_inst_corr_err =   np.zeros(2)

    def save_mc_sim_matrix(self, value_matrix, line='none',
                    err_filename=None):
        """
        Save the MC error simulation fit values to a pickle
        """
        if err_filename is None:
            raise ValueError("Must set err_filename")

        pickle.dump(value_matrix, open(err_filename, "wb"))

        return None

    def load_mc_sim_matrix(self, line='none', err_filename=None):
        """
        Load the MC error simulation fit values from the pickle
        """
        if err_filename is None:
            raise ValueError("Must specifiy err_filename")

        value_matrix = pickle.load(open(err_filename, "rb"))

        return value_matrix
