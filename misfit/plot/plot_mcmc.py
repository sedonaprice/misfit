# misfit/plot/plot_mcmc.py
# Module to contain plotting functions for MCMC analysis: 2D + 1D posterior distributions
#
# Copyright 2014-2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

# Hidden modules prepended with '_'
from __future__ import print_function

import numpy as _np
import os as _os
from astropy.extern import six as _six

import matplotlib
#matplotlib.use('agg')
try:
    _os.environ["DISPLAY"] 
except:
    matplotlib.use("agg")
    
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt

import corner

def plot_param_corner(fitEmis2D, fileout=None,verbose=False,showLinked=True):
    
    # Chop off the last two: this is sigma_RE, sigma_2.2
    
    
    # Grab the best-fit mcmc values (inclucing V(R_E)) to plot as the "truths"
    # Bestfit for params: fitEmis2D.mcmc_results.bestfit_theta
    truths = fitEmis2D.mcmc_results.bestfit_theta.copy()
    
    
    truths = truths[:-2]
    
    
    
    
    # 1, 2 sigma quantiles:  [.02275, 0.15865, 0.84135, .97725]
    samples = fitEmis2D.sampler_dict['flatchain'][:,:-2]
    
    
    fig = corner.corner(samples, labels=fitEmis2D.mcmc_results.theta_names_nice, 
                            quantiles= [.02275, 0.15865, 0.84135, .97725],
                            truths=truths,  
                            plot_datapoints=False,
                            show_titles=True, 
                            bins=40,
                            plot_contours=True,
                            verbose=verbose)
                            
    axes = fig.axes
    
    names = fitEmis2D.mcmc_results.theta_names_nice
    sig1err = _np.array(zip(*fitEmis2D.mcmc_results.err_theta_1sig.copy()))
    l68_unc = sig1err[0,:]
    u68_unc = sig1err[1,:]
    nFreeParam = len(truths)
    for i in _six.moves.xrange(nFreeParam):
        ax = axes[i*nFreeParam + i]
        # Format the quantile display.
        best = truths[i]
        q_m = l68_unc[i]
        q_p = u68_unc[i]
        title_fmt=".2f"
        fmt = "{{0:{0}}}".format(title_fmt).format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(fmt(best), fmt(q_m), fmt(q_p))
        
        # Add in the column name if it's given.
        if names is not None:
            title = "{0} = {1}".format(names[i], title)
        ax.set_title(title)
    
    
    # 
    # # Figure out which axis is which:
    # for i in _six.moves.xrange(len(axes)):
    #     axes[i].annotate('i='+str(i), (0.9,0.9), 
    #             xycoords='axes fraction', fontsize=12,
    #             color='red')
    
    
    # If linked results:
    if (fitEmis2D.theta_linked_posteriors is not None) & (showLinked):
        for j in _six.moves.xrange(len(fitEmis2D.theta_linked_posteriors)):
            ind = fitEmis2D.theta_linked_posteriors[j][0] + fitEmis2D.theta_linked_posteriors[j][1]*\
                    samples.shape[1]
        
            xlim = axes[ind].get_xlim()
            ylim = axes[ind].get_ylim()
            
            
            
            col2arr = _np.linspace(ylim[0], ylim[1],num=101)
            
            vre_best = truths[-2]
            v22_best = truths[-1]
        
            ###
            theta_orig = fitEmis2D.kinModel.kinProfile.theta.copy()
            len_tmp = len(col2arr)
            for i in _six.moves.xrange(len(fitEmis2D.kinModel.kinProfile.theta)):
                if i in fitEmis2D.theta_linked_posteriors[j]:
                    if i == fitEmis2D.theta_linked_posteriors[j][0]:
                        theta_tmp = _np.array([_np.ones(len_tmp)])
                    else:
                        # The fixed dimension:
                        theta_tmp = _np.append(theta_tmp, 
                                    _np.array([col2arr]), axis=0)
                else:
                    if i > 0:
                        theta_tmp = _np.append(theta_tmp, 
                                    _np.array([_np.repeat(fitEmis2D.kinModel.bestfit.theta[i], 
                                            len_tmp)]), axis=0)
                    else:
                        theta_tmp = _np.array([_np.repeat(fitEmis2D.kinModel.bestfit.theta[i], 
                                    len_tmp)])
                                    
            fitEmis2D.kinProfile.update_theta(theta_tmp)
            
            V_re_arr_dummy = fitEmis2D.kinModel.kinProfile.vel(fitEmis2D.galaxy.re_arcsec, 0.)
            V_22_arr_dummy = fitEmis2D.kinModel.kinProfile.vel(2.2/1.676 * fitEmis2D.galaxy.re_arcsec, 0.)
            
            
            fitEmis2D.kinProfile.theta = theta_orig
            fitEmis2D.kinProfile.update_theta(theta_orig)
            
            #This only works if col1 variable is multiplicative only on the velProfile.
            col1arr_re = vre_best/V_re_arr_dummy
            col1arr_22 = v22_best/V_22_arr_dummy
            
            axes[ind].plot(col1arr_re, col2arr, ls='-', lw=1, color='orange')
            axes[ind].plot(col1arr_22, col2arr, ls='-', lw=1, color='magenta')
            axes[ind].set_xlim(xlim)
            axes[ind].set_ylim(ylim)
    
    
    
    
    # Annotate R_E, n
    galparam_str = r'$n=%0.2f$, $R_E=%0.2f "$' % (fitEmis2D.galaxy.n, fitEmis2D.galaxy.re_arcsec)
    fig.gca().annotate(galparam_str, (0.01, 0.), xycoords='figure fraction', 
                            xytext=(0,2), textcoords='offset points', 
                            ha='left', va='bottom')
    
    # Annotate sigma
    wh_sig = _np.where(_np.array(fitEmis2D.mcmc_results.theta_names) == 'sigma0')[0][0]
    val = "%3.2f" % fitEmis2D.mcmc_results.bestfit_theta[wh_sig]
    lower = "%3.2f" % fitEmis2D.mcmc_results.err_theta_1sig[wh_sig][0]
    upper = "%3.2f" % fitEmis2D.mcmc_results.err_theta_1sig[wh_sig][1]
    sig_string = "$\sigma="+val+"^{"+upper+"}_{"+lower+"}$"
    fig.gca().annotate(sig_string, xy=(.33, 0.), xycoords="figure fraction", 
                        xytext=(0,2), textcoords="offset points",
                        ha="right", va="bottom")
                        
                        
    # Annotate V(R_E)
    wh_vre = _np.where(_np.array(fitEmis2D.mcmc_results.theta_names) == 'V_RE')[0][0]
    val = "%3.2f" % fitEmis2D.mcmc_results.bestfit_theta[wh_vre]
    lower = "%3.2f" % fitEmis2D.mcmc_results.err_theta_1sig[wh_vre][0]
    upper = "%3.2f" % fitEmis2D.mcmc_results.err_theta_1sig[wh_vre][1]
    vre_lower = fitEmis2D.mcmc_results.bestfit_theta[wh_vre] - \
                    fitEmis2D.mcmc_results.err_theta_1sig[wh_vre][0]
    vre_upper = fitEmis2D.mcmc_results.bestfit_theta[wh_vre] + \
                    fitEmis2D.mcmc_results.err_theta_1sig[wh_vre][1]
    if (_np.sign(vre_lower) != _np.sign(vre_upper)):
        keep_decision = r"$\mathrm{disp}$"
    else:
        keep_decision = r"$\mathrm{rot}$"
    vre_string = "$V(R_E)="+val+"^{"+upper+"}_{"+lower+"}$"+', '+keep_decision
    fig.gca().annotate(vre_string, xy=(.62, 0.), xycoords="figure fraction", 
                        xytext=(0,2), textcoords="offset points",
                        ha="right", va="bottom")
    
    
    # Annotate V(R_E) -- 2sigma cut!
    # Calculate 2sigma ranges:
    wh_vre = _np.where(_np.array(fitEmis2D.mcmc_results.theta_names) == 'V_RE')[0][0]
    
    percent_othersign = fitEmis2D.mcmc_results.percent_oppsign[wh_vre]
    
    val = "%3.2f" % fitEmis2D.mcmc_results.bestfit_theta[wh_vre]
    lower = "%3.2f" % fitEmis2D.mcmc_results.err_theta_2sig[wh_vre][0]
    upper = "%3.2f" % fitEmis2D.mcmc_results.err_theta_2sig[wh_vre][1]
    vre_lower = fitEmis2D.mcmc_results.bestfit_theta[wh_vre] - \
                    fitEmis2D.mcmc_results.err_theta_2sig[wh_vre][0]
    vre_upper = fitEmis2D.mcmc_results.bestfit_theta[wh_vre] + \
                    fitEmis2D.mcmc_results.err_theta_2sig[wh_vre][1]
    if (_np.sign(vre_lower) != _np.sign(vre_upper)):
        keep_decision = r"$\mathrm{disp}$"
    else:
        keep_decision = r"$\mathrm{rot}$"
    vre_string = r"$V(R_E)="+val+r"^{"+upper+r"}_{"+lower+r"}$, $2 \sigma$"
    vre_string = vre_string+", "+keep_decision+r', $%0.3f$' % percent_othersign + r'$ \% \, \mathrm{opp}$'
    fig.gca().annotate(vre_string, xy=(.99, 0.), xycoords="figure fraction", 
                        xytext=(0,2), textcoords="offset points",
                        ha="right", va="bottom")
    
    
    
    if fileout is not None:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_param_corner_specific_params(fitEmis2D, param_names=None, 
            fileout=None,verbose=False):
            
    
    
    
    return None
