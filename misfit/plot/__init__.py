# Copyright 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

try:
    from plot_bestfit import plot_bestfit_model
    from plot_image_slit import plot_image_slit, box_integrate_pstamp, box_integrate
    from plot_trace import plot_trace
    from plot_mcmc import plot_param_corner, plot_param_corner_specific_params
except:
    from .plot_bestfit import plot_bestfit_model
    from .plot_image_slit import plot_image_slit, box_integrate_pstamp, box_integrate
    from .plot_trace import plot_trace
    from .plot_mcmc import plot_param_corner, plot_param_corner_specific_params