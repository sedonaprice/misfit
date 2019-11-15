MISFIT: Misaligned Kinematic Fitting
-------------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge


Description
------------
2D and 1D kinematic fitting of galaxies misaligned with slits, using spatially-resolved imaging profiles.


Usage
------------

Main usage:
*   fitEmis2D = misfit.FitEmissionLines2D class
 -       (in fit.fit2D)
*   fitEmis2D.fit(thetaSettings=thetaSettings)
 -       if a thetaSettings class hasn't already been passed to fitEmis2D.
        
        

Dependencies
------------
* numpy/scipy/matplotlib
* astropy (version >= ????)
* emcee
* acor
* corner
* dill


License
-------

This project is Copyright (c) Sedona Price and licensed under the terms of the BSD 3-Clause license. See the LICENSE for more information.
