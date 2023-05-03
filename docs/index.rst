.. misfit documentation main file, created by
   sphinx-quickstart
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========================================
``misfit``: Misaligned kinematic fitting
========================================


``misfit`` is a package for the 2D fitting and 1D modeling of kinematic observations 
taken with galaxy-slit misalignments, using structural information derived from 
spatially-resolved ancillary imaging to model the effects of the misalignments. 

Simple analytic profiles for galaxy rotation curves, velocity dispersions, 
and light intensity distributions are used to construct 3D 
spatial models (including line-of-sight projection). 
These 3D models are used to construct a 4D model of the galaxy 
over three spatial and one spectral dimension. 


For 2D fitting, this model is then collapsed along the line-of-sight, 
convolved with the PSF, integrated over the slit spatial direction, 
and finally convolved in the spectral direction by the instrumental resolution. 

When interpreting spatially unresolved 1D line profiles, 
the 3D rotation, dispersion and intensity models  
are used to derive a correction from an observed line velocity dispersion 
to an intrinsic RMS velocity (combining both rotation and random dispersion). 
These correction factors depend on the 
galaxy structure and orientation relative to the slit, 
and also require the assumption of a fixed ratio of 
ordered-to-disordered motion (:math:`V/\sigma`).


.. However, by analyzing many such unresolved galaxies, it is possible to 
.. determine the ensemble-average $V/\sigma$.


This code was developed for the analysis of Keck/MOSFIRE spectra 
from MOSDEF survey, and used in the analysis presented in 
`Price et al. (2016)`_ and `Price et al. (2020)`_. 
Modifications have been made in prepartion for extending this code 
for use with JWST/NIRSpec MSA spectra.

Full details about this code can be found in Appendices A & B of `Price et al. (2016)`_. 
Testing of the 2D fitting algorithm is presented in Appendix A of 
`Price et al. (2020)`_. 



.. _Price et al. (2016): https://ui.adsabs.harvard.edu/abs/2016ApJ...819...80P/abstract
.. _Price et al. (2020): https://ui.adsabs.harvard.edu/abs/2020ApJ...894...91P/abstract



Repository
----------
`https://github.com/sedonaprice/misfit`_ 

.. _https://github.com/sedonaprice/misfit: https://github.com/sedonaprice/misfit



Acknowledgement
---------------

If you use ``misfit`` in a publication,
please cite `Price et al., 2016, ApJ 819 80`_ (presenting the methodology) 
and `Price et al., 2020, ApJ 894 91`_ (testing of the 2D fitting).


.. _Price et al., 2016, ApJ 819 80: https://ui.adsabs.harvard.edu/abs/2016ApJ...819...80P/abstract
.. _Price et al., 2020, ApJ 894 91: https://ui.adsabs.harvard.edu/abs/2020ApJ...894...91P/abstract


License
-------
This project is Copyright (c) Sedona Price and 
licensed under the terms of the BSD 3-Clause license. 
See the LICENSE on the `GitHub repository`_ for more information.


.. _GitHub repository: https://github.com/sedonaprice/misfit



   ..  TOMAKE.rst

.. toctree::
   :maxdepth: 1
   :caption: Reference/API:

   api.rst


.. 
  .. automodapi:: misfit


.. 
  .. toctree::
..    :maxdepth: 2
..    :caption: Tutorials:

..    tutorials/dysmalpy_quickstart_example.rst
..    tutorials/models/index.rst
..    tutorials/fitting/index.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
.. * :ref:`search`



