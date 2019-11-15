# misfit/io.py
# File I/O handling for MISFIT
# 
# Copyright 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Hidden modules prepended with '_'
from __future__ import print_function

import numpy as _np
import os as _os
import sys as _sys


from astropy.io import ascii as _ascii

deg2rad = _np.pi/180.


def read_line_names(param_filename, name=None):
    """
    Read line set file: name of grouping and indiv names
        format: commented header: # name  lineset 
        
    Set strongest line of set to 1.
    """
    # Read the set of lines which are part of the single line
    
    data = _ascii.read(param_filename, format='commented_header')
    
    # print "param_filename=", param_filename
    # print "data=", data
    
    wh = _np.where(data['name'] == name)[0]
    if len(wh) == 0:
        raise ValueError("Problem with param file: {}, name: {}".format(param_filename, name))
    else:
        wh = wh[0]
    
    lineset = data['lineset'][wh]
    linename_arr = lineset.split(',')
    
    
    return linename_arr
    
    

def read_restwave(param_filename, linename=None):
    """
    Read line rest wavelengths
        format: commented header: # linename  restwave 
    """
    data = _ascii.read(param_filename, format='commented_header')
    
    wh = _np.where(data['linename'] == linename)[0]
    if len(wh) == 0:
        raise ValueError("Problem with param file: {}".format(param_filename) )
    else:
        wh = wh[0]
    
    
    return data['restwave'][wh]
    
def read_line_ratio(param_filename, linename=None):    
    """
    Read line set ratio file. 
        format: commented header: # linename  strength 
        
    Set strongest line of set to 1.
    """
    data = _ascii.read(param_filename, format='commented_header')
    
    wh = _np.where(data['linename'] == linename)[0]
    if len(wh) == 0:
        raise ValueError("Problem with param file: {}".format(param_filename) )
    else:
        wh = wh[0]
    
    
    return data['strength'][wh]


def read_wave_range(param_filename,linename=None):
    """
    Read fitting or other wavelength range from a parameters file. 
        format: commented header: # linename  wave_l   wave_u 
    """
    data = _ascii.read(param_filename, format='commented_header')
    
    wh = _np.where(data['linename'] == linename)[0]
    if len(wh) == 0:
        raise ValueError("Problem with param file: {}".format(param_filename) )
    else:
        wh = wh[0]
    
    trim_restwave_range = _np.array([data['wave_l'][wh], data['wave_u'][wh]])
    
    return trim_restwave_range
    
#
def read_skyline_band_cutoff(param_filename, band=None):
    """
    Read fitting or other wavelength range from a parameters file. 
        format: commented header: # band  sn_cutoff   units 
    """
    data = _ascii.read(param_filename, format='commented_header')
    
    wh = _np.where(data['band'] == band)[0]
    if len(wh) == 0:
        raise ValueError("Problem with param file: {}".format(param_filename) )
    else:
        wh = wh[0]
    
    return data['sn_cutoff'][wh], data['units'][wh]
    
    
def get_param_filename(filename):
    d = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'lib')
    return _os.path.join(d, filename)
