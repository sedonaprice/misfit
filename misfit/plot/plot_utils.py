# General utilities for handling data
# Copyright 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function

import pandas as pd
import numpy as np
import scipy as _sp

import six

from astropy.wcs import WCS
from astropy import coordinates as coords

from scipy.ndimage.interpolation import rotate as scipyrotate
from scipy.signal import fftconvolve

deg2rad = np.pi/180.


def pstamp_position_angle(pstamp_hdr):
    # try:
    #     return pstamp_hdr['ORIENTAT']
    # except:
    try:
        cd21 = pstamp_hdr['CD2_1']
    except:
        cd21 = 0.
    try:
        cd22 = pstamp_hdr['CD2_2']
    except:
        cd22 = 0.

    if (cd22 == 0.) and (cd21 == 0.):
        # Don't have any information, so you should return 0.
        return 0.
    else:
        if cd22 == 0.:
            return 90.
        else:
            return np.arctan2(cd21, cd22)*180./np.pi


def pos_angle(pos1, pos2, units='degrees'):
    # port of IDL posang, borrowing from agpy
    #input IRCS coords already:
    #pos1 = c_slit = IRCS(ra1, dec1, unit=(u.hour, u.degree))
    #c_targ = FK5((14, 19, 4.9), (52, 47, 36.2), unit=(u.hour, u.degree))

    ra1, dec1 = pos1.ra.rad, pos1.dec.rad
    ra2, dec2 = pos2.ra.rad, pos2.dec.rad

    radiff = (ra1-ra2)
    angle = np.arctan2(-np.sin(radiff), np.cos(dec1)*np.tan(dec2)
                - np.sin(dec1)*np.cos(radiff))

    if units=='degrees':
        return angle*180./np.pi
    elif units=='radians':
        return angle
    else:
        raise ValueError("invalid units: %s" % units)

def sign_from_pos_angle(pos_angle):
    ## Input in degrees
    if pos_angle < 0.:
        pos_angle += 360.

    if pos_angle > 180.:
        PA_old = pos_angle.copy()
        pos_angle = 360. - PA_old

    if pos_angle > 90.:
        return -1.

    if pos_angle < 90.:
        return 1.

    if pos_angle == 90.:
        # dot product is zero
        return 0.

def y_sign_pos_angle(pos_angle):
    # PA in degrees
    return sign_from_pos_angle(pos_angle)

def x_sign_pos_angle(pos_angle):
    # PA in degrees
    return sign_from_pos_angle(pos_angle-90.)


def rotate_pstamp(galaxy):
        """
        Rotate the pstamp so that the slit is vertical
            -- whether it's a galaxy in the pstamp struc or a star
        Input:
            mosdef_general.base object -- star or galaxy
        """
        try:
            # Here: care about slit_PA *ON IMAGE*
            # slit_PA_on_img = galaxy.spec2D.slit_PA - galaxy.pstamp.img_PA
            img_angle = galaxy.spec2D.slit_PA - galaxy.pstamp.img_PA
            #                     # scipy rotate needs positive angle to rotate CW,
                                  # opposite of convention otherwise used.
        except:
            img_angle = galaxy.spec1D.slit_PA - galaxy.pstamp.img_PA

        pstamp_rot = scipyrotate(galaxy.pstamp.pstamp, img_angle,
                axes=(1, 0),
                reshape=False, order=5)

        return pstamp_rot


def rot_coord_angle(arr, th, x0=0., y0=0.):
    """
    Input angle in radians.
    x0,y0 are center of rot.
    x1_off, y1_off are arbitrary offset to apply after transform.
    """
    # Rotate around zp if given:
    pos_arr = np.array([arr[0]-x0, arr[1]-y0])

    # Rotation direction: positively oriented (CCW is positive)
    x1 = pos_arr[0]*np.cos(th) - pos_arr[1]*np.sin(th)
    y1 = pos_arr[0]*np.sin(th) + pos_arr[1]*np.cos(th)

    return x1 + x0 , y1 + y0

def rot_corner_coords(coords, th, x0=0., y0=0.):
    """
    Rotate a rectangle, and return two arrs: xcorr, ycorr.
    Input: [[x1,y1],[x2,y2], [x3,y3], [x4,y4]]

    x0,y0 are center of rot.
    x1_off, y1_off are arbitrary *additative* offset to apply after transform.
    """
    corners = np.array([])
    for c in coords:
        cx, cy = rot_coord_angle(c, th, x0=x0, y0=y0)
        if np.shape(corners)[0] == 0:
            corners = np.array([cx, cy]).flatten()
        else:
            corners = np.vstack((corners, np.array([cx, cy]).flatten()))

    corners = np.array(corners)

    return corners.T[0], corners.T[1]

def rot_object_center(x0, y0, cent_pix, angle):
    return rot_corner_coords([[x0,y0]],
                angle, x0=cent_pix[0], y0=cent_pix[1])



#

def convolve_rot_gal(galaxy=None, instrument=None, instrument_img=None,
                    conv_sigma=None,verbose=False):
    # Conv_sigma: sigma to convolve input with to get desired output resolution
    # # Conv sigma in arcseconds

    # General utility:
    #   input the galaxy object
    # return rotated, convolved pstamp

    if verbose:
        print("convolve_rot_gal: conv_sigma={:0.3f}".format(conv_sigma) )

    ##################################################################

    w = WCS(galaxy.pstamp.pstamp_hdr)
    w.sip = None

    origin = 'lower'   # Origin of pstamp for imshow
    originWCS = 1     # FITS standard

    # x is column, y is row: arrays are row, column
    cent_pix= np.array([(np.shape(galaxy.pstamp.pstamp)[1]-1)/2.,
            (np.shape(galaxy.pstamp.pstamp)[0]-1)/2.])

    x0WCS, y0WCS = w.wcs_world2pix(galaxy.RA, galaxy.DEC, originWCS)
    x0WCS -= 1. # convert to np pixel coordinates
    y0WCS -= 1. # convert to np pixel coordinates

    try:
        slit_PA = galaxy.spec2D.slit_PA
    except:
        slit_PA = galaxy.spec1D.slit_PA


    x0_rot, y0_rot = rot_object_center(x0WCS, y0WCS,
                                cent_pix, -1.*slit_PA*deg2rad)
    # in np pixel coordinates

    # Rotate the pstamp so the slit is vertical
    pstamp_rot = rotate_pstamp(galaxy)

    ######################################
    ## Do convolution:
    #conv_sigma_pix = conv_sigma/instrument_img.pixscale

    pstamp_rot_orig = pstamp_rot.copy()
    sum_pstamp = np.sum(pstamp_rot_orig)

    # Make 2d gaussian:
    #conv_stamp = np.zeros(np.shape(pstamp_rot))
    x = np.linspace(0, np.shape(pstamp_rot)[1], num=np.shape(pstamp_rot)[1],
                endpoint=False)
    y = np.linspace(0, np.shape(pstamp_rot)[0], num=np.shape(pstamp_rot)[0],
                endpoint=False)
    #cent = [y0_rot, x0_rot]

    xoff = x - x0_rot
    yoff = y - y0_rot
    xoffarc = xoff * instrument_img.pixscale
    yoffarc = yoff * instrument_img.pixscale


    #
    conv_stamp = instrument.PSF.generate_conv_stamp(xoffarc, yoffarc,
                PSF_FWHM=conv_sigma*(2.*np.sqrt(2.*np.log(2.))))


    pstamp_rot = fftconvolve(pstamp_rot_orig, conv_stamp, mode='same')

    # Scale back to the total "flux" in the stamp:
    pstamp_rot *= sum_pstamp/np.sum(pstamp_rot)

    ######################################

    # Return rotated pstamp, with convolution:
    return pstamp_rot
