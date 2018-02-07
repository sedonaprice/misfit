# misfit/plot/plot_image_slit.py
# Copyright 2016, 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

from __future__ import print_function

import os as _os
import numpy as _np

from astropy.wcs import WCS as _WCS

import matplotlib
#matplotlib.use('agg')

try:
    _os.environ["DISPLAY"] 
except:
    matplotlib.use("agg")
    
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

from plot_utils import rotate_pstamp, rot_object_center, rot_corner_coords, convolve_rot_gal


from astropy.extern import six as _six

cmap = cm.gray_r
cmap_name = 'gray_r'

# Setup constants:
d2r = _np.pi/180.


def plot_image_slit(galaxy=None, instrument=None, 
        instrument_img=None, 
        m0=None, rotate=True, 
        convolve=False, conv_sigma=None, 
        n_spatial_pixels=None, 
        downsample=False,
        ax=None,saveToFile=False,
        fileout=None, show_galfit=True,
        shift_label=None, lw_slit=1,
        verbose=False ):
    """
    Input:
        Galaxy object
        Instrument object (Imaging)
        
    Optional:
        ax:             axes instance. If set, must reside in pre-defined figure, 
                          and already be created (using proper gridspec positioning, etc)
        saveToFile:     (True/False) Save current axis instance to file. 'fileout' must be set.
    """
    if galaxy.__dict__.has_key('re_arcsec'):
        plot_galfit = True
    else:
        plot_galfit = False
    if not show_galfit:
        plot_galfit = False
    
    if ax is None:
        external_ax = False
        fig, ax = plt.subplots(1)
        fig.set_size_inches(3.,3.)
    else:
        external_ax = True
        
    #
    # print "conv_sigma=", conv_sigma
    
    if convolve & (conv_sigma is None):
        # Define conv_sigma if not set: crude case
        conv_FWHM = _np.sqrt(instrument.PSF.PSF_FWHM**2 - instrument_img.PSF.PSF_FWHM**2)
        conv_sigma = conv_FWHM/(2.*_np.sqrt(2.*_np.log(2.)))
    
    ax.set_axis_off()
    ax.set_clip_on(True)
    
    ###############################################
    # Setup coordinates:
    
    w = _WCS(galaxy.pstamp.pstamp_hdr)
    w.sip = None
    
    origin = 'lower'   # Origin of pstamp for imshow
    origin_wcs = 1     # FITS standard
    
    x0_wcs_sci, y0_wcs_sci = w.wcs_world2pix(galaxy.RA, galaxy.DEC, origin_wcs)
    x0_wcs_sci -= 1. # Convert to np pixel coords
    y0_wcs_sci -= 1. # Convert to np pixel coords
    if plot_galfit:
        x0_galfit, y0_galfit = w.wcs_world2pix(galaxy.sersicRA, galaxy.sersicDEC, origin_wcs)
        x0_galfit -= 1. # Convert to np pixel coords
        y0_galfit -= 1. # Convert to np pixel coords
    
    # x is column, y is row: arrays are row, column
    cent_pix= _np.array([(_np.shape(galaxy.pstamp.pstamp)[1]-1)/2., 
            (_np.shape(galaxy.pstamp.pstamp)[0]-1)/2.])
    
    if rotate is True:
        # Need to rotate to counter slit offset
        if convolve:
            # Convolve HST/WFC3 resolution to Keck resolution:
            pstamp_trim = convolve_rot_gal(galaxy=galaxy, 
                                    instrument=instrument, 
                                    instrument_img=instrument_img, 
                                    conv_sigma=conv_sigma,
                                    verbose=verbose)
        else:
            pstamp_trim = rotate_pstamp(galaxy)
            
        try:
            # Here: care about slit_PA *ON IMAGE*
            # slit_PA_on_img = galaxy.spec2D.slit_PA - galaxy.pstamp.img_PA
            slit_PA = galaxy.spec2D.slit_PA - galaxy.pstamp.img_PA
        except:
            slit_PA = galaxy.spec1D.slit_PA - galaxy.pstamp.img_PA
            
        # Rotate x0,y0:
        x0_gal, y0_gal = rot_object_center(x0_galfit, y0_galfit, cent_pix, 
                                -1.*slit_PA*d2r)
        x0_sci, y0_sci = rot_object_center(x0_wcs_sci, y0_wcs_sci, 
                            cent_pix, -1.*slit_PA*d2r)
    else:
        x0_sci = x0_wcs_sci.copy()
        y0_sci = y0_wcs_sci.copy()
        
        x0_gal = x0_galfit.copy()
        y0_gal = y0_galfit.copy()
        pstamp_trim = galaxy.pstamp.pstamp
    
    # x0, y0: galfit RA, DEC
    # x0_sci, y0_sci: IMAGE RA, DEC
    
    
    if verbose:
        print("CHECK SIZE={:0.3f}".format(pstamp_trim.shape[0]*instrument_img.pixscale) )
    
    if downsample is True:        
        # Original image pix/arc arrays
        yarr_orig = _np.linspace(0,_np.shape(pstamp_trim)[0], 
                            num=_np.shape(pstamp_trim)[0], endpoint=False)
        xarr_orig = _np.linspace(0,_np.shape(pstamp_trim)[1], 
                            num=_np.shape(pstamp_trim)[1], endpoint=False)
                            
        # Center on x0_sci, y0_sci, but including offset from ypos_mos to m0:
        xarr_orig -= x0_gal
        xarc_orig = xarr_orig*instrument_img.pixscale
        yarr_orig -= y0_gal
        yarc_orig = yarr_orig*instrument_img.pixscale
        
        # Output pix/arc array:
        n_round = _np.int(_np.floor(_np.shape(pstamp_trim)[0]*instrument_img.pixscale/instrument.pixscale))
        
        ### New:
        yarr_new = _np.linspace(0, n_round, num=n_round, endpoint=False)
        # Center on m0:
        ycent = m0
        
        yarr_new -= ycent
        yarc_new = yarr_new*instrument.pixscale
        
        if verbose:
            #print "yarc_orig=", yarc_orig
            print("yarc_new={:0.3f}".format(yarc_new) )
        
        #!!!!!!!!!!!!!!!!!!!!!!!
        
        
        xarr_new = _np.linspace(0, n_round, num=n_round, endpoint=False)
        # Center on center:
        xcent = (n_round-1)/2.
        xarr_new -= xcent
        xarc_new = xarr_new*instrument.pixscale
        
        
        # Do box integration:
        pstamp_trim = box_integrate_pstamp(pstamp_trim, xarc_orig, yarc_orig,
                                xarc_new, yarc_new)
        
        ########
        ### Overall conversions:
        #     pix -> offset pix -> arcsec -> offset pix of other -> pix of other
        x0_gal_orig = x0_gal.copy()
        y0_gal_orig = y0_gal.copy()
        
        def x_trans(x):
            return (x-x0_gal_orig)*instrument_img.pixscale/instrument.pixscale + xcent
        def y_trans(y):
            return (y-y0_gal_orig)*instrument_img.pixscale/instrument.pixscale + ycent
        
        # Change x0, y0 positions: use same tranformation as downsampling.
        x0_gal = x_trans(x0_gal)
        y0_gal = y_trans(y0_gal)
        
            
        x0_sci = x_trans(x0_sci)
        y0_sci = y_trans(y0_sci)
        
        # Change cent_pix to new dimensions:
        # x is column, y is row: arrays are row, column
        cent_pix= _np.array([(_np.shape(pstamp_trim)[1]-1)/2., 
                (_np.shape(pstamp_trim)[0]-1)/2.])
        
        pscale = instrument.pixscale
        
    else:
        pscale = instrument_img.pixscale
    
    #################################################################    
    # Set a specific number of arcsec:
    trim_size_orig = _np.repeat(2./pscale, 2)
    
    if n_spatial_pixels is not None:
        if _np.average(trim_size_orig*pscale) != n_spatial_pixels*instrument.pixscale:
            trim_size = _np.repeat(n_spatial_pixels*instrument.pixscale/pscale, 2)
        else:
            trim_size = trim_size_orig.copy()
    else:
        trim_size = trim_size_orig.copy()
        
    #######################################################################
    # left,bottom pix edges = -0.5
    xlim = _np.array([x0_gal-(trim_size[1])/2., x0_gal+(trim_size[1])/2.])
    ylim = _np.array([y0_gal-(trim_size[0])/2., y0_gal+(trim_size[0])/2.])
    
    
    if m0 is not None:
        # Offset xlim, ylim by the amount between m0 and center of image!!!
        y_off_mid = (y0_gal-ylim[0]) - (m0+0.5)*instrument.pixscale/pscale
        ylim += y_off_mid
        if verbose:
            print("(y0_gal-ylim[0]) - (m0+0.5)={:0.3f}".format( (y0_gal-ylim[0]) - (m0+0.5) ) )
    #
    if verbose:
        print("ylim = {}".format(ylim) )
        
    #######################################
    # Scale based on noise levels:
    xlim_round = _np.array([_np.int(_np.round(xlim[0])), _np.int(_np.round(xlim[1]))])
    ylim_round = _np.array([_np.int(_np.round(ylim[0])), _np.int(_np.round(ylim[1]))])
    
    tmp_trim = pstamp_trim[ylim_round[0]:ylim_round[1],xlim_round[0]:xlim_round[1]].copy()
    tmp_trim = tmp_trim.flatten()
    range_spec = tmp_trim[_np.isfinite(tmp_trim)].copy()
    range_spec = _np.abs(range_spec)
    range_spec.sort()
    # Only consider the lowest 50\% *abs* points:
    range_spec = range_spec[0:_np.int(_np.ceil(len(range_spec)/2.))]
    median_err = _np.median(range_spec)
    
    ext = 3.
    vmin = _np.log10((-1.+ext)*median_err)
    vmax = _np.log10((100.+ext)*median_err)
    
    
    ax.imshow(_np.log10(pstamp_trim + median_err*(ext) ), cmap=cmap, 
            interpolation='Nearest', origin=origin, vmin=vmin, vmax=vmax)
            
    if external_ax is False:
        try:
            title = '\_{}'.join(galaxy.maskname.split('_'))+'.'+instrument_img.band+'.'+str(galaxy.ID)
        except:
            title = '\_{}'.join(galaxy.field.split('_'))+'.'+instrument_img.band+'.'+str(galaxy.ID)
        ax.set_title(title)
        
    
    ############################################################
    # Do calculations for slit:        
    slit_wid = instrument.slit_width
    try:
        slit_len = instrument.slit_length
    except:
        slit_len = _np.shape(pstamp_trim)[0]+20.
    
    if rotate is True:
        slit_angle = 0.
    else:
        # True x0, y0 of the object:
        try:
            # Here: care about slit_PA *ON IMAGE*
            # slit_PA_on_img = galaxy.spec2D.slit_PA - galaxy.pstamp.img_PA
            slit_angle = galaxy.spec2D.slit_PA - galaxy.pstamp.img_PA
        except:
            slit_angle = galaxy.spec1D.slit_PA - galaxy.pstamp.img_PA
    
    # Offset in slit coords -- pixels
    y0_off = galaxy.spec2D.yoffset/instrument.pixscale  
    
    ##############
    # Corner coords: slit rectangle
    off_angle = 4.-0.22   # exaggerate for test. really: 4.-0.22
    y_shear = slit_wid/pscale*_np.tan(off_angle*d2r)
    
    pos_11 = _np.array([x0_sci-0.5*slit_wid/pscale, y0_sci-y0_off-0.5*slit_len/pscale])
    pos_12 = _np.array([x0_sci-0.5*slit_wid/pscale, y0_sci-y0_off+0.5*slit_len/pscale])
    pos_21 = _np.array([x0_sci+0.5*slit_wid/pscale, y0_sci-y0_off-0.5*slit_len/pscale])
    pos_22 = _np.array([x0_sci+0.5*slit_wid/pscale, y0_sci-y0_off+0.5*slit_len/pscale])
    
    # Modify BL, TR corner y coords: shear
    pos_11[1] += y_shear
    pos_22[1] -= y_shear
    
    rect_sci_x, rect_sci_y = rot_corner_coords([pos_11,pos_21,pos_22,pos_12,pos_11], 
            slit_angle*d2r, x0=x0_sci, y0=y0_sci)
    
    ax.plot(rect_sci_x, rect_sci_y, lw=lw_slit, ls='-', c='green')
    
    ##############
    ## Overplot semi-major, minor axes:
    ## True x0, y0 of the object:
    #print galaxy.galfit_RA, galaxy.galfit_DEC
    if rotate is True:
        if plot_galfit:
            try:
                # Here: care about slit_PA *ON IMAGE*
                # slit_PA_on_img = galaxy.spec2D.slit_PA - galaxy.pstamp.img_PA
                galfit_angle = galaxy.sersicPA - ( galaxy.spec2D.slit_PA - galaxy.pstamp.img_PA)
            except:
                galfit_angle = galaxy.sersicPA - (galaxy.spec1D.slit_PA - galaxy.pstamp.img_PA)
            # eg galfit_angle = galaxy.delt_PA
            galfit_angle = _np.arctan(_np.tan(galfit_angle*_np.pi/180.))*180./_np.pi
    else:
        if plot_galfit:
            galfit_angle = galaxy.sersicPA
            galfit_angle = _np.arctan(_np.tan(galfit_angle*_np.pi/180.))*180./_np.pi
        
    if plot_galfit:
        axis_ratio = galaxy.q
        semi_major = galaxy.re_arcsec
        semi_minor = semi_major*axis_ratio
        
        # Output string
        s='b/a='+str(axis_ratio)
        
        if convolve:
            # Need to convolve semi_major axis to the same resolution, using sigma_conv
            # Conv_sigma is in arcseconds, as is semi_major
            # Semi-major is HWHM of total flux profile
            sigma_a = semi_major/(_np.sqrt(2.*_np.log(2.)))
            
            # print "FIX THIS!"
            # raise ValueError
            
            conv_sigma_galfit = instrument.PSF.PSF_FWHM/(2.*_np.sqrt(2.*_np.log(2.)))
            sigma_a_new = _np.sqrt(conv_sigma_galfit**2 + sigma_a**2)
            semi_major = sigma_a_new*(_np.sqrt(2.*_np.log(2.)))
            
            # Do the same for semi-minor axis -- will get more smearing of one than 
            #       the other, if the resolution element is ~= semi-minor axis
            sigma_b = semi_minor/(_np.sqrt(2.*_np.log(2.)))
            sigma_b_new = _np.sqrt(conv_sigma_galfit**2 + sigma_b**2)
            semi_minor = sigma_b_new*(_np.sqrt(2.*_np.log(2.)))
            
            # Update output string
            q_new = semi_minor/semi_major
            s='b/a$_{\mathrm{int}}$='+str(axis_ratio)+', b/a={:.4f}'.format(q_new)
        
        
        if show_galfit:
            ############
            # Information in sub-title:
            if external_ax is False:
                ax.annotate(s=s, xy=[.5, 0.05], xycoords='figure fraction', 
                            ha='center', va='bottom')
            if verbose:            
                print("% misfit.plot.plot_image_slit:")
                try:
                    print("{} {:d}".format(galaxy.maskname, galaxy.ID) )
                except:
                    print(galaxy.field, galaxy.ID)
                print("\t Using convolved R_E for ellipse")
                print("\t R_E int = {:0.3f}".format(galaxy.re_arcsec) )
                print("\t semi-major = R_E conv = {:0.3f}".format(semi_major) )
    
            # Semi-minor axis
            pos_11 = _np.array([x0_gal-semi_minor/pscale, y0_gal])
            pos_22 = _np.array([x0_gal+semi_minor/pscale, y0_gal])
            line_b_x, line_b_y = rot_corner_coords([pos_11, pos_22],
                        galfit_angle*d2r, x0=x0_gal, y0=y0_gal)
            ax.plot(line_b_x, line_b_y, lw=1, ls='-', c='red', zorder=10)
            
            # Semi-major axis
            pos_11 = _np.array([x0_gal, y0_gal-semi_major/pscale])
            pos_22 = _np.array([x0_gal, y0_gal+semi_major/pscale])
            # ax.plot([pos_11[0], pos_22[0]], [pos_11[1], pos_22[1]], ls='-', c='orange')
            line_a_x, line_a_y = rot_corner_coords([pos_11, pos_22],
                        galfit_angle*d2r, x0=x0_gal, y0=y0_gal)    
        
            ax.plot(line_a_x, line_a_y, lw=1, ls='-', c='cyan', zorder=10)

            # Plot ellipse:
            ell = Ellipse((x0_gal,y0_gal), 2.*semi_minor/pscale, 
                            2.*semi_major/pscale, galfit_angle,
                            ec='blue', fc='none', fill=False, zorder=11)
             
            ####################################
            # Plot 1 arcsec line:
            x_base = xlim[0] + (xlim[1]-xlim[0])*.1
            y_base = ylim[0] + (ylim[1]-ylim[0])*.1
            y_text = ylim[0] + (ylim[1]-ylim[0])*.13
            if cmap_name == 'gray_r':
                bar_color = 'k'
                lw=2
            else:
                bar_color = 'w'
                lw=2
            
            try:
                len_line = 1.*(galaxy.re_arcsec/galaxy.re_kpc)*(1./pscale)
                string = '1 kpc' # 1 kpc long
            except:
                len_line = 0.25/pscale
                string = '0.25 arcsec' # 1 arcsec long:
            
            if shift_label is not None:
                y_base += shift_label
                y_text += shift_label
    
            ax.plot([x_base, x_base+len_line], [y_base, y_base], c=bar_color, ls='-',lw=lw)
            ax.annotate(string, xy=(x_base+len_line/2., y_text), 
                            xycoords="data", xytext=(0,0), textcoords="offset points", 
                            ha="center", va="bottom",fontsize=8.)
                    
            ell.set_clip_box(ax.bbox)
            ax.add_artist(ell)
           
    if verbose:    
        print("CHECK RANGES:")
        print((ylim[1]-ylim[0])*pscale)
        print((xlim[1]-xlim[0])*pscale)
        if n_spatial_pixels is not None:
            print(n_spatial_pixels*instrument.pixscale)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if external_ax is False:
        if saveToFile:
            plt.savefig(fileout, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            plt.show()
    


def box_integrate_pstamp(pstamp_trim_orig, xarc_orig, yarc_orig,
                    xarc_new, yarc_new):
    # Do box integration
    
    #   first, for all columns, collapse over rows
    pstamp_arr_tmp = _np.zeros((len(yarc_new), len(xarc_orig)))
    for i in _six.moves.xrange(len(xarc_orig)):
        pslice = pstamp_trim_orig[:,i]
        
        # Now it's an array you can normal box integrate over:
        pslice_down = box_integrate(yarc_orig, pslice, yarc_new)
        pstamp_arr_tmp[:,i] = pslice_down
        
    
    #   then for all rows, collapse over columns
    pstamp_trim = _np.zeros((len(yarc_new), len(xarc_new)))
    for j in _six.moves.xrange(len(yarc_new)):
        pslice = pstamp_arr_tmp[j,:]
        pslice_down = box_integrate(xarc_orig, pslice, xarc_new)
        pstamp_trim[j,:] = pslice_down
        
    return pstamp_trim
    
    
    
# Define box integration for downsampling
def box_integrate(x, y, x_new, debug=False):
    # x: array of positions of old arr
    # y: array of values corresponding to old arr
    # x_new: array of positions for new arr
    
    delt_x_new = _np.average(x_new[1:]-x_new[:-1])
    delt_x = _np.average(x[1:]-x[:-1])
    
    x_low = x - delt_x/2.
    x_high = x + delt_x/2.
    
    y_new = _np.zeros(len(x_new))
    
    for i in _six.moves.xrange(len(x_new)):
        # For each element in the new array:
        # Bounds of that interval
        low = x_new[i] - delt_x_new/2.
        high = x_new[i] + delt_x_new/2.
        
        # Initialize integrated value:
        int_y = 0.
        
        try:
            # Pixels wholly contained within the new range:
            c_l = _np.where(x_low >= low)[0]
            c_r = _np.where(x_high <= high)[0]
            ind_c = _np.array(list(set(c_l) & set(c_r)))
            ind_c.sort()
            
            int_y += _np.sum(y[ind_c])*delt_x
        except:
            ind_c = 'None'
            
            
        # For outside edges: wrap in try..except
        #   to cover case of x_new extending further than
        #   original x -- the missing data will implicitly be zero.
        try:
            # Find element of x_low just lower than low
            ind_l = _np.where(x_low <= low)[0].max()
            # range corresponding to this interval:
            # (x_high[ind_l] - low)
            
            int_y += y[ind_l]*(x_high[ind_l]-low)
        except:
            ind_l = 'None'
            
        try:
            # find element of x_high just higher than high
            ind_r = _np.where(x_high >= high)[0].min()
            # range corresponding to this interval:
            # (high - x_low[ind_r])
            
            int_y += y[ind_r]*(high-x_low[ind_r])
        except:
            ind_r = 'None'
            
        if debug:
            print("indicies = {:d} {:d} {:d}".format(ind_l, ind_c, ind_r) )
            
            
        y_new[i] = int_y/delt_x_new
        
        
    return y_new