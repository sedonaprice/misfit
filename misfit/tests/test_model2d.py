# misfit/tests/test_model2d.py
# Test functionality, speed of 2D MISFIT models
# 
# Copyright 2019 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

#from unittest import TestCase

import misfit
import numpy as np

import timeit
import time

# class TestModel2D(object):
#     # def test_model_creation_time(self):
#     #     print(timeit.timeit("self.test_model_creation()"))#setup="from __main__ import test"))
#     #     #timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
#     #     
#         
#     def test_model_creation(self):
#         """
#         Test time it takes to create single MISFIT 2D model
#         
#         """
#         dlam = 2.1691  # for K band MOSFIRE
#         nwave = 25
#         lam0 = 6564.60
#         z = 2.
#         
#         Va = 150.
#         rt = 0.5
#         sigma0 = 50.
#         n = 1.
#         re_arcsec = 1.
#         q = 0.6
#         q0 = 0.19
#         delt_PA = 35.
#         
#         wave =  np.linspace(lam0*(1.+z) -dlam*(nwave-1)/2., 
#                         lam0*(1.+z) +dlam*(nwave-1)/2.,num=nwave ) 
# 
#         # Use new function:
#         model_img = misfit.mock.generate_mock_slit_obs(z=z, 
#                 ID='misfit_test', 
#                 Va = Va, 
#                 rt = rt, 
#                 sigma0 = sigma0, 
#                 yshift=0, 
#                 n=n, 
#                 re_arcsec=re_arcsec, 
#                 q = q, 
#                 q0 = q0, 
#                 delt_PA = delt_PA, 
#                 dither = False, 
#                 wave=wave,
#                 instrument_name='MOSFIRE', 
#                 pixscale=0.1799,
#                 instrument_resolution = 35.,
#                 slit_width=0.7,
#                 slit_length=np.ceil(4.*re_arcsec),
#                 nymin = 13, 
#                 PSF_FWHM=0.6, 
#                 PSF_type="Gaussian", 
#                 band = 'K', 
#                 line_primary='HA6565',linegroup_name='Halpha',
#                 nSubpixels=2,
#                 pad_factor=0.5, 
#                 do_position_wave_shift=False,
#                 do_inst_res_conv_effective=False)
                
def test_2d_model_creation_time():
    N = 10
    repeat = 3
    rept = timeit.repeat("test_2d_model_creation()", 
            setup="from misfit.tests.test_model2d import test_2d_model_creation", 
            repeat=repeat, number=N)
    best = min(rept)
    usec = best * 1e6 / N
    if usec < 1e3:
        besttime = "{:0.1f} us".format(usec)
    else:
        msec = usec / 1e3
        if msec < 1e3:
            besttime = "{:0.1f} ms".format(msec)
        else:
            sec = msec / 1e3
            besttime = "{:0.1f} s".format(sec)
    print("------------------------------------------------")
    print("misfit.tests.test_2d_model_creation_time():")
    print("   {} loops, best of {}: {} per loop".format(N, repeat, besttime))
    print("------------------------------------------------")
    
def test_2d_model_creation():
    """
    Test time it takes to create single MISFIT 2D model
    
    """
    dlam = 2.1691  # for K band MOSFIRE
    nwave = 25
    lam0 = 6564.60
    z = 2.
    
    Va = 150.
    rt = 0.5
    sigma0 = 50.
    n = 1.
    re_arcsec = 1.
    q = 0.6
    q0 = 0.19
    delt_PA = 35.
    
    wave =  np.linspace(lam0*(1.+z) -dlam*(nwave-1)/2., 
                    lam0*(1.+z) +dlam*(nwave-1)/2.,num=nwave ) 

    # Use new function:
    model_img = misfit.mock.generate_mock_slit_obs(z=z, 
            ID='misfit_test', 
            Va = Va, 
            rt = rt, 
            sigma0 = sigma0, 
            yshift=0, 
            n=n, 
            re_arcsec=re_arcsec, 
            q = q, 
            q0 = q0, 
            delt_PA = delt_PA, 
            dither = False, 
            wave=wave,
            instrument_name='MOSFIRE', 
            pixscale=0.1799,
            instrument_resolution = 35.,
            slit_width=0.7,
            slit_length=np.ceil(4.*re_arcsec),
            nymin = 13, 
            PSF_FWHM=0.6, 
            PSF_type="Gaussian", 
            band = 'K', 
            line_primary='HA6565',linegroup_name='Halpha',
            nSubpixels=2,
            pad_factor=0.5, 
            do_position_wave_shift=False,
            do_inst_res_conv_effective=False)
                
#
if __name__ == '__main__':
    # # #test_model_creation()
    # # import timeit
    # # import time
    # N = 10
    # repeat = 3
    # # tim = timeit.timeit("test_model_creation()", 
    # #         setup="from __main__ import test_model_creation", 
    # #         number=N)
    # # print(tim/(1.*N))
    # rept = timeit.repeat("test_2d_model_creation()", 
    #         setup="from __main__ import test_2d_model_creation", 
    #         repeat=repeat, number=N)
    # best = min(rept)
    # usec = best * 1e6 / N
    # if usec < 1e3:
    #     besttime = "{:0.1f} us".format(usec)
    # else:
    #     msec = usec / 1e3
    #     if msec < 1e3:
    #         besttime = "{:0.1f} ms".format(msec)
    #     else:
    #         sec = msec / 1e3
    #         besttime = "{:0.1f} s".format(sec)
    # print("------------------------------------------------")
    # print("misfit.tests.test_2d_model_creation_time():")
    # print("   {} loops, best of {}: {} per loop".format(N, repeat, besttime))
    # print("------------------------------------------------")
    
    test_2d_model_creation_time()