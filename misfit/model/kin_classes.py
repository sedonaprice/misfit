# Copyright 2017 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE

from __future__ import print_function

import numpy as _np
from astropy.extern import six as _six

import kin_functions as _kfuncs


class VelProfile(object):
    """
    Basic class for velocity profiles. Setting up expected structure.
    theta:          parameter array
    """
    def __init__(self,**kwargs):
        self.theta = None
        self.name = None
        self.setAttr(**kwargs)
        
    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))
    
    def vel(self, r, z):
        return 0.*r
        
    
class DispProfile(object):
    """
    Basic class for velocity dispersion profiles. Setting up expected structure.
    theta:          parameter array
    """
    def __init__(self,**kwargs):
        self.theta = None
        self.name = None
        self.setAttr(**kwargs)
        
    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))
    
    def sigma(self, r, z):
        return 0.*r
    
    
class VelArctanProfile(VelProfile):
    """
    Arctan velocity profile. Parameters are:
        theta = [V_a, r_t]
                V_a:    asymptotic velocity [km/s]
                r_t:    turnover radius     [arcsec]
    """
    def __init__(self,**kwargs):
        super(VelArctanProfile, self).__init__(**kwargs)
        
        # Theta:
        self.theta = _np.array([0.,0.])
        self.name = 'VelArctanProfile'
        
    def vel(self, r, z):
        try:
            return 2./_np.pi * self.theta[0] * _np.arctan( r /self.theta[1] )
        except:
            return _np.inf


class DispConstProfile(DispProfile):
    """
    Constant velocity dispersion profile. Parameters are:
        theta = [sigma0]
                sigma0:    velocity dispersion [km/s]
    """
    def __init__(self,**kwargs):
        super(DispConstProfile, self).__init__(**kwargs)
        
        # Theta:
        self.theta = _np.array([0.])
        self.name = 'DispConstProfile'
        
    def sigma(self, r, z):
        return 0.*r + self.theta[0] 




class KinProfile(object):
    def __init__(self, theta=None, name=None, theta_names=None, theta_names_nice=None, 
                        velProfile=None, dispProfile=None, **kwargs):
        
        self.theta = theta
        self.name = name
        
        self.theta_names = theta_names
        self.theta_names_nice = theta_names_nice
        
        self.velProfile = velProfile
        self.dispProfile = dispProfile
        
        self.setAttr(**kwargs)
        self.setup_profiles()
        
    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))
    
    def setup_profiles(self):
        if self.velProfile is not None:
            self.velProfile.n_params = len(self.velProfile.theta)
        if self.dispProfile is not None:
            self.dispProfile.n_params = len(self.dispProfile.theta)
            
        #self.update_theta(self.theta)
            
    def update_theta(self, theta):
        """
        Update theta for the velocity profile: 
            including splitting up vel, dispersion parameters
        """
        self.theta = theta
        
        # Distribute params based on input theta:
        if self.velProfile is not None:
            self.velProfile.theta = self.theta[:self.velProfile.n_params]
        
        if self.dispProfile is not None:
            self.dispProfile.theta = self.theta[self.velProfile.n_params:self.velProfile.n_params+self.dispProfile.n_params]
        
    def vel(self, r, z):
        # Units on r: arcsec
        return self.velProfile.vel(r, z)
                                
    def sigma(self, r, z):
        return self.dispProfile.sigma(r, z)
        
        
                                
class KinProfileFiducial(KinProfile):
    def __init__(self, **kwargs):
        super(KinProfileFiducial, self).__init__(**kwargs)
        
        self.name = 'VelArctanDispConstProfile'
        
        self.velProfile = VelArctanProfile()
        self.dispProfile = DispConstProfile()
        
        self.setAttr(**kwargs)
        self.setup_profiles()
        
        
######################################
class IntensityProfile(object):
    def __init__(self, galaxy, intProfile_r=None, intProfile_z=None, **kwargs):
        self.intProfile_r = intProfile_r
        self.intProfile_z = intProfile_z
        self.setAttr(**kwargs)
        
    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))
        
    def update_values(self, galaxy):
        self.intProfile_r.update_values(galaxy)
        self.intProfile_z.update_values(galaxy)
        
    def int(self, r_perp, z):
        return self.intProfile_r.int(r_perp) * self.intProfile_z.int(z) 
        
        
        
class IntensityProfileFiducial(IntensityProfile):
    def __init__(self, galaxy, **kwargs):
        super(IntensityProfileFiducial, self).__init__(galaxy, **kwargs)
        
        self.intProfile_r = IntProfileSersic(galaxy)
        self.intProfile_z = IntProfileExpZ(galaxy)
        
        self.setAttr(**kwargs)
        self.update_values(galaxy)
        
#

class IntensityProfileSersicExpScale(IntensityProfile):
    def __init__(self, galaxy, **kwargs):
        super(IntensityProfileSersicExpScale, self).__init__(galaxy, **kwargs)

        self.intProfile_r = IntProfileSersic(galaxy)
        self.intProfile_z = IntProfileExpZScale(galaxy)

        self.setAttr(**kwargs)
        self.update_values(galaxy)
        
        
class IntProfileCompBase(object):
    def __init__(self, galaxy, n=None, re_arcsec=None, **kwargs):
        self.n = n
        #self.q = None
        #self.delt_PA = None
        self.re_arcsec = re_arcsec
        
        self.setAttr(**kwargs)
        self.get_gal_params(galaxy)
        
    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))
    #
    def get_gal_params(self, galaxy):
        self.n = galaxy.n
        #self.q = galaxy.q
        #self.delt_PA = galaxy.delt_PA
        self.re_arcsec = galaxy.re_arcsec
        
        self.q0 = galaxy.q0  # Assumed intrinsic galaxy thickness
        
    def update_values(self, galaxy):
        self.get_gal_params(galaxy)
    
    
class IntProfileSersic(IntProfileCompBase):
    def __init__(self, galaxy, **kwargs):
        super(IntProfileSersic, self).__init__(galaxy, **kwargs)
        
        self.Ie = 1.  # Placeholder -- will be scaled.
        
        self.setAttr(**kwargs)
        self.get_gal_params(galaxy)
        
    def int(self, r):
        # Sersic surface intensity profile
        b_n = 2.*self.n - 0.324      # Ciotti+Bertin99
        return self.Ie*_np.exp( -b_n * (_np.power( (r/self.re_arcsec), 1./_np.float(self.n) ) -1. ) )

class IntProfileExpZ(IntProfileCompBase):
    def __init__(self, galaxy, **kwargs):
        super(IntProfileExpZ, self).__init__(galaxy, **kwargs)

        self.setAttr(**kwargs)
        self.update_values(galaxy)
        
    def int(self, z):
        return _np.exp(-_np.abs(z)/self.z0)
        
    def update_values(self, galaxy):
        self.get_gal_params(galaxy)
        self.z0 = self.q0*self.re_arcsec

#
class IntProfileExpZScale(IntProfileCompBase):
    def __init__(self, galaxy, **kwargs):
        super(IntProfileExpZScale, self).__init__(galaxy, **kwargs)

        self.setAttr(**kwargs)
        self.update_values(galaxy)
        
    def int(self, z):
        return _np.exp(-_np.abs(z)/self.z0)
        
    def update_values(self, galaxy):
        self.get_gal_params(galaxy)
        self.z0 = self.q0*(self.re_arcsec/1.676)
    
######################################


class ThetaPriorFlat(object):
    def __init__(self,theta=None, theta_vary=None, theta_bounds=None, **kwargs):
        
        self.name = 'ThetaPriorFlat'
        
        self.theta = theta
        self.theta_vary = theta_vary
        self.theta_bounds = theta_bounds
        
        
    def log_prior(self, theta_fitting=None):
        i_free = 0
        
        statements = []
        for i in _six.moves.xrange(len(self.theta)):
            if self.theta_vary[i]:
                statements.append(_kfuncs._between(self.theta_bounds[i], theta_fitting[i_free]))
                i_free += 1
            
        if _np.array(statements).all():
            prior = 0.
            
            return prior
        else:
            return -_np.inf
            

class Theta2DSettings(object):
    """
    Class to initialize the settings needed for the fitting parameters, 
        which is passed to the fitEmis2D object.
    """
    def __init__(self, theta=None, theta_vary=None, 
                    theta_names=None, theta_names_nice=None, 
                    theta_bounds=None, theta_linked_posteriors=None, **kwargs):
        self.theta = theta
        self.theta_vary = theta_vary
        self.theta_names = theta_names 
        self.theta_names_nice = theta_names_nice
        
        self.theta_bounds = theta_bounds
        
        self.theta_linked_posteriors = theta_linked_posteriors
        
        self.setAttr(**kwargs)

    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(dict([(key, kwargs.get(key)) for key in kwargs if key in self.__dict__]))

    


class Theta2DSettingsFiducial(Theta2DSettings):
    """
    Class to initialize the settings needed for the fitting parameters, 
        which is passed to the fitEmis2D object.
        Contains defaults for fiducial (arctan + const velocity dispersion) kinModel
        
    Designed to have the Velocity profile parameters first, then Dispersion profile params, then 
        composite / other parameters (eg, spatial + wavelength shift)
    """
    def __init__(self, **kwargs):
        super(Theta2DSettingsFiducial, self).__init__(**kwargs)
        
        self.theta = _np.zeros(5)
        self.theta_vary = _np.array([True, True, True, False, False])
        self.theta_names = ['V_a', 'r_t', 'sigma0', 'm0_shift', 'z']
        self.theta_names_nice = [r'$V_a$', r'$r_t$', r'$\sigma_0$', 
                            r'$m_{0,shift}$', r'$z$']
        
        self.setAttr(**kwargs)

        