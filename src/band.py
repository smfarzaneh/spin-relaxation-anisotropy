from ellipse import *
import numpy as np

# A general anisotropic band structure 
# described by elliptic equations at low energy 
# Note: All variables are in atomic units (AU). 
class Band(object):

    def __init__(self, 
        band_gap, 
        effective_mass_x_c, 
        effective_mass_x_v, 
        effective_mass_y_c, 
        effective_mass_y_v, 
        rashba_strength):
        self.eg = band_gap
        self.mxc = effective_mass_x_c
        self.mxv = effective_mass_x_v  
        self.myc = effective_mass_y_c
        self.myv = effective_mass_y_v  
        self.lambdar = rashba_strength 

    def energy(self, arg_1, arg_2, band, coordinate='polar'): 
        if coordinate == 'polar':
            kx, ky = self._polarToCartesian(arg_1, arg_2)
        else:
            kx = arg_1
            ky = arg_2
        return self._energyCartesian(kx, ky, band)

    def _energyCartesian(self, kx, ky, band='c'):
        if band == 'c':
            return self.eg/2.0 + (np.power(kx, 2)/self.mxc + np.power(ky, 2)/self.myc)
        else:
            return -self.eg/2.0 - (np.power(kx, 2)/self.mxv - np.power(ky, 2)/self.myv)
        
    def _polarToCartesian(self, k, theta):
        return k*np.cos(theta), k*np.sin(theta)

    def momentum(self, energy_level, theta):
        a, b, band = self._axes(energy_level)
        kx = Ellipse.coordinateX(theta, a, b)
        ky = Ellipse.coordinateX(theta, a, b)
        return kx, ky

    def fieldRashba(self, energy_level, theta):
        a, b, band = self._axes(energy_level)
        omega = np.zeros(3)
        k = Ellipse.normPolar(theta, a, b)
        if band == 1:
            mx = self.mxc
            my = self.myc
        else:
            mx = self.mxv
            my = self.myv
        omega[0] = self.lambdar/my*k*np.sin(theta)
        omega[1] = -self.lambdar/mx*k*np.cos(theta)
        return omega

    def _axes(self, energy_level):
        if abs(energy_level) < self.eg/2.0:
            raise ValueError('Invalid energy level inside the band gap.')
        elif energy_level >= self.eg/2.0:
            mx = self.mxc
            my = self.myc
            band = 1
        else:
            mx = self.mxv
            my = self.myv
            band = -1
        a = mx*abs(self.eg/2.0 - energy_level)
        b = my*abs(self.eg/2.0 - energy_level)
        return a, b, band
