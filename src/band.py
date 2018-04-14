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
        rashba_strength,
        energy_level):
        self.eg = band_gap
        self.mxc = effective_mass_x_c
        self.mxv = effective_mass_x_v  
        self.myc = effective_mass_y_c
        self.myv = effective_mass_y_v  
        self.lambdar = rashba_strength 
        self.energy = energy_level
        self.band = 1
        self.a = 1.0
        self.b = 1.0
        self.updateEnergyDependentParameters()

    def updateEnergyDependentParameters(self):
        # conduction or valance band
        if abs(self.energy) <= self.eg/2.0:
            raise ValueError('invalid energy level inside the band gap.')
        elif self.energy > self.eg/2.0:
            band = 1
        else:
            band = -1
        # major and minor axes of elliptic Fermi contour
        mx, my = self.getEffectiveMass()
        a = np.sqrt(mx*(abs(self.energy) - self.eg/2.0))
        b = np.sqrt(my*(abs(self.energy) - self.eg/2.0))
        # update
        self.band = band
        self.a = a
        self.b = b

    def energyMomentum(self, arg_1, arg_2, band, coordinate='polar'): 
        if coordinate == 'polar':
            kx, ky = self._polarToCartesian(arg_1, arg_2)
        else:
            kx = arg_1
            ky = arg_2
        return self._energyCartesian(kx, ky)

    def _energyCartesian(self, kx, ky):
        if self.band == 1:
            return self.eg/2.0 + (np.power(kx, 2)/self.mxc + np.power(ky, 2)/self.myc)
        else:
            return -self.eg/2.0 - (np.power(kx, 2)/self.mxv - np.power(ky, 2)/self.myv)
        
    def _polarToCartesian(self, k, theta):
        return k*np.cos(theta), k*np.sin(theta)

    def momentum(self, theta):
        kx = Ellipse.coordinateX(theta, self.a, self.b)
        ky = Ellipse.coordinateX(theta, self.a, self.b)
        return kx, ky

    def fieldRashba(self, theta):
        if isinstance(theta, (list, tuple, np.ndarray)):
            omega = np.zeros((3, len(theta)))
        else: 
            omega = np.zeros((3, 1))
        k = Ellipse.normPolar(theta, self.a, self.b)
        mx, my = self.getEffectiveMass()
        omega[0, :] = self.lambdar/my*k*np.sin(theta)
        omega[1, :] = -self.lambdar/mx*k*np.cos(theta)
        return omega

    def gradientEnergy(self, theta):
        term_1 = Ellipse.normPolar(theta, self.a, self.b)
        term_2 = 2.0*abs(self.energy) - self.eg
        term_3 = np.sqrt(np.cos(theta)**2/self.a**4 + np.sin(theta)**2/self.b**4)
        gradient = term_1*term_2*term_3
        return gradient

    def getEffectiveMass(self):
        if self.band == 1:
            mx = self.mxc
            my = self.myc
        else:
            mx = self.mxv
            my = self.myv
        return mx, my
