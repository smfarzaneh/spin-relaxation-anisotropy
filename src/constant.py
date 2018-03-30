import numpy as np
import scipy.constants as sc

class Constant(object):
    eps0 = sc.epsilon_0 # permittivity of free space
    hbar = sc.hbar # reduced Planck constant
    e = sc.e # unit electric charge
    me = sc.electron_mass # mass of electron
    a0 = 4.0*np.pi*eps0*hbar**2/(me*e**2) # Bohr radius 
    Eh = me*e**4/(4.0*np.pi*eps0*hbar)**2 # Hartree energy
