import numpy as np

class Scatterer(object):

    def __init__(self, potential_type='coulomb', coulomb_depth=10.0, defect_radius=5.0, defect_amplitude=1.0):
        self.type = potential_type
        self.d = coulomb_depth
        self.sigma = defect_radius
        self.v_0 = defect_amplitude
    
    def potentialFourier(self, q):
        if self.type == 'coulomb':
            return np.divide(2*np.pi*np.exp(-q*self.d), q, out=np.zeros_like(q), where=q!=0)
        elif self.type == 'defect':
            return 2.0*np.pi*self.sigma**2*self.v_0*np.exp(-2.0*np.pi**2*self.sigma**2*q**2)
        else: 
            return np.zeros_like(q)
