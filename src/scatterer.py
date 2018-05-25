import numpy as np

# Note: All variables are in atomic units (AU). 
class Scatterer(object):

    def __init__(self, screening_length=0.0, depth=10.0, potential_type='coulomb'):
        self.l = screening_length
        self.d = depth
        self.type = potential_type

    def potential(self, r):
        return 2.0/r*np.exp(-r/self.l)
    
    def potentialFourier(self, q):
        if self.type == 'coulomb':
            return np.divide(2*np.pi*np.exp(-q*self.d), q, out=np.zeros_like(q), where=q!=0)
        elif self.type == 'defect':
            return np.ones_like(q)
        else: 
            return np.zeros_like(q)
