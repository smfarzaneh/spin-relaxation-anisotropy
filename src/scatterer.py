import numpy as np

# Note: All variables are in atomic units (AU). 
class Scatterer(object):

    def __init__(self, screening_length=0.0, depth=10.0, type='yukawa'):
        self.l = screening_length
        self.d = depth

    def potential(self, r):
        return 2.0/r*np.exp(-r/self.l)
    
    def potentialFourier(self, q):
        return np.divide(2*np.pi*np.exp(-q*self.d), q, out=np.zeros_like(q), where=q!=0)
