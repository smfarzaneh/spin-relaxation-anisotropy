import numpy as np

class Pauli(object):

    def __init__(self):

        pass

    def x(self):

        x = np.zeros((2, 2), dtype=np.complex)
        x[0, 1] = 1.0
        x[1, 0] = 1.0
        
        return x
    
    def y(self):

        y = np.zeros((2, 2), dtype=np.complex)
        y[0, 1] = -1.0j
        y[1, 0] = 1.0j

        return y

    def z(self):

        z = np.zeros((2, 2), dtype=np.complex)
        z[0, 0] = 1.0
        z[1, 1] = -1.0

        return z

    def i(self):

        i = np.zeros((2, 2), dtype=np.complex)
        i[0, 0] = 1.0
        i[1, 1] = 1.0

        return i
