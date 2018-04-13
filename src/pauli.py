import numpy as np

class Pauli():

    @staticmethod
    def x():
        x = np.zeros((2, 2), dtype=np.complex)
        x[0, 1] = 1.0
        x[1, 0] = 1.0
        return x

    @staticmethod
    def y():
        y = np.zeros((2, 2), dtype=np.complex)
        y[0, 1] = -1.0j
        y[1, 0] = 1.0j
        return y

    @staticmethod
    def z():
        z = np.zeros((2, 2), dtype=np.complex)
        z[0, 0] = 1.0
        z[1, 1] = -1.0
        return z

    @staticmethod
    def i():
        i = np.zeros((2, 2), dtype=np.complex)
        i[0, 0] = 1.0
        i[1, 1] = 1.0
        return i
