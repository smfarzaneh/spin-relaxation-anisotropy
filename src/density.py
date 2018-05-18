import numpy as np 

from pauli import *

class Density(object):

    def __init__(self, spin_x, spin_y, spin_z):
        self.spin = np.zeros((2, 2), dtype=np.complex)
        self.matrix = np.zeros((2, 2), dtype=np.complex)
        self.update(spin_x, spin_y, spin_z)

    def update(self, spin_x, spin_y, spin_z):
        self.spin = spin_x*Pauli.x() + spin_y*Pauli.y() + spin_z*Pauli.z()
        self.matrix = 0.5*Pauli.i() + self.spin
