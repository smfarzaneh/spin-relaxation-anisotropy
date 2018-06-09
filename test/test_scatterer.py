import unittest
from ..src.scatterer import *
import numpy as np

class TestScatterer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.coulomb = Scatterer(potential_type='coulomb')

    def testCoulomb(self):
        self.assertAlmostEqual(self.coulomb.potentialFourier(0.0), 0.0)
        self.assertNotAlmostEqual(self.coulomb.potentialFourier(0.01), 0.0)
        
if __name__ == '__main__':
    unittest.main()