import unittest
from ..src.ellipse import *
import numpy as np

class TestEllipse(unittest.TestCase):
    
    def testCrossing(self):
        self.assertAlmostEqual(Ellipse.normPolar(np.pi/4.0, 1.0, 1.0), 1.0)
        self.assertAlmostEqual(Ellipse.normPolar(0.0, 1.0, 2.0), 1.0)
        self.assertAlmostEqual(Ellipse.normPolar(np.pi/2.0, 1.0, 2.0), 2.0)
        self.assertAlmostEqual(Ellipse.normPolar(np.pi, 1.0, 2.0), 1.0)
        self.assertAlmostEqual(Ellipse.normPolar(3.0*np.pi/2.0, 1.0, 2.0), 2.0)
        self.assertAlmostEqual(Ellipse.coordinateX(np.pi, 1.0, 2.0), -1.0)
        self.assertAlmostEqual(Ellipse.coordinateY(3.0*np.pi/2.0, 1.0, 2.0), -2.0)

if __name__ == '__main__':
    unittest.main()