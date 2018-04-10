import unittest
from ..src.integral import *
import numpy as np

class TestIntegral(unittest.TestCase):
    
    def testSine(self):
        a = 0.0
        b = np.pi
        num = 101
        x_vals = np.linspace(a, b, num)
        f_vals = np.sin(x_vals)
        self.assertAlmostEqual(Integral.trapz(x_vals, f_vals), 2.0, places=3)
        
if __name__ == '__main__':
    unittest.main()