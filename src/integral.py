import numpy as np 

class Integral():

    @classmethod
    def trapz(cls, x_vals, f_vals):
        num = len(x_vals)
        if num == 1:
            raise ValueError('invalid input array')
        h = (x_vals[num - 1] - x_vals[0])/(num - 1.0) 
        i = h*(np.sum(f_vals) - 0.5*f_vals[0] - 0.5*f_vals[num - 1])
        return i
