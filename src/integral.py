import numpy as np 

class Integral():

    @staticmethod
    def trapz(x_vals, f_vals):
        num = len(x_vals)
        if num == 1:
            raise ValueError('invalid input array')
        h = (x_vals[num - 1] - x_vals[0])/(num - 1.0) 
        i = h*(np.sum(f_vals) - 0.5*f_vals[0] - 0.5*f_vals[num - 1])
        return i

    @staticmethod
    def simpson(x_vals, f_vals):
        num = len(x_vals)
        if num == 1:
            raise ValueError('invalid input array')
        elif num%2 == 0:
            print('simpson integration warning, number of grid points should be odd.')
        h = (x_vals[num - 1] - x_vals[0])/(num - 1.0) 
        f_vals_short = f_vals[1:num-1]
        odd_vals = f_vals_short[0::2]
        even_vals = f_vals_short[1::2]
        i = h/3.0*(f_vals[0] + f_vals[num - 1] + 4.0*np.sum(odd_vals) + 2.0*np.sum(even_vals))
        return i
