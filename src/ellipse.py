import numpy as np 

class Ellipse(object):

    @classmethod
    def normPolar(cls, theta, a, b):
        return a*b/np.sqrt((b*np.cos(theta))**2 + (a*np.sin(theta))**2)

    @classmethod
    def coordinateX(cls, theta, a, b):
        return cls.normPolar(theta, a, b)*np.cos(theta)

    @classmethod
    def coordinateY(cls, theta, a, b):
        return cls.normPolar(theta, a, b)*np.sin(theta)
