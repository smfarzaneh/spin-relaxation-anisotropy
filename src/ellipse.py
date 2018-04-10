import numpy as np 

class Ellipse():

    @classmethod
    def normPolar(cls, theta, a, b):
        aux = cls._aux(theta, a, b)
        return a*b/np.sqrt(aux)

    @classmethod
    def coordinateX(cls, theta, a, b):
        return cls.normPolar(theta, a, b)*np.cos(theta)

    @classmethod
    def coordinateY(cls, theta, a, b):
        return cls.normPolar(theta, a, b)*np.sin(theta)

    @classmethod
    def normDerivative(cls, theta, a, b):
        aux = cls._aux(theta, a, b)
        return -a*b*(a**2 - b**2)*np.sin(theta)*np.cos(theta)/np.sqrt(aux**3)

    @classmethod
    def normTangent(cls, theta, a, b):
        k = cls.normPolar(theta, a, b)
        kp = cls.normDerivative(theta, a, b)
        return np.sqrt(k**2 + kp**2)

    @classmethod
    def perimeter(cls, a, b):
        h = (a - b)**2/(a + b)**2
        p = np.pi*(a + b)*(1.0 + 3.0*h/(10.0 + np.sqrt(4.0 - 3.0*h)))
        return p

    @classmethod
    def _aux(cls, theta, a, b):
        return  (a*np.sin(theta))**2 + (b*np.cos(theta))**2
