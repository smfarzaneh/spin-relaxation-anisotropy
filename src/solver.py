import numpy as np

from ellipse import *
from integral import *
from pauli import *

class Solver(object):

    def __init__(self, band_structure, scatterer, density_avg, num_grid=200, num_iteration=400):
        self.band = band_structure
        self.scatterer = scatterer
        self.density_avg = density_avg
        self.num_grid = num_grid
        self.num_iteration = num_iteration
        self.reset()

    def reset(self):
        self.d_theta = 2.0*np.pi/self.num_grid
        self.theta_vals = np.linspace(0.0 + self.d_theta/2.0, 2.0*np.pi - self.d_theta/2.0, self.num_grid)
        self.density_vals = np.zeros((self.density_avg.matrix.size, self.num_grid), dtype=np.complex)
    
    def iterate(self):
        for _ in range(self.num_iteration):
            self.density_vals = self._densityNext() 
    
    def _densityNext(self):
        density_next = np.zeros_like(self.density_vals)
        collision = np.zeros_like(self.density_avg.matrix)
        for i in range(self.num_grid):
            precession = self._precession(i, self.density_avg.matrix)
            collision = self._averageCollision(i)
            rate_tot = self.sumTransitionRate(i)
            rho_p = (precession + collision)/rate_tot
            density_next[:, i] = np.reshape(rho_p, (4, ))
        return density_next

    def spinRelaxationSingle(self):
        density_time_derivative = -1.0*self._timeDerivative() # minus sign for exponential decay equation
        rate_s = np.real(np.trace(np.dot(density_time_derivative, self.density_avg.spin)))
        return rate_s

    def spinRelaxationRate(self):
        density_time_derivative = -1.0*self._timeDerivative() # minus sign for exponential decay equation
        rate_x = np.real(np.trace(np.dot(density_time_derivative, Pauli.x())))
        rate_y = np.real(np.trace(np.dot(density_time_derivative, Pauli.y())))
        rate_z = np.real(np.trace(np.dot(density_time_derivative, Pauli.z())))
        return rate_x, rate_y, rate_z

    def _timeDerivative(self):
        precession = np.zeros_like(self.density_vals)
        time_derivative = np.zeros_like(self.density_avg.matrix)
        for i in range(self.num_grid):
            rho_p = np.reshape(self.density_vals[:, i], (2, 2))
            precession[:, i] = np.reshape(self._precession(i, rho_p), (4, ))
        perimeter = Ellipse.perimeter(self.band.a, self.band.b)
        norm_tangent = Ellipse.normTangent(self.theta_vals, self.band.a, self.band.b)
        time_derivative[0, 0] = Integral.trapz(self.theta_vals, 1.0/perimeter*norm_tangent*precession[0, :])
        time_derivative[0, 1] = Integral.trapz(self.theta_vals, 1.0/perimeter*norm_tangent*precession[1, :])
        time_derivative[1, 0] = Integral.trapz(self.theta_vals, 1.0/perimeter*norm_tangent*precession[2, :])
        time_derivative[1, 1] = Integral.trapz(self.theta_vals, 1.0/perimeter*norm_tangent*precession[3, :])               
        return time_derivative
    
    def _averageCollision(self, theta_index):
        collision = np.zeros((2, 2), dtype=np.complex)
        w = self._transitionRate(theta_index)
        collision[0, 0] = Integral.trapz(self.theta_vals, w*self.density_vals[0, :])
        collision[0, 1] = Integral.trapz(self.theta_vals, w*self.density_vals[1, :])
        collision[1, 0] = Integral.trapz(self.theta_vals, w*self.density_vals[2, :])
        collision[1, 1] = Integral.trapz(self.theta_vals, w*self.density_vals[3, :])
        return 1.0/(2.0*np.pi)*collision

    def _avergeDensityPerturbation(self):
        average = np.zeros_like(self.density_vals[:, 0])
        average[0] = Integral.trapz(self.theta_vals, np.real(self.density_vals[0, :]))
        average[1] = Integral.trapz(self.theta_vals, np.real(self.density_vals[1, :]))
        average[2] = Integral.trapz(self.theta_vals, np.real(self.density_vals[2, :]))
        average[3] = Integral.trapz(self.theta_vals, np.real(self.density_vals[3, :]))
        return average

    def sumTransitionRate(self, theta_index):
        w = self._transitionRate(theta_index)
        return 1.0/(2.0*np.pi)*Integral.trapz(self.theta_vals, w)

    def _transitionRate(self, theta_index):
        norm_tangent = Ellipse.normTangent(self.theta_vals, self.band.a, self.band.b)
        q = Ellipse.chord(self.theta_vals[theta_index], self.theta_vals, self.band.a, self.band.b)
        potential_squared = self.scatterer.potentialFourier(q)**2
        gradient_reciprocal = np.reciprocal(self.band.gradientEnergy(self.theta_vals))
        return norm_tangent*potential_squared*gradient_reciprocal

    def _precession(self, theta_index, density_matrix):
        spin_orbit = self.band.spinOrbit(self.theta_vals[theta_index])
        return 1.0j/2.0*(np.dot(density_matrix, spin_orbit) - np.dot(spin_orbit, density_matrix))
    
    def spinPerturbation(self):
        spin_x = np.zeros(self.num_grid, dtype=np.float) 
        spin_y = np.zeros(self.num_grid, dtype=np.float) 
        spin_z = np.zeros(self.num_grid, dtype=np.float) 
        for i in range(self.num_grid):
            rho_p = np.reshape(self.density_vals[:, i], (2, 2))
            spin_x[i] = np.real(np.trace(np.dot(rho_p, Pauli.x())))
            spin_y[i] = np.real(np.trace(np.dot(rho_p, Pauli.y())))
            spin_z[i] = np.real(np.trace(np.dot(rho_p, Pauli.z())))
        return spin_x, spin_y, spin_z
