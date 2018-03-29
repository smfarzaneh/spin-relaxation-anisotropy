import numpy as np

class Evolution(object):

	def __init__(self, hamiltonian, density_operator, t_now=0.0):

		self.H = hamiltonian # in units of 1/s
		self.rho = density_operator
		self.t = t_now

	def evolve(self, t_final, time_step, method='rk2'):

		if t_final < self.t:
			raise ValueError('Invalid final time')

		h = time_step
		rho_now = self.rho

		while self.t < t_final:

			if self.t + h >= t_final:
				h = t_final - self.t

			if method == 'rk2':
				rho_now = self._rungeKuttaSecond(rho_now, h)
			else:
				raise ValueError('Invalid method')
		
			# update time and density operator
			self.t += h
			self.rho = rho_now

	def _rungeKuttaSecond(self, rho_now, h):

		k1 = h*self._integrand(rho_now)
		k2 = h*self._integrand(rho_now + k1/2.0)
		rho_h = rho_now + k2

		return rho_h

	def _integrand(self, rho_now):

		return 1.j*self._commutator(rho_now, self.H)

	def _commutator(self, A, B):

		return np.dot(A, B) - np.dot(B, A)
