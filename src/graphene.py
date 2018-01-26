import scipy.constants as sc
import numpy as np

class Graphene(object):

	eps0 = sc.epsilon_0
	hbar = sc.hbar
	e = sc.e
	me = sc.electron_mass
	a0 = 4.0*np.pi*eps0*hbar**2/(me*e**2) # Bohr radius
	Eh = me*e**4/(4.0*np.pi*eps0*hbar)**2 # Hartree energy

	def __init__(self, a=0.142, t0=3, t1=0.3, tD=0.0, tR=0.0, delta=0.0, layer='mo'):

		self.lattice_constant = a*1e-9/self.a0 # nm to bohr radius
		self.t0 = t0*self.e/self.Eh # SWMc parameter, eV to Hartree energy
		self.t1 = t1*self.e/self.Eh # SWMc parameter, eV to Hartree energy
		self.tD = tD*self.e/self.Eh # Dresselhaus parameter, eV to Hartree energy
		self.tR = tR*self.e/self.Eh # Rashba parameter, eV to Hartree energy
		self.delta = delta*self.e/self.Eh # interlayer energy asymmetry, eV to Hartree energy

		if layer == 'mo':
			self.layer = layer
		elif layer == 'bi':
			self.layer = layer
		else:
			self.layer = 'mo'
			raise ValueError('Use layer=\'mo\' for monolayer or layer=\'bi\' for bilayer.')

	def bandMono(self, kx, ky):

		H = self.hamiltonianMono(kx, ky)
		w, v = np.linalg.eig(H)

		return w

	def bandBi(self, kx, ky):

		H = self.hamiltonianBi(kx, ky)
		w, v = np.linalg.eig(H)

		return w

	def hamiltonianMono(self, kx, ky):

		non_interacting = self.t0*self._M04x4(kx, ky)
		dresselhaus = self.tD*self._MD4x4(kx, ky)
		rashba = self.tR*self._MR4x4(kx, ky)

		return non_interacting + dresselhaus + rashba

	def hamiltonianBi(self, kx, ky):

		non_interacting = self.t0*self._M08x8(kx, ky)
		dresselhaus = self.tD*self._MD8x8(kx, ky)
		rashba = self.tR*self._MR8x8(kx, ky)

		return non_interacting + dresselhaus + rashba

	def _MR8x8(self, kx, ky):

		M4x4 = self._MR4x4(kx, ky)
		M = np.zeros((8, 8), dtype=np.complex)
		M[0:4, 0:4] = M4x4
		M[4:8, 4:8] = M4x4

		return M

	def _MD8x8(self, kx, ky):

		M4x4 = self._MD4x4(kx, ky)
		M = np.zeros((8, 8), dtype=np.complex)
		M[0:4, 0:4] = M4x4
		M[4:8, 4:8] = M4x4

		return M

	def _M08x8(self, kx, ky):

		A = self._A(kx, ky)
		M4x4 = self._M04x4(kx, ky)
		I = np.identity(4)
		M = np.zeros((8, 8), dtype=np.complex)
		M[0:4, 0:4] = -self.delta/2.0*I + self.t0*M4x4
		M[4:8, 4:8] = self.delta/2.0*I + self.t0*M4x4
		M[0:4, 4:8] = A
		M[4:8, 0:4] = np.conjugate(np.transpose(A))

		return M

	def _A(self, kx, ky):

		A = np.zeros((4, 4), dtype=np.complex)
		A[0, 2] = self.t1
		A[1, 3] = self.t1

		return A
	
	def _M04x4(self, kx, ky): 

		gam = self._gamma(kx, ky)
		gamc = np.conjugate(gam)
		M = np.zeros((4, 4), dtype=np.complex)
		M[0, 2] = gam
		M[1, 3] = gam
		M[2, 0] = gamc
		M[3, 1] = gamc

		return M

	def _MD4x4(self, kx, ky):

		eta = self._eta(kx, ky)
		M = np.zeros((4, 4), dtype=np.complex)
		M[0, 0] = eta
		M[1, 1] = -eta
		M[2, 2] = -eta
		M[3, 3] = eta

		return M

	def _MR4x4(self, kx, ky):

		N = self._N(kx, ky)
		Nd = np.conjugate(np.transpose(N))
		M = np.zeros((4, 4), dtype=np.complex)
		M[0:2, 2:4] = N
		M[2:4, 0:2] = Nd

		return M

	def _N(self, kx, ky):

		xi_1 = self._xi1(kx, ky)
		xi_2 = self._xi2(kx, ky)
		N = np.zeros((2, 2), dtype=np.complex)
		N[0, 1] = 1j*(xi_1 + xi_2)
		N[1, 0] = 1j*(xi_1 - xi_2)

		return N

	def _xi1(self, kx, ky):

		a = self.lattice_constant
		xi1 = np.exp(1j*0.5*a*kx)*(np.exp(-1j*np.sqrt(3.0)/2.0*a*ky) - np.cos(0.5*a*kx))

		return xi1

	def _xi2(self, kx, ky):

		a = self.lattice_constant
		xi2 = np.sqrt(3.0)*np.exp(1j*0.5*a*kx)*np.sin(0.5*a*kx)

		return xi2

	def _eta(self, kx, ky):

		a = self.lattice_constant
		eta = 2.0*np.sin(a*kx) - 4.0*np.sin(a*kx/2.0)*np.cos(a*ky*np.sqrt(3.0)/2.0)

		return eta

	def _gamma(self, kx, ky):

		a = self.lattice_constant
		gammak = np.exp(1j*a/np.sqrt(3.0)*ky)*(1.0 + 2.0*np.exp(-1j*np.sqrt(3.0)/2.0*a*ky)*np.cos(a/2.0*kx))

		return gammak
