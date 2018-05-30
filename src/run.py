import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import matplotlib.cbook as cbook
from matplotlib.colorbar import Colorbar
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 8
rcParams['text.usetex'] = True
from matplotlib.ticker import MultipleLocator

from solver import *
from constant import *
from band import *
from scatterer import *
from density import *
from rw import *

def main():
    solver_coulomb = setupSolver('coulomb')
    solver_defect = setupSolver('defect')
    # taskPerturbation(solver)
    # anisotropy(solver_defect)
    energy(solver_defect)
    # spinRelaxationVsPolarization(solver)
    # plotSpinRelaxationVsPolarization()
    # plotSpinRelaxationVsPolarization2D()
    # plotRashbaField(solver)
    # num_cores = multiprocessing.cpu_count()
    # print(num_cores)
    
def setupBandStructure():
    band_gap = 1.0*Constant.e/Constant.Ry # eV to AU
    mx = 1.0
    my = 0.1
    rashba_strength = 1.0
    energy_level = 0.6*Constant.e/Constant.Ry # eV to AU
    band_structure = Band(band_gap, mx, mx, my, my, rashba_strength, energy_level)
    return band_structure

def setupSpinPolarization(s_x = 1.0, s_y = 0.0, s_z = 0.0):
    norm = np.sqrt(s_x**2 + s_y**2 + s_z**2)
    return Density(s_x/norm, s_y/norm, s_z/norm) 

def setupSolver(potential):
    band_structure = setupBandStructure()
    scatterer = Scatterer(potential_type=potential)
    density = setupSpinPolarization()
    solver = Solver(band_structure, scatterer, density)
    return solver

def compute(solver):
    print('iterative solver, solving for perturbation density matrix.')
    print(str(solver.num_iteration) + ' iterations, ' + str(solver.num_grid) + ' grid points.')
    solver.iterate()

def taskPerturbation(solver):
    solver.num_grid = 75
    solver.num_iteration = 1
    # isotropic
    solver.band.mxc = 1.0
    solver.band.myc = 1.0
    kx, ky, spin_z = perturbation(solver)
    data = np.stack((kx, ky, spin_z), axis=0)
    RW.saveData(data, 'perturbation-iso-defect.out')
    # anisotropic
    solver.band.mxc = 0.1
    solver.band.myc = 1.0
    kx, ky, spin_z = perturbation(solver)
    data = np.stack((kx, ky, spin_z), axis=0)
    RW.saveData(data, 'perturbation-aniso-defect.out')

def perturbation(solver):
    solver.reset()
    solver.band.updateEnergyDependentParameters()
    compute(solver)
    _, _, spin_z = solver.spinPerturbation()
    kx = Ellipse.coordinateX(solver.theta_vals, solver.band.a, solver.band.b)
    ky = Ellipse.coordinateY(solver.theta_vals, solver.band.a, solver.band.b)
    return kx, ky, spin_z

def plotRashbaField(solver):
    x_vals = Ellipse.coordinateX(solver.theta_vals, solver.band.a, solver.band.b)
    y_vals = Ellipse.coordinateY(solver.theta_vals, solver.band.a, solver.band.b)
    # plot Fermi contour
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, zorder=1, color='#57068c')
    # plot Rashba field
    theta = np.array([0.0, np.pi/16.0, np.pi/8.0, np.pi/4.0, np.pi/2.0, 3.0*np.pi/4.0, 7.0*np.pi/8.0, 15.0*np.pi/16.0])
    theta = np.hstack((theta, theta + np.pi))
    for i in range(len(theta)):
        x = Ellipse.coordinateX(theta[i], solver.band.a, solver.band.b)
        y = Ellipse.coordinateY(theta[i], solver.band.a, solver.band.b)
        dx, dy = solver.band.fieldRashba(theta[i])
        ax.arrow(x, y, dx/10., dy/10., width=0.0005, fc='#799a05', ec='#799a05')
    ax.set_xlabel(r'$k_x$ $[1/a_0]$')
    ax.set_ylabel(r'$k_y$ $[1/a_0]$')
    ax.set_xlim((-0.1, 0.1))
    ax.set_ylim((-0.03, 0.03))
    fig.set_size_inches(3.5, 1.05)
    RW.saveFigure(fig, 'rashba.pdf')

def process(arg):
    solver, myc= arg
    solver.band.myc = myc
    solver.band.updateEnergyDependentParameters()
    solver.iterate()
    rate_x, rate_y, rate_z = solver.spinRelaxationRate()
    return rate_x, rate_y, rate_z

def polarizationBlackPhosphorus(solver):
    solver.band.mxc = 1.26
    solver.band.myc = 0.17
    solver.band.updateEnergyDependentParameters()
    spinRelaxationVsPolarization(solver, filename='polarization-bp.out', grid_num=20)

def spinRelaxationVsPolarization(solver, filename='polarization.out', grid_num=10):
    phi = np.linspace(0.0, 1.0*np.pi, grid_num)
    theta = np.linspace(0.0, np.pi/2.0, grid_num)
    rate_s = np.zeros((grid_num, grid_num))
    for i in range(grid_num):
        for j in range(grid_num):
            s_x = np.sin(theta[j])*np.cos(phi[i])
            s_y = np.sin(theta[j])*np.sin(phi[i])
            s_z = np.cos(theta[j])
            density = setupSpinPolarization(s_x, s_y, s_z)
            solver.density_avg = density
            solver.reset()
            solver.iterate()
            rate_s[i, j] = solver.spinRelaxationSingle()
            print(str(i*10 + j + 1) + '/' + str(grid_num*grid_num) + ' done.')
    angle = np.vstack((phi, theta))
    data = np.vstack((angle, rate_s))
    RW.saveData(data, filename)

def plotSpinRelaxationVsPolarization2D():
    data = RW.loadData('polarization-aniso-reduced.out')
    num_grid = len(data[0, :])
    phi = np.hstack((data[0, :], data[0, :] + np.pi))
    theta = np.hstack((data[1, :], data[1, :] + np.pi/2.0))[::-1]
    rate_s = np.zeros((num_grid*2, num_grid*2))
    rate_s[0:num_grid, 0:num_grid] = data[2:, :]
    rate_s[0:num_grid, num_grid:2*num_grid] = np.fliplr(data[2:, :])
    rate_s[num_grid:2*num_grid, 0:num_grid] = np.flipud(data[2:, :])
    rate_s[num_grid:2*num_grid, num_grid:2*num_grid] = np.fliplr(np.flipud(data[2:, :]))
    rate_s = np.transpose(rate_s)
    rate_s = rate_s/Constant.a0**2 # AU to SI
    fig, ax = plt.subplots()
    PHI, THETA = np.meshgrid(phi, theta)
    plt.contourf(PHI, THETA, rate_s)
    labels_theta = [r'$0$', r'$\pi/2$', '$\pi$']
    ticks_theta = [0.0, np.pi/2.0, np.pi]
    labels_phi = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    ticks_phi = [0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi]
    plt.xticks(ticks_phi, labels_phi)
    plt.yticks(ticks_theta, labels_theta)
    plt.colorbar()
    plt.title(r'$1/\tau_{\hat{s}\hat{s}}$ $[\lambda^2_R/n_i]$, $\hat{s} = \hat{s}(\theta, \phi)$')
    ax.set_xlabel(r'Azimuthal $\phi$')
    ax.set_ylabel(r'Polar $\theta$')
    fig.set_size_inches(3.5, 3.5)
    RW.saveFigure(fig, 'polarization2d.pdf')

def plotSpinRelaxationVsPolarization():
    data = RW.loadData('polarization-aniso-reduced.out')
    num_grid = len(data[0, :])
    phi = np.zeros(num_grid*2)
    theta = np.zeros(num_grid*2)
    rate_s = np.tile(data[2:, :], (2, 2))
    phi[:num_grid] = data[0, :]
    phi[num_grid:] = data[0, :] + np.pi
    theta[:num_grid] = data[1, :]
    theta[num_grid:] = data[1, :] + np.pi
    x = np.zeros_like(rate_s)
    y = np.zeros_like(rate_s)
    z = np.zeros_like(rate_s)
    for i in range(num_grid*2):    
        for j in range(num_grid*2):
            x[i, j] = 1.0/rate_s[i, j]*np.sin(theta[j])*np.cos(phi[i])
            y[i, j] = 1.0/rate_s[i, j]*np.sin(theta[j])*np.sin(phi[i])
            z[i, j] = 1.0/rate_s[i, j]*np.cos(theta[j])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)
    fig.set_size_inches(3.5, 3.5)
    RW.saveFigure(fig, 'polarization-aniso.pdf')

def energy(solver):
    if solver.scatterer.type == 'coulomb':
        filename = 'energy.out'
    elif solver.scatterer.type == 'defect':
        solver.num_iteration = 1
        filename = 'energy-defect.out'
    else: 
        raise ValueError('invalid scattering potential.')
    solver.band.mxc = 1.26 # Black Phosphorus 
    solver.band.myc = 0.17 # Black Phosphorus
    solver.band.updateEnergyDependentParameters()
    num_grid = 10
    energy_level = np.linspace(0.5 + 0.01, 0.8, num_grid)*Constant.e/Constant.Ry # eV to AU
    rate_s = np.zeros((9, len(energy_level)), dtype=np.float)
    solver.density_avg.update(1.0, 0.0, 0.0)
    for i in range(len(energy_level)):
        solver.band.energy = energy_level[i]
        solver.band.updateEnergyDependentParameters()
        solver.iterate()
        rate_s[0, i], rate_s[1, i], rate_s[2, i] = solver.spinRelaxationRate()
        solver.reset()
    print('x-polarized done.')
    solver.density_avg.update(0.0, 1.0, 0.0)
    for i in range(len(energy_level)):
        solver.band.energy = energy_level[i]
        solver.band.updateEnergyDependentParameters()
        solver.iterate()
        rate_s[3, i], rate_s[4, i], rate_s[5, i] = solver.spinRelaxationRate()
        solver.reset()
    print('y-polarized done.')
    solver.density_avg.update(0.0, 0.0, 1.0)
    for i in range(len(energy_level)):
        solver.band.energy = energy_level[i]
        solver.band.updateEnergyDependentParameters()
        solver.iterate()
        rate_s[6, i], rate_s[7, i], rate_s[8, i] = solver.spinRelaxationRate()
        solver.reset()
    print('z-polarized done.')
    # save to file
    data = np.vstack((energy_level, rate_s))
    RW.saveData(data, filename)

def anisotropy(solver):
    if solver.scatterer.type == 'coulomb':
        filename = 'anisotropy.out'
    elif solver.scatterer.type == 'defect':
        solver.num_iteration = 1
        filename = 'anisotropy-defect.out'
    else: 
        raise ValueError('invalid scattering potential.')
    num_grid = 5
    mx = np.logspace(0.0, -2.0, 2*num_grid + 1)
    my = np.logspace(-2.0, 0.0, 2*num_grid + 1)
    ratio = my/mx
    rate_s = np.zeros((9, len(ratio)), dtype=np.float)
    solver.density_avg.update(1.0, 0.0, 0.0)
    for i in range(len(ratio)):
        solver.band.mxc = mx[i]
        solver.band.myc = my[i]
        solver.band.updateEnergyDependentParameters()
        solver.iterate()
        rate_s[0, i], rate_s[1, i], rate_s[2, i] = solver.spinRelaxationRate()
        solver.reset()
    print('x-polarized done.')
    solver.density_avg.update(0.0, 1.0, 0.0)
    for i in range(len(ratio)):
        solver.band.mxc = mx[i]
        solver.band.myc = my[i]
        solver.band.updateEnergyDependentParameters()
        solver.iterate()
        rate_s[3, i], rate_s[4, i], rate_s[5, i] = solver.spinRelaxationRate()
        solver.reset()
    print('y-polarized done.')
    solver.density_avg.update(0.0, 0.0, 1.0)
    for i in range(len(ratio)):
        solver.band.mxc = mx[i]
        solver.band.myc = my[i]
        solver.band.updateEnergyDependentParameters()
        solver.iterate()
        rate_s[6, i], rate_s[7, i], rate_s[8, i] = solver.spinRelaxationRate()
        solver.reset()
    print('z-polarized done.')
    # save to file
    data = np.vstack((ratio, rate_s))
    RW.saveData(data, filename)

# run
main()