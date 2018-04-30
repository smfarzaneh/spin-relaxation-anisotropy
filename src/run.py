import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 9
rcParams['text.usetex'] = True
from joblib import Parallel, delayed
import multiprocessing

from solver import *
from constant import *
from band import *
from scatterer import *
from density import *

def main():
    solver = setupSolver()
    # plotSpinPerturbation(solver)
    # spinRelaxationAnisotropy(solver)
    # plotSpinRelaxationAnisotropy()
    # spinRelaxationVsEnergy(solver)
    plotSpinRelaxationVsEnergy()
    # num_cores = multiprocessing.cpu_count()
    # print(num_cores)
    
def setupBandStructure():
    band_gap = 1.0*Constant.e/Constant.Ry # eV to AU
    mx = 1.0
    my = 1.0
    rashba_strength = 1.0
    energy_level = 0.6*Constant.e/Constant.Ry # eV to AU
    band_structure = Band(band_gap, mx, mx, my, my, rashba_strength, energy_level)
    return band_structure

def setupSpinPolarization():
    s_x = 1.0
    s_y = 0.0
    s_z = 0.0
    norm = np.sqrt(s_x**2 + s_y**2 + s_z**2)
    return Density(s_x/norm, s_y/norm, s_z/norm) 

def setupSolver():
    band_structure = setupBandStructure()
    scatterer = Scatterer()
    density = setupSpinPolarization()
    solver = Solver(band_structure, scatterer, density)
    return solver

def compute(solver):
    print('iterative solver, solving for perturbation density matrix.')
    print(str(solver.num_iteration) + ' iterations, ' + str(solver.num_grid) + ' grid points.')
    solver.iterate()

def plotSpinPerturbation(solver):
    solver.num_grid = 100
    solver.reset()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    kx = Ellipse.coordinateX(solver.theta_vals, solver.band.a, solver.band.b)
    ky = Ellipse.coordinateY(solver.theta_vals, solver.band.a, solver.band.b)
    solver.band.mxc = 1.0
    solver.band.myc = 1.0
    solver.band.updateEnergyDependentParameters()
    solver.reset()
    compute(solver)
    spin_x, spin_y, spin_z = solver.spinPerturbation()
    ax.scatter(kx, ky, spin_z, c='r', facecolors='none', marker='o', label=r'$m_y/m_x=1.0$')
    solver.band.mxc = 1.0
    solver.band.myc = 0.1
    solver.band.updateEnergyDependentParameters()
    solver.reset()
    compute(solver)
    spin_x, spin_y, spin_z = solver.spinPerturbation()
    ax.scatter(kx, ky, spin_z, c='g', facecolors='none', marker='s', label=r'$m_y/m_x=0.1$')
    solver.band.mxc = 1.0
    solver.band.myc = 10.
    solver.band.updateEnergyDependentParameters()
    solver.reset()
    compute(solver)
    spin_x, spin_y, spin_z = solver.spinPerturbation()
    ax.scatter(kx, ky, spin_z, c='b', facecolors='none', marker='^', label=r'$m_y/m_x=10.0$')
    ax.legend()
    ax.grid(linestyle=':')
    # ax.set_xlim(-0.1, 0.1)
    # ax.set_ylim(-0.1, 0.1)
    # plt.locator_params(axis='x', nticks=7)
    # plt.locator_params(axis='y', nticks=7)
    # ax.set_zlim((-2e-4, 2e-4))
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_zlabel(r'$s^{\prime}_z$')
    fig.set_size_inches(5, 5)
    saveFigure(fig, 'perturbation.pdf')

def plotRashbaField():
    rlx = setupSolver()
    a = rlx.band.a
    b = rlx.band.b
    k_max = max(a, b)
    X, Y = np.mgrid[-k_max:k_max:100j, -k_max:k_max:100j]        
    omega_vals = rlx.band.fieldRashba(rlx.theta_vals)
    fig, ax = plt.subplots()
    ax.plot(kx_vals, ky_vals)
    saveFigure(fig, 'rashba.pdf')

def process(arg):
    solver, myc= arg
    solver.band.myc = myc
    solver.band.updateEnergyDependentParameters()
    solver.iterate()
    rate_x, rate_y, rate_z = solver.spinRelaxationRate()
    return rate_x, rate_y, rate_z

def spinRelaxationVsEnergy(solver):
    solver.num_iteration = 100
    solver.band.mxc = 1.0
    solver.band.mxy = 0.1
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
    saveData(data, 'energy.out')

def plotSpinRelaxationVsEnergy():
    data = loadData('energy.out')
    energy_level = data[0, :]/Constant.e*Constant.Ry # AU to eV
    rate_xx = data[1, :]
    rate_yy = data[5, :]
    rate_zz = data[9, :]
    fig, ax = plt.subplots()
    ax.plot(energy_level - 0.5, rate_xx, c='r', marker='o', markerfacecolor='None', label=r'$1/\tau_{s,xx}$')
    ax.plot(energy_level - 0.5, rate_yy, c='g', marker='s', markerfacecolor='None', label=r'$1/\tau_{s,yy}$')
    ax.plot(energy_level - 0.5, rate_zz, c='b', marker='^', markerfacecolor='None', label=r'$1/\tau_{s,zz}$')
    ax.legend()
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$E$ - $E_g/2$ [eV]')
    ax.set_ylabel(r'Spin Relaxation Rate $\times \lambda^2_{R}$')
    ax.grid(linestyle=':', which="major", axis='x')
    ax.grid(linestyle=':', which="both", axis='y')
    fig.set_size_inches(3.5, 3.5)
    saveFigure(fig, 'energy.pdf')

def spinRelaxationAnisotropy(solver):
    num_grid = 5
    mx = np.ones(2*num_grid + 1, dtype=np.float)
    my = np.ones(2*num_grid + 1, dtype=np.float)
    ratio = np.logspace(-2.0, 2.0, 2*num_grid + 1)
    mx[num_grid + 1:] = 1.0/ratio[num_grid + 1:]
    my[:num_grid] = ratio[:num_grid]
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
    saveData(data, 'anisotropy.out')
    
def plotSpinRelaxationAnisotropy():
    data = loadData('anisotropy.out')
    mass_ratio = data[0, :]
    rate_xx = data[1, :]
    rate_yy = data[5, :]
    rate_zz = data[9, :]
    fig, ax = plt.subplots()
    ax.plot(mass_ratio, rate_xx/rate_yy, c='r', marker='o', markerfacecolor='None', label=r'$\tau_{s,yy}/\tau_{s,xx}$')
    ax.plot(mass_ratio, rate_yy/rate_zz, c='g', marker='s', markerfacecolor='None', label=r'$\tau_{s,zz}/\tau_{s,yy}$')
    ax.plot(mass_ratio, rate_zz/rate_xx, c='b', marker='^', markerfacecolor='None', label=r'$\tau_{s,xx}/\tau_{s,zz}$')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Effective Mass Ratio, $m_y/m_x$')
    ax.set_ylabel(r'Spin Relaxation Ratio, $\tau_{s,\alpha\alpha}/\tau_{s,\beta\beta}$')
    ax.grid(linestyle=':', which="major", axis='x')
    ax.grid(linestyle=':', which="both", axis='y')
    fig.set_size_inches(3.5, 3.5)
    saveFigure(fig, 'anisotropy.pdf')

def loadData(filename):
    directory = getDirectory()
    data = np.loadtxt(directory + filename, delimiter=',')
    return data

def saveData(data, filename):
    directory = getDirectory()
    np.savetxt(directory + filename, data, delimiter=',', fmt='%1.4e')
    print(filename + str(' was saved.'))

def saveFigure(fig, filename):
    directory = getDirectory()
    fig.savefig(directory + filename, bbox_inches='tight')
    print(filename + str(' was saved.'))

def getDirectory(directory='../out/'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# run
main()