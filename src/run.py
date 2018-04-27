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
    # compute(solver)
    # plotSpinPerturbation(solver)
    spinRelaxationAnisotropy(solver)
    # plotSpinRelaxationAnisotropy()
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    kx = Ellipse.coordinateX(solver.theta_vals, solver.band.a, solver.band.b)
    ky = Ellipse.coordinateY(solver.theta_vals, solver.band.a, solver.band.b)
    spin_x, spin_y, spin_z = solver.spinPerturbation()
    ax.scatter(kx, ky, spin_x, c='r', marker='o', facecolors='none', label=r'$s_x$')
    ax.scatter(kx, ky, spin_y, c='b', marker='*', facecolors='none', label=r'$s_y$')
    ax.scatter(kx, ky, spin_z, c='g', marker='^', facecolors='none', label=r'$s_z$')
    ax.legend()
    ax.grid(linestyle=':')
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    # plt.locator_params(axis='x', nticks=7)
    # plt.locator_params(axis='y', nticks=7)
    ax.set_zlim((-2e-4, 2e-4))
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_zlabel(r'Spin Perturbation')
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

def spinRelaxationAnisotropy(solver):
    num_grid = 5
    mx = np.ones(2*num_grid + 1, dtype=np.float)
    my = np.ones(2*num_grid + 1, dtype=np.float)
    ratio = np.logspace(-2.0, 2.0, 2*num_grid + 1)
    mx[num_grid + 1:] = 1.0/ratio[num_grid + 1:]
    my[:num_grid] = ratio[:num_grid]
    rate_s = np.zeros((9, len(my)), dtype=np.float)
    for i in range(len(my)):
        solver.band.mxc = mx[i]
        solver.band.myc = my[i]
        solver.band.updateEnergyDependentParameters()
        solver.iterate()
        rate_s[0, i], rate_s[1, i], rate_s[2, i] = solver.spinRelaxationRate()
        if i == 0:
            pass
            # solver.num_iteration = 100
        solver.reset()
    print('x-polarized done.')
    solver.num_iteration = 400
    solver.density_avg.update(0.0, 1.0, 0.0)
    solver.reset()
    for i in range(len(my)):
        solver.band.myc = my[i]
        solver.band.updateEnergyDependentParameters()
        solver.iterate()
        rate_s[3, i], rate_s[4, i], rate_s[5, i] = solver.spinRelaxationRate()
        if i == 0:
            pass
            # solver.num_iteration = 100
        solver.reset()
    print('y-polarized done.')
    solver.num_iteration = 400
    solver.density_avg.update(0.0, 0.0, 1.0)
    solver.reset()
    for i in range(len(my)):
        solver.band.myc = my[i]
        solver.band.updateEnergyDependentParameters()
        solver.iterate()
        rate_s[6, i], rate_s[7, i], rate_s[8, i] = solver.spinRelaxationRate()
        if i == 0:
            pass
            # solver.num_iteration = 100
        solver.reset()
    print('z-polarized done.')
    # save to file
    data = np.vstack((my, rate_s))
    saveData(data, 'aniso.out')
    
def plotSpinRelaxationAnisotropy():
    data = loadData('aniso.out')
    mass_ratio = data[0, :]
    rate_xx = data[1, :]
    rate_yy = data[5, :]
    rate_zz = data[9, :]
    fig, ax = plt.subplots()
    ax.plot(mass_ratio, rate_yy/rate_xx, marker='o', markerfacecolor='None', label=r'$\tau_{s,yy}/\tau_{s,xx}$')
    ax.plot(mass_ratio, rate_zz/rate_yy, marker='s', markerfacecolor='None', label=r'$\tau_{s,zz}/\tau_{s,yy}$')
    ax.plot(mass_ratio, rate_xx/rate_zz, marker='^', markerfacecolor='None', label=r'$\tau_{s,xx}/\tau_{s,zz}$')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Effective Mass Ratio, $m_y/m_x$')
    ax.set_ylabel(r'Spin Relaxation Ratio, $\tau_{s,\alpha\alpha}/\tau_{s,\beta\beta}$')
    ax.grid(linestyle=':')
    fig.set_size_inches(3.5, 3.5)
    saveFigure(fig, 'aniso.pdf')

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