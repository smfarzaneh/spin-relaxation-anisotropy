import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import matplotlib.cbook as cbook
from matplotlib.colorbar import Colorbar
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

from rw import *
from constant import *

def globalSettings():
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 8
    rcParams['text.usetex'] = True

globalSettings()

def do():
    fig, ax = plt.subplots(1,1)
    plotAnisotropy(fig, ax)
    fig, ax = plt.subplots(1,1)
    plotEnergy(fig, ax)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotPerturbation(fig, ax)

def plotAnisotropy(fig, ax):
    data = RW.loadData('anisotropy.out')
    mass_ratio = data[0, :]
    rate_xx = data[1, :]
    rate_yy = data[5, :]
    rate_zz = data[9, :]
    ax.plot(mass_ratio, rate_xx/rate_yy, c='k', marker='o', markerfacecolor='None', label=r'$\tau_{s,yy}/\tau_{s,xx}$', linewidth=1.0)
    ax.plot(mass_ratio, rate_yy/rate_zz, c='k', marker='s', markerfacecolor='None', label=r'$\tau_{s,zz}/\tau_{s,yy}$', linewidth=1.0)
    ax.plot(mass_ratio, rate_zz/rate_xx, c='k', marker='^', markerfacecolor='None', label=r'$\tau_{s,xx}/\tau_{s,zz}$', linewidth=1.0)
    ax.legend(loc=9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$m_y/m_x$')
    ax.set_ylabel(r'$\tau_{s,\alpha\alpha}/\tau_{s,\beta\beta}$')
    ax.grid(linestyle=':', which="major", axis='x')
    ax.grid(linestyle=':', which="both", axis='y')
    fig.set_size_inches(3.37, 3.37)
    RW.saveFigure(fig, 'anisotropy.pdf')

def plotEnergy(fig, ax):
    data = RW.loadData('energy.out')
    energy_level = data[0, :]/Constant.e*Constant.Ry # AU to eV
    rate_xx = data[1, :]/Constant.a0**2
    rate_yy = data[5, :]/Constant.a0**2
    rate_zz = data[9, :]/Constant.a0**2
    ax.plot(energy_level - 0.5, rate_xx, c='k', marker='o', markerfacecolor='None', label=r'$1/\tau_{s,xx}$', linewidth=1.0)
    ax.plot(energy_level - 0.5, rate_yy, c='k', marker='s', markerfacecolor='None', label=r'$1/\tau_{s,yy}$', linewidth=1.0)
    ax.plot(energy_level - 0.5, rate_zz, c='k', marker='^', markerfacecolor='None', label=r'$1/\tau_{s,zz}$', linewidth=1.0)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel(r'$E - E_\mathrm{g}/2$ [eV]')
    ax.set_ylabel(r'$1/\tau_{s,\alpha\alpha}$ $[\lambda^2_\mathrm{R}/n_\mathrm{i}]$')
    ax.grid(linestyle=':', which="major", axis='x')
    ax.grid(linestyle=':', which="major", axis='y')
    fig.set_size_inches(3.37, 3.37)
    RW.saveFigure(fig, 'energy.pdf')

def plotPerturbation(fig, ax):
    # isotropic 
    data = RW.loadData('perturbation-iso.out')
    kx = data[0, :]
    ky = data[1, :]
    spin = data[2, :]
    ax.scatter(kx, ky, spin, color='k', facecolor='none', marker='o', label=r'$m_y/m_x=1.0$')
    # anisotropic 
    data = RW.loadData('perturbation-aniso.out')
    kx = data[0, :]
    ky = data[1, :]
    spin = data[2, :]
    ax.scatter(kx, ky, spin, color='k', facecolor='none', marker='^', label=r'$m_y/m_x=10.0$')
    # figure settings
    ax.legend(loc=2, bbox_to_anchor=(0.1, 1.05))
    ax.set_xlabel(r'$k_x$ $[1/a_0]$')
    ax.set_ylabel(r'$k_y$ $[1/a_0]$')
    ax.text(0.07, 0.07, 0.00018, r'$s^{\prime}_z$ $[\lambda^2_\mathrm{R}/n_\mathrm{i}]$')
    ax.xaxis.labelpad = -7 
    ax.yaxis.labelpad = 2 
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlim([-0.075, 0.075])
    ax.set_ylim([-0.075, 0.075])
    ax.set_zlim([-0.00015, 0.00015])
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.zaxis.set_major_locator(MultipleLocator(0.0001))
    ax.tick_params(axis='x', which='major', pad=-5)
    ax.tick_params(axis='y', which='major', pad=-4)
    ax.tick_params(axis='z', which='major', pad=3)
    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    # ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    # ax.grid(False)
    ax.view_init(elev=30, azim=-60)
    # ax.xaxis.pane.set_edgecolor('k')
    # ax.yaxis.pane.set_edgecolor('k')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    fig.set_size_inches(3.37, 3.37)
    RW.saveFigure(fig, 'perturbation.pdf')

do()
