import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u

#   universal gas constant (erg/mol/K)
RGAS = 8.3143e7
AVOG = 6.02e23
KB = RGAS / AVOG
PF = 1000000  # Reference pressure [dyn / cm2]

def plot_full_structure(self, y, title=''):
    """
    Plot the cloud structure.

    :param self: Nimbus object
    :param y: solution of solve ivp
    :param title: title for the plot
    :return:
    """

    # ==== General plotting set up
    plt.figure()
    plt.title(title)
    plt.plot([], [], color='k', linestyle='-', label='xv')
    plt.plot([], [], color='k', linestyle=':', label='xc')
    plt.plot([], [], color='k', linestyle='--', label='xn')

    # ==== show time evolution and convergence of the model
    nit = len(y[0])-1
    for i in range(nit):
        plt.plot(y[:self.sz, i], self.pres, color=cm.viridis(i/nit), linestyle='-', alpha=0.3)
        plt.plot(y[self.sz:self.sz*2, i], self.pres, color=cm.viridis(i/nit), linestyle=':', alpha=0.3)
        plt.plot(y[self.sz*2:, i], self.pres, color=cm.viridis(i/nit), linestyle='--', alpha=0.3)


    # ==== Plot cloud particle radius (initial and final)
    mp = np.nan_to_num((y[self.sz:self.sz*2, -1]) * self.m_ccn / y[self.sz*2:, -1])
    rg = np.cbrt(3*mp/(4*np.pi*self.rhop))
    rg = np.maximum(rg, self.r_ccn)
    plt.plot(self.rg_in[self.mask_psupsat], self.pres[self.mask_psupsat], color='k', linestyle='-.', label=r'r$_\mathrm{in}$ [$\mu$m]')
    plt.plot(self.rg[self.mask_psupsat], self.pres[self.mask_psupsat], color='orange', linestyle='-.', label=r'r$_\mathrm{new}$ [$\mu$m]')
    plt.plot(rg[self.mask_psupsat], self.pres[self.mask_psupsat], color='green', linestyle='-.', label=r'r$_\mathrm{out}$ [$\mu$m]')
    plt.vlines([1e-4], self.pres[-1], self.pres[0], linestyle='-.', color='gray', label='1 $\mu$m')
    #
    # ==== plot cloud particle number density
    ncl = y[self.sz * 2:, -1] / self.m_ccn * self.rhoatmo
    ncl_o_ngas = ncl #/ self.natmo
    plt.plot(ncl_o_ngas, self.pres, color='blue', label='n/ngas', linestyle='--')

    # ==== plot fsed
    fsed = self.vsed(rg) / self.kzz * self.h
    plt.plot(fsed*1e-10, self.pres, color='green', label='fsed', linestyle='-')
    plt.plot(np.ones_like(self.pres)*1e-13, self.pres, color='green', linestyle=':')
    plt.plot(np.ones_like(self.pres)*1e-10, self.pres, color='green', linestyle=':')
    plt.plot(np.ones_like(self.pres)*1e-7, self.pres, color='green', linestyle=':')

    # ==== Nucleation and growth rate
    n1 = y[:self.sz, -1] * self.rhoatmo / self.m1
    nuc_rate = np.nan_to_num(self.nuc_rate(n1, self.temp))
    growth_rate = self.acc_rate(self.rg, self.temp, n1, ncl)
    plt.plot(nuc_rate, self.pres, label='nucleation rate', color='red', linestyle='-')
    plt.plot(growth_rate, self.pres, label='accretion rate', color='magenta', linestyle='-')

    # ==== Plot vapour pressure limit
    plt.plot(self.pvap * self.mw / self.pres / self.mmw, self.pres, label='q_vap', color='blue', linestyle='-.')

    # ==== Plot vsed
    plt.plot(self.vsed(self.rg)*1e-10, self.pres, label='v_sed', color='orange', linestyle='-')

    # ==== Plot final time step
    plt.plot(y[:self.sz, -1], self.pres, color='k', linestyle='-')
    plt.plot(y[self.sz:self.sz*2, -1], self.pres, color='k', linestyle=':')
    plt.plot(y[self.sz*2:, -1], self.pres, color='k', linestyle='--')

    # ==== General plotting settings
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc=2)
    plt.xlim(1e-18, 1e0)
    plt.ylim(self.pres[-1], self.pres[0])
    plt.ylabel('pres [dyne/cm2]')
    plt.xlabel('various')
    # plt.savefig(self.working_dir + '/ap_structure_' + title + '.png')
    plt.savefig(self.working_dir + '/ap_structure.png')
    plt.show()
    plt.close()

def plot_initial_conditions(self, x0):
    """
    Simple plotting routine to analyse the initial conditions.

    :param self: Nimbus clss
    :param x0: initial mass mixing ratios
    """
    plt.figure()
    plt.loglog(x0[:self.sz], self.pres, label='xv')
    plt.loglog(x0[self.sz:2*self.sz], self.pres, label='xc')
    plt.loglog(x0[self.sz*2:3*self.sz], self.pres, label='xn')
    plt.ylim(self.pres[-1], self.pres[0])
    plt.legend()
    plt.ylabel('pres [dyne/cm2]')
    plt.xlabel('MMR [g/g]')
    plt.savefig(self.working_dir + '/ap_initial_profiles.png')
