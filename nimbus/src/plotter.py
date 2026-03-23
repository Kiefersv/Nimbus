""" Functions for diagnostic plotting """

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from .atmosphere_physics import mass_to_radius

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
    fig, ax = plt.subplots(1, 6, figsize=(10, 3))
    logp = np.log10(self.pres[self.mask_psupsat])
    def rund(val):
        return np.log10(val[self.mask_psupsat])

    ax[1].plot([], [], color='k', linestyle='-', label='Gas-phase')
    ax[1].plot([], [], color='k', linestyle='-.', label='limit')
    ax[3].plot([], [], color='k', linestyle='-', label='Gas-phase')

    # ==== values
    xrun = y[:, -1].reshape((self.nspec * 2 + 1, self.sz))
    xrun[xrun < self.ode_minimum_mmr] = self.ode_minimum_mmr
    xtot = np.sum(xrun[1::2], axis=0)
    rhotot = np.sum(xrun[1::2] * self.rhop[:, np.newaxis], axis=0) / xtot
    rg = mass_to_radius(self, xrun[-1], xtot, rhotot)
    ncl = xrun[-1] * self.rhoatmo / self.m_ccn  # cloud particle number density [1/cm3]
    ngas = self.pres / self.temp / self.kb
    ax[0].plot(rund(self.rg_in*1e4), logp, color='k', linestyle='-.', label=r'r$_\mathrm{in}$ [$\mu$m]')
    ax[0].plot(rund(rg*1e4), logp, color='orange', linestyle='-.', label=r'r$_\mathrm{new}$ [$\mu$m]')
    ax[0].plot(rund(self.rg*1e4), logp, color='green', linestyle='-.', label=r'r$_\mathrm{out}$ [$\mu$m]')
    ax[0].vlines([-4], logp[-1], logp[0], linestyle='-.', color='gray', label=r'1 $\mu$m')

    for s, spec in enumerate(self.species):
        ax[1].plot(rund(xrun[s*2]), logp, color=cm.tab10(s/10))
        pvap = self.ds.vapor_pressures(self.species[s], self.temp, self.mh)
        vaps = (pvap * self.mw[s] / self.pres / self.mmw)
        ax[1].plot(rund(vaps), logp, color=cm.tab10(s/10), linestyle='-.')

        ax[2].plot(rund(xrun[s*2+1]), logp, color=cm.tab10(s/10), label=spec)

        n1 = xrun[s * 2] * self.rhoatmo / self.m1[s]  # gas-phase number density [1/cm3]
        acc_rate = self.acc_rate(rg, self.temp, n1, ncl, s)  # accretion rate [1/cm3/s]
        nuc_rate = self.nuc_rate(n1, self.temp, s)  # nucleation rate [1/cm3/s]
        ax[3].plot(rund(n1 / ngas), logp)
        ax[4].plot(rund(acc_rate), logp, color=cm.tab10(s/10))
        ax[5].plot(rund(nuc_rate), logp, color=cm.tab10(s/10))

    ax[2].plot(rund(xtot), logp, color='k', label='total')
    ax[3].plot(rund(ncl / ngas), logp, label='cloud', color='k', linestyle='-.')

    for a, aa in enumerate(ax):
        aa.set_ylim(logp[-1], logp[0])
        aa.legend()
        if a != 0:
            aa.tick_params(labelleft=False)
    ax[0].set_ylabel('log(p [bar])')
    ax[0].set_xlabel(r'log(r [$\mu$m])')
    ax[1].set_xlabel('log(gas mmr)')
    ax[2].set_xlabel('log(cloud mmr)')
    ax[3].set_xlabel('log(nr/ngas)')
    ax[4].set_xlabel('log(G [cm3/s])')
    ax[4].set_xlim(-10)
    ax[5].set_xlabel('log(J [cm3/s])')
    ax[5].set_xlim(-10)
    plt.subplots_adjust(wspace=0, bottom=0.15, left=0.07, right=0.98, top=0.98)
    plt.show()


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
