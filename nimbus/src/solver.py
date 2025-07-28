""" All set-up functionalities of NIMBUS """

import numpy as np
from scipy.optimize import root_scalar

from .atmosphere_physics import (mass_to_radius, get_rhop,
                                 get_settling_velocity_function)

def set_initial_condidtions(self):
    """
    The current initial conditions assume no cloud particles in the cloud layers and a
    saturaded vapour. Below the cloud layer, the deep MMR is assumed. This has proven to
    be a generally accaptable choice. However, improvments could be made here.
    """
    # set all values to mimimum of ode solver
    x0 = np.zeros(3 * self.sz) + self.ode_minimum_mmr
    # calculate vapour mmr
    x0[:self.sz] = self.pvap * self.mw / self.pres / self.mmw
    # assighn deep mmr
    x0[:self.sz][~self.mask_psupsat] = self.deep_gas_mmr

    return x0

def set_up_solver(self):
    """
    Set up the solver of Nimbus. This can only be run after atmospheric initialisation.

    Parameters
    ----------
    self : Nimbus class
        Nimbus object to set up
    """

    # ==== Check if atmosphere is set up
    if not self.isset_atmosphere:
        raise ValueError("Call set_up_atmosphere before set_up_solver.")

    # ==== set initial conditions of solver
    self.x0 = np.zeros((len(self.species), self.sz))
    self.x0 += np.asarray(self.initial_mmrs)[:, np.newaxis]

    # # ==== find pressure levels which are always supersaturated
    # ndeep = self.initial_mmrs * self.rhoatmo / self.m1  # deep particle number density
    # pdeep = ndeep * self.kb * self.temp  # deep partial pressure
    # self.mask_psupsat = self.pvap / pdeep < 1  # mask where vapour can condense

    # ==== define settling velocity function
    self.vsed = get_settling_velocity_function(self)

    # ==== prepare nucleation rates by adding indexes
    for nreac in self.nuc_reacs:
        idx = self.species.index(self.nuc_reacs[nreac]['i'])
        self.nuc_reacs[nreac]['idx'] = idx

    # ==== prepare accreation reates by adding indexes and mw
    for areac in self.acc_reacs:
        # index of cloud particle material
        self.acc_reacs[areac]['idc'] = self.species.index(self.acc_reacs[areac]['c'])
        self.acc_reacs[areac]['idr'] = []  # indexes of reactants
        self.acc_reacs[areac]['idp'] = []  # indexes of products
        mask = np.zeros(len(self.species))  # mask for depletion of species
        for re in self.acc_reacs[areac]['i']:
            self.acc_reacs[areac]['idr'].append(self.species.index(re))
            mask[self.species.index(re)] -= 1
        for re in self.acc_reacs[areac]['o']:
            self.acc_reacs[areac]['idp'].append(self.species.index(re))
            mask[self.species.index(re)] += 1

        mask[self.acc_reacs[areac]['idc']] += 1  # also add cloud material
        self.acc_reacs[areac]['mask'] = mask  # save mask

    # ==== Calculate initial radius
    self.rg = np.zeros_like(self.pres)
    for i, _ in enumerate(self.pres):
        # minimisation function
        def vsed_f(rg):
            rhop = 3  # assume a generic silicate for initial conditions
            v_c = self.vsed(rg, 3, self.temp)[i]  # settling veloctity
            vk = self.fsed * self.kzz[i] / self.h[i]  # fsed velocity
            return vk - v_c
        # call of minimisation function with optimised initial condiaitons
        self.rg[i] = np.maximum(root_scalar(vsed_f, x0=self.r_ccn).root, self.r_ccn)

    # ==== Solver
    def fex(t, x):
        """
        Function to be solved by solve_ivp

        Parameters
        ----------
        t : float
            time [s]
        x : ndarray
            Mass mixing ratios of the form:
            [xv(p1), ..., xv(pN), xc(p1), ..., xc(pN), xn(p1), ..., xn(pN)]

        Return
        ------
        dfdt : ndarray
            change of MMRs with time [g/cm3/s]
        """

        # ==== Read in the input ========================================================
        xw = x.reshape((len(self.species), self.sz))  # reshape array
        # prevent underflow of values
        xw[xw < self.ode_minimum_mmr] = self.ode_minimum_mmr
        # default arrays
        d_dif = np.zeros_like(xw)  # diffusion terms
        d_adv = np.zeros_like(xw)  # advection terms
        d_nuc = np.zeros_like(xw)  # nucleation terms
        d_acc = np.zeros_like(xw)  # nucleation terms

        # ==== Physical valuse ==========================================================
        ncl = xw[0] * self.rhoatmo / self.m_ccn  # cloud particle number density [1/cm3]
        xc = np.sum(x[self.idl_clmat], axis=0)  # total cloud mass mixing ratio [g/g]
        rhop = get_rhop(self, xw)  # mixed cloud particle dnesity [g/cm3]
        if self.static_rg:  # use static rg
            rg = self.rg  # cloud particle radius [cm]
        else: # calculate rg on the fly
            rg = mass_to_radius(self, xw[0], xc, rhop)  # cloud particle radius [cm]
            self.rg = rg
        vsed = self.vsed(rg, rhop, self.temp)  # settling velocity [cm/s]

        # ==== Diffusion terms ==========================================================
        # !!! Note: Rounding errors prevents the definition of prefactors !!!
        d_dif[:, 0] = self.kzz[0] * self.rhoatmo[0] * np.diff(xw[:, :2], axis=1)[:, 0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
        d_dif[:, -1] = 0
        d_dif[:, 1:-1] = np.diff(self.kzz_mid * self.rhoatmo_mid * np.diff(xw, axis=1) / self.dz_mid, axis=1) / self.dz[1:-1] / self.rhoatmo[1:-1]

        # ==== Advection terms ==========================================================
        # !!! Note: Rounding errors prevents the definition of prefactors !!!
        d_adv[self.idl_clmat, 0] = self.rhoatmo[0] * xw[self.idl_clmat, 0] * vsed[0] / self.dz_mid[0] / self.rhoatmo[0]
        d_adv[self.idl_clmat, -1] = 0
        d_adv[self.idl_clmat, 1:-1] = np.diff((self.rhoatmo * vsed * xw)[self.idl_clmat, :-1], axis=1) / self.dz[1:-1] / self.rhoatmo[1:-1]

        # ==== Nucleation terms =========================================================
        for nreac in self.nuc_reacs:
            idx = self.nuc_reacs[nreac]['idx']  # index of nucleating species
            # number density of nucelating species [1/cm3]
            n1 = xw[idx] * self.rhoatmo / self.nuc_reacs[nreac]['mw'] * self.avog
            nuc_rate = self.nuc_reacs[nreac]['k'](n1, self.temp)  # [1/cm3/s]
            d_nuc[0] = nuc_rate * self.m_ccn / self.rhoatmo  # add new cloud particles
            d_nuc[idx] = - nuc_rate * self.m_ccn / self.rhoatmo  # remove gas phase

        # ==== Accretion terms ==========================================================
        for areac in self.acc_reacs:
            # === Prepare calcualtions
            idc = self.acc_reacs[areac]['idc']  # cloud particle index
            nkey = np.ones_like(self.temp) * np.inf  # key species (lowest number density)
            nfac = 1  # supersaturation factor from number densities
            m1 = self.acc_reacs[areac]['mwc'] / self.avog
            pvap = self.acc_reacs[areac]['pvap']
            mask = self.acc_reacs[areac]['mask']
            # ==== Find number densities
            for i, idr in enumerate(self.acc_reacs[areac]['idr']):
                # find number density of each reactant [1/cm3]
                nr = xw[idr] * self.rhoatmo / self.acc_reacs[areac]['mwr'][i] * self.avog
                nkey[nr < nkey] = nr[nr < nkey]  # find smallest reactant
                nfac *= nr
            for i, idp in enumerate(self.acc_reacs[areac]['idp']):
                nr = xw[idp] * self.rhoatmo / self.acc_reacs[areac]['mwp'][i] * self.avog
                nfac /= nr
            # ==== calcualte accretion rate
            pfac = nfac * self.kb * self.temp  # number density to partial pressure
            acc_pref = self.acc_reacs[areac]['k'](rg, self.temp)  # Accreation prefactor
            acc_rate = acc_pref * nkey * ncl * (1 - pvap(self.temp) / pfac)  # Accreation rate
            # ==== Deplete accordingly
            d_acc = mask[:, np.newaxis] * acc_rate[np.newaxis, :] * m1 / self.rhoatmo

        # ==== Finalsing output =========================================================
        # combine all parts of the ode
        dx = d_dif + d_adv + d_nuc + d_acc
        # # set all values below the vapour pressure to zero (speeds up calculation)
        # dx[~self.mask_psupsat] = 0
        # print progress information
        if self.verbose:
            prog = np.log10(t)/np.log10(self.tend) * 100
            print('\r[INFO] Loop ' + str(self.loop_nr) + '' + self.it_str
                  + ' || Current loop progress ' + f"{prog:05.2f}%", end='')

        # ==== Return time derivative
        return dx.flatten()

    # ==== Set the functions
    self.fex = fex
    self.isset_solver = True
    print('[INFO] Solver set up.')
