""" All set-up functionalities of NIMBUS """

import numpy as np

from .atmosphere_physics import mass_to_radius

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
    self.x0 = self.initial_mmrs[:, np.newaxis]

    # ==== find pressure levels which are always supersaturated
    ndeep = self.initial_mmrs * self.rhoatmo / self.m1  # deep particle number density
    pdeep = ndeep * self.kb * self.temp  # deep partial pressure
    self.mask_psupsat = self.pvap / pdeep < 1  # mask where vapour can condense

    # ==== prepare nucleation rates by adding indexes
    for nreac in self.nuc_reacs:
        idx = self.species.index(self.nuc_reacs[nreac]['i'])
        self.nuc_reacs[nreac]['idx'] = idx

    # ==== prepare accreation reates by adding indexes and mw
    for areac in self.acc_reacs:
        self.acc_reacs[areac]['idr'] = []  # indexes of reactants
        self.acc_reacs[areac]['idp'] = []  # indexes of products
        for re in self.acc_reacs[areac]['i']:
            self.acc_reacs[areac]['idr'].append(self.species.index(re))
        for re in self.acc_reacs[areac]['i']:
            self.acc_reacs[areac]['idp'].append(self.species.index(re))

    # ==== Calculate initial radius
    self.rg = np.zeros_like(self.pres)
    for i, _ in enumerate(self.pres):
        # minimisation function
        def vsed_f(rg):
            v_c = self.vsed(rg)[i]  # settling veloctity
            vk = self.fsed * self.kzz[i] / self.h[i]  # fsed velocity
            return vk - v_c
        # call of minimisation function with optimised initial condiaitons
        self.rg[i] = np.maximum(root_scalar(vsed_f, x0=self.r1 * 1e2).root, self.r_ccn)

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

        # ==== Physical valuse ==========================================================
        ncl = xw[0] * self.rhoatmo / self.m_ccn  # cloud particle number density [1/cm3]
        xc = np.sum(x[self.idl_clmat], axis=0)  # total cloud mass mixing ratio [g/g]
        if self.static_rg:  # use static rg
            rg = self.rg  # cloud particle radius [cm]
        else: # calculate rg on the fly
            rg = mass_to_radius(self, xw[0], xc)  # cloud particle radius [cm]
            self.rg = rg
        vsed = self.vsed(rg)  # settling velocity [cm/s]

        # ==== Diffusion terms ==========================================================
        # !!! Note: Rounding errors prevents the definition of prefactors !!!
        d_dif[:, 0] = self.kzz[0] * self.rhoatmo[0] * np.diff(xw[:, 2], axis=1) / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
        d_dif[:, -1] = 0
        d_dif[:, 1:-1] = np.diff(self.kzz_mid * self.rhoatmo_mid * np.diff(xw, axis=1) / self.dz_mid, axis=1) / self.dz[1:-1] / self.rhoatmo[1:-1]

        # ==== Advection terms ==========================================================
        # !!! Note: Rounding errors prevents the definition of prefactors !!!
        d_adv[self.idl_clmat, 0] = self.rhoatmo[0] * xc[self.idl_clmat, 0] * vsed[0] / self.dz_mid[0] / self.rhoatmo[0]
        d_adv[self.idl_clmat, -1] = 0
        d_adv[self.idl_clmat, 1:-1] = np.diff((self.rhoatmo * vsed * xc)[self.idl_clmat, :-1], axis=1) / self.dz[1:-1] / self.rhoatmo[1:-1]

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
            n_key = np.inf  # key species (lowest number density)
            n_fac = 1  # supersaturation factor from number densities
            for i, idr in enumerate(self.acc_reacs[areac]['idr']):
                # find number density of each reactant [1/cm3]
                nr = xw[idr] * self.rhoatmo / self.nuc_reacs[nreac]['mwr'][i] * self.avog
                if nr < n_key: n_key = nr  # find smallest reactant
                n_fac *= nr
            for i, idp in enumerate(self.acc_reacs[areac]['idp']):
                nr = xw[idp] * self.rhoatmo / self.nuc_reacs[nreac]['mwp'][i] * self.avog
                n_fac /= nr

        # ==== source terms =============================================================
        dxv_src = - acc_rate * self.m1 / self.rhoatmo - nuc_rate * self.m_ccn / self.rhoatmo
        dxc_src = acc_rate * self.m1 / self.rhoatmo + nuc_rate * self.m_ccn / self.rhoatmo
        dxn_src = nuc_rate * self.m_ccn / self.rhoatmo

        # ==== Finalsing output =========================================================
        # combine all parts of the ode
        dx = d_dif + d_adv
        # set all values below the vapour pressure to zero (speeds up calculation)
        dx[~self.mask_psupsat] = 0
        # print progress information
        if self.verbose:
            prog = np.log10(t)/np.log10(self.tend) * 100
            print('\r[INFO] Loop ' + str(self.loop_nr) + '' + self.it_str
                  + ' || Current loop progress ' + f"{prog:05.2f}%", end='')

        # ==== Return time derivative
        return dx

    # # ==== Jacobian
    # def jac(t, x):
    #     """
    #     Function to be solved by solve_ivp
    #
    #     Parameters
    #     ----------
    #     t : float
    #         time [s]
    #     x : ndarray
    #         Mass mixing ratios of the form:
    #         [xv(p1), ..., xv(pN), xc(p1), ..., xc(pN), xn(p1), ..., xn(pN)]
    #     """
    #
    #     # define jacobian
    #     J = np.zeros((self.sz * 3, self.sz * 3))
    #
    #     # # advection terms
    #     # vsed = self.vsed(self.rg)
    #     # J[self.sz, self.sz] = self.rhoatmo[0] * vsed[0] / self.dz_mid[0] / self.rhoatmo[0]
    #     # J[self.sz*2, self.sz*2] = self.rhoatmo[0] * vsed[0] / self.dz_mid[0] / self.rhoatmo[0]
    #     # for i in range(self.sz)[1:-1]:
    #     #     J[self.sz*1+i, self.sz*1+i] = self.rhoatmo[i] * vsed[i] / self.dz[i] / self.rhoatmo[i]
    #     #     J[self.sz*1+i, self.sz*1+i-1] = - self.rhoatmo[i-1] * vsed[i-1] / self.dz[i] / self.rhoatmo[i]
    #     #     J[self.sz*2+i, self.sz*2+i] = self.rhoatmo[i] * vsed[i] / self.dz[i] / self.rhoatmo[i]
    #     #     J[self.sz*2+i, self.sz*2+i-1] = - self.rhoatmo[i-1] * vsed[i-1] / self.dz[i] / self.rhoatmo[i]
    #     #
    #     # # diffusion terms
    #     # J[self.sz*0, self.sz*0] = - self.kzz[0] * self.rhoatmo[0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
    #     # J[self.sz*0, self.sz*0+1] = self.kzz[0] * self.rhoatmo[0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
    #     # J[self.sz*1, self.sz*1] = - self.kzz[0] * self.rhoatmo[0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
    #     # J[self.sz*1, self.sz*1+1] = self.kzz[0] * self.rhoatmo[0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
    #     # J[self.sz*2, self.sz*2] = - self.kzz[0] * self.rhoatmo[0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
    #     # J[self.sz*2, self.sz*2+1] = self.kzz[0] * self.rhoatmo[0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
    #     # for i in range(self.sz)[1:-1]:
    #     #     J[self.sz*0+i, self.sz*0+i-1] = self.kzz_mid[i-1] * self.rhoatmo_mid[i-1] / self.dz_mid[i-1] / self.dz[i] / self.rhoatmo[i]
    #     #     J[self.sz*0+i, self.sz*0+i+0] = - self.kzz_mid[i-1] * self.rhoatmo_mid[i-1] / self.dz_mid[i-1] / self.dz[i] / self.rhoatmo[i] - self.kzz_mid[i] * self.rhoatmo_mid[i] / self.dz_mid[i] / self.dz[i] / self.rhoatmo[i]
    #     #     J[self.sz*0+i, self.sz*0+i+1] = self.kzz_mid[i] * self.rhoatmo_mid[i] / self.dz_mid[i] / self.dz[i] / self.rhoatmo[i]
    #     #     J[self.sz*1+i, self.sz*1+i-1] = self.kzz_mid[i-1] * self.rhoatmo_mid[i-1] / self.dz_mid[i-1] / self.dz[i] / self.rhoatmo[i]
    #     #     J[self.sz*1+i, self.sz*1+i+0] = - self.kzz_mid[i-1] * self.rhoatmo_mid[i-1] / self.dz_mid[i-1] / self.dz[i] / self.rhoatmo[i] - self.kzz_mid[i] * self.rhoatmo_mid[i] / self.dz_mid[i] / self.dz[i] / self.rhoatmo[i]
    #     #     J[self.sz*1+i, self.sz*1+i+1] = self.kzz_mid[i] * self.rhoatmo_mid[i] / self.dz_mid[i] / self.dz[i] / self.rhoatmo[i]
    #     #     J[self.sz*2+i, self.sz*2+i-1] = self.kzz_mid[i-1] * self.rhoatmo_mid[i-1] / self.dz_mid[i-1] / self.dz[i] / self.rhoatmo[i]
    #     #     J[self.sz*2+i, self.sz*2+i+0] = - self.kzz_mid[i-1] * self.rhoatmo_mid[i-1] / self.dz_mid[i-1] / self.dz[i] / self.rhoatmo[i] - self.kzz_mid[i] * self.rhoatmo_mid[i] / self.dz_mid[i] / self.dz[i] / self.rhoatmo[i]
    #     #     J[self.sz*2+i, self.sz*2+i+1] = self.kzz_mid[i] * self.rhoatmo_mid[i] / self.dz_mid[i] / self.dz[i] / self.rhoatmo[i]
    #
    #     # # source terms:
    #     # dx = np.maximum(1e-15, np.abs(x) * 1e-3)
    #     # y0 = fex(t, x)  # , src_only=True)
    #     # for i in range(self.sz * 3):
    #     #     x0 = x.copy()
    #     #     x0[i] += dx[i]
    #     #     y1 = fex(t, x0)  # , src_only=True)
    #     #     J[:, i] += (y1 - y0) / dx[i]
    #
    #     return J

    # ==== Set the functions
    self.fex = fex
    self.isset_solver = True
    print('[INFO] Solver set up.')
