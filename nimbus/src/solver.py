""" All set-up functionalities of NIMBUS """
# pylint: disable=C0301

import numpy as np
from time import time
from .atmosphere_physics import mass_to_radius

def set_initial_condidtions(self):
    """
    The current initial conditions assume no cloud particles in the cloud layers and a
    saturaded vapour. Below the cloud layer, the deep MMR is assumed. This has proven to
    be a generally accaptable choice. However, improvments could be made here.
    """
    # ==== Initialise array, and set all values to the minimum value
    x0 = np.zeros((self.nspec*2 + 1, self.sz)) + self.ode_minimum_mmr

    # ==== loop over the materials
    for s, deep in enumerate(self.deep_gas_mmr):
        # if self.deep_gas_mmr[s] > 0:
        # calculate vapour mmr
        pvap = self.db.vapor_pressures(self.species[s], self.temp, self.mh)
        x0[s*2] = np.minimum(pvap * self.mw[s] / self.pres / self.mmw, deep)
        # # assign deep mmr
        x0[s*2, ~self.mask_psupsat] = deep

    # ==== set flag to true
    self.isset_initialisation = True

    # ==== return initial conditions
    return x0.flatten()

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
            [xv1(p1), ..., xv1(pN), xc1(p1), ..., xc1(pN),
             xv2(p1), ..., xv2(pN), xc2(p1), ..., xc2(pN),
             xn(p1), ..., xn(pN)]

        Return
        ------
        dfdt : ndarray
            change of MMRs with time [g/cm3/s]
        """

        # ==== Read in the input ========================================================
        xw = x.reshape((self.nspec*2 + 1, self.sz))  # reshape array
        # prevent underflow of values
        xw[xw < self.ode_minimum_mmr] = self.ode_minimum_mmr
        dx = np.zeros((self.nspec*2 + 1, self.sz))
        # total mass
        xtot = np.sum(xw[1::2], axis=0)
        rhotot = np.sum(xw[1::2]*self.rhop[:, np.newaxis], axis=0)/xtot

        # ==== Check timeout condition
        if self.timeout is not None:
            if time() - self.start_time > self.timeout:
                self.complete = False
                return dx.flatten()

        # ==== calcualte physical parameters ============================================
        if self.static_rg:  # use static rg
            rg = self.rg  # cloud particle radius [cm]
        else: # calculate rg on the fly
            rg = mass_to_radius(self, xw[-1], xtot, rhotot)  # cloud particle radius [cm]
            self.rg = rg  # safe rg for outside of function
        ncl = xw[-1] * self.rhoatmo / self.m_ccn  # cloud particle number density [1/cm3]
        vsed = self.vsed(rg, rhotot)  # settling velocity [cm/s]
        self.calc_atmos_struct()   # Update atmosphere

        # ==== Rate calculations
        for s, _ in enumerate(self.species):
            n1 = xw[s*2] * self.rhoatmo / self.m1[s]  # gas-phase number density [1/cm3]
            acc_rate = self.acc_rate(rg, self.temp, n1, ncl, s)  # accretion rate [1/cm3/s]
            nuc_rate = self.nuc_rate(n1, self.temp, s)  # nucleation rate [1/cm3/s]

            # ==== source terms =============================================================
            dx[s*2] += - acc_rate * self.m1[s] / self.rhoatmo - nuc_rate * self.m_ccn / self.rhoatmo
            dx[s*2+1] += acc_rate * self.m1[s] / self.rhoatmo + nuc_rate * self.m_ccn / self.rhoatmo
            dx[-1] += nuc_rate * self.m_ccn / self.rhoatmo

        # ==== Diffusion terms ==========================================================
        # !!! Note: Rounding errors prevents the definition of prefactors !!!
        for s in range(self.nspec*2 + 1):
            dx[s, 0] += self.kzz[0] * self.rhoatmo[0] * np.diff(xw[s, :2])[0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
            dx[s, -1] += 0
            dx[s, 1:-1] += np.diff(self.kzz_mid * self.rhoatmo_mid * np.diff(xw[s]) / self.dz_mid) / self.dz[1:-1] / self.rhoatmo[1:-1]

        # ==== Advection terms ==========================================================
        # !!! Note: Rounding errors prevents the definition of prefactors !!!
        for s, _ in enumerate(self.species):
            dx[s * 2 + 1, 0] += self.rhoatmo[0] * xw[s * 2 + 1, 0] * vsed[0] / self.dz_mid[0] / self.rhoatmo[0]
            dx[s * 2 + 1, 1:-1] += np.diff((self.rhoatmo * vsed * xw[s * 2 + 1])[:-1]) / self.dz[1:-1] / self.rhoatmo[1:-1]
        dx[-1, 0] += self.rhoatmo[0] * xw[-1, 0] * vsed[0] / self.dz_mid[0] / self.rhoatmo[0]
        dx[-1, 1:-1] += np.diff((self.rhoatmo * vsed * xw[-1])[:-1]) / self.dz[1:-1] / self.rhoatmo[1:-1]

        # ===== additional top of atmosphere influx ====================================
        if self.tf is not None:
            dx += self.tf(self.pres, self.temp, t)

        # ==== Finalsing output =========================================================
        # set all values below the vapour pressure to zero (speeds up calculation)
        dx[:, ~self.mask_psupsat] = 0
        # print progress information
        if self.verbose and not self.mute:
            prog = np.log10(t)/np.log10(self.tend) * 100
            print('\r[INFO] Loop ' + str(self.loop_nr) + '' + self.it_str
                  + ' || Current loop progress ' + f"{prog:05.2f}%", end='')

        # ==== Return time derivative
        return dx.flatten()

    # ==== Set the functions
    self.fex = fex
    self.isset_solver = True
    if not self.mute:
        print('[INFO] Solver set up.')
