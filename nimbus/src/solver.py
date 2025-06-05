""" All set-up functionalities of NIMBUS """

import numpy as np

def set_initial_condidtions(self):
    # x0 = self.set_initial_condidtions()
    x0 = np.zeros(3 * self.sz) + 1e-30
    x0[:self.sz] = self.pvap * self.mw / self.pres / self.mmw
    x0[self.sz - 6:self.sz] = self.deep_gas_mmr
    x0[:self.sz] = self.deep_gas_mmr
    x0[self.sz:self.sz * 2] = 1e-30
    x0[self.sz * 2:self.sz * 3] = 1e-30

    return x0

def set_up_solver(self):
    """
    Set up the solver of Nimbus. This can only be run after atmospheric initialisation

    Parameters
    ----------
    self : Nimbus class
        Nimbus object to set up
    """

    # ==== Check if atmosphere is set up
    if not self.isset_atmosphere:
        raise ValueError("Call set_up_atmosphere before set_up_solver.")

    # ==== Solver
    def fex(t, x, src_only=False):
        """
        Function to be solved by solve_ivp

        Parameters
        ----------
        t : float
            time [s]
        x : ndarray
            Mass mixing ratios of the form:
            [xv(p1), ..., xv(pN), xc(p1), ..., xc(pN), xn(p1), ..., xn(pN)]
        src_only : bool
            If true, only the source function is evaluated.

        """
        xw = x.reshape((3, self.sz))
        # xw = np.nan_to_num(xw)
        xw[xw < 1e-30] = 1e-30
        xv = xw[0]
        xc = xw[1]
        xn = xw[2]

        dxv_src = np.zeros_like(xv)
        dxv_dif = np.zeros_like(xv)
        dxc_src = np.zeros_like(xc)
        dxc_dif = np.zeros_like(xc)
        dxc_adv = np.zeros_like(xc)
        dxn_src = np.zeros_like(xn)
        dxn_dif = np.zeros_like(xn)
        dxn_adv = np.zeros_like(xn)

        # calcualte physical parameters
        # mp = np.nan_to_num((xc + xn) * self.m_ccn / xn)
        # rg = np.cbrt(3 * mp / (4 * np.pi * self.rhop))
        ncl = xn * self.rhoatmo / self.m_ccn
        n1 = xv * self.rhoatmo / self.m1
        rg = self.rg

        acc_rate = self.acc_rate(rg, self.temp, n1, ncl)
        nuc_rate = np.nan_to_num(self.nuc_rate(n1, self.temp))
        vsed = self.vsed(rg)

        # source terms
        dxv_src = - acc_rate * self.m1 / self.rhoatmo - nuc_rate * self.m_ccn / self.rhoatmo
        dxc_src = acc_rate * self.m1 / self.rhoatmo + nuc_rate * self.m_ccn / self.rhoatmo
        dxn_src = nuc_rate * self.m_ccn / self.rhoatmo

        if not src_only:
            # diffusion
            dxv_dif[0] = self.kzz[0] * self.rhoatmo[0] * np.diff(xv[:2]) / self.dz_mid[0] / self.dz[0] / \
                         self.rhoatmo[0]
            dxv_dif[-1] = 0
            dxv_dif[1:-1] = np.diff(self.kzz_mid * self.rhoatmo_mid * np.diff(xv) / self.dz_mid) / self.dz[
                                                                                                   1:-1] / self.rhoatmo[
                                                                                                           1:-1]

            # diffusion
            dxn_dif[0] = self.kzz[0] * self.rhoatmo[0] * np.diff(xn[:2]) / self.dz_mid[0] / self.dz[0] / \
                         self.rhoatmo[0]
            dxn_dif[-1] = 0
            dxn_dif[1:-1] = np.diff(self.kzz_mid * self.rhoatmo_mid * np.diff(xn) / self.dz_mid) / self.dz[
                                                                                                   1:-1] / self.rhoatmo[
                                                                                                           1:-1]

            # diffusion
            dxc_dif[0] = self.kzz[0] * self.rhoatmo[0] * np.diff(xc[:2]) / self.dz_mid[0] / self.dz[0] / \
                         self.rhoatmo[0]
            dxc_dif[-1] = 0
            dxc_dif[1:-1] = np.diff(self.kzz_mid * self.rhoatmo_mid * np.diff(xc) / self.dz_mid) / self.dz[
                                                                                                   1:-1] / self.rhoatmo[
                                                                                                           1:-1]

            # advection
            dxn_adv[0] = self.rhoatmo[0] * xn[0] * vsed[0] / self.dz_mid[0] / self.rhoatmo[0]
            dxn_adv[-1] = 0
            dxn_adv[1:-1] = np.diff((self.rhoatmo * vsed * xn)[:-1]) / self.dz[1:-1] / self.rhoatmo[1:-1]

            # advection
            dxc_adv[0] = self.rhoatmo[0] * xc[0] * vsed[0] / self.dz_mid[0] / self.rhoatmo[0]
            dxc_adv[-1] = 0
            dxc_adv[1:-1] = np.diff((self.rhoatmo * vsed * xc)[:-1]) / self.dz[1:-1] / self.rhoatmo[1:-1]

        dxv = dxv_dif + dxv_src
        dxc = dxc_dif + dxc_src + dxc_adv
        dxn = dxn_dif + dxn_src + dxn_adv

        dxv[~self.mask_psupsat] = 0
        dxc[~self.mask_psupsat] = 0
        dxn[~self.mask_psupsat] = 0

        dx = np.hstack([dxv, dxc, dxn])
        print('--------- ', np.log10(t))
        return dx

    # ==== Jacobian
    def jac(t, x):
        """
        Function to be solved by solve_ivp

        Parameters
        ----------
        t : float
            time [s]
        x : ndarray
            Mass mixing ratios of the form:
            [xv(p1), ..., xv(pN), xc(p1), ..., xc(pN), xn(p1), ..., xn(pN)]
        """

        # define jacobian
        J = np.zeros((self.sz * 3, self.sz * 3))

        # # advection terms
        # vsed = self.vsed(self.rg)
        # J[self.sz, self.sz] = self.rhoatmo[0] * vsed[0] / self.dz_mid[0] / self.rhoatmo[0]
        # J[self.sz*2, self.sz*2] = self.rhoatmo[0] * vsed[0] / self.dz_mid[0] / self.rhoatmo[0]
        # for i in range(self.sz)[1:-1]:
        #     J[self.sz*1+i, self.sz*1+i] = self.rhoatmo[i] * vsed[i] / self.dz[i] / self.rhoatmo[i]
        #     J[self.sz*1+i, self.sz*1+i-1] = - self.rhoatmo[i-1] * vsed[i-1] / self.dz[i] / self.rhoatmo[i]
        #     J[self.sz*2+i, self.sz*2+i] = self.rhoatmo[i] * vsed[i] / self.dz[i] / self.rhoatmo[i]
        #     J[self.sz*2+i, self.sz*2+i-1] = - self.rhoatmo[i-1] * vsed[i-1] / self.dz[i] / self.rhoatmo[i]
        #
        # # diffusion terms
        # J[self.sz*0, self.sz*0] = - self.kzz[0] * self.rhoatmo[0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
        # J[self.sz*0, self.sz*0+1] = self.kzz[0] * self.rhoatmo[0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
        # J[self.sz*1, self.sz*1] = - self.kzz[0] * self.rhoatmo[0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
        # J[self.sz*1, self.sz*1+1] = self.kzz[0] * self.rhoatmo[0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
        # J[self.sz*2, self.sz*2] = - self.kzz[0] * self.rhoatmo[0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
        # J[self.sz*2, self.sz*2+1] = self.kzz[0] * self.rhoatmo[0] / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
        # for i in range(self.sz)[1:-1]:
        #     J[self.sz*0+i, self.sz*0+i-1] = self.kzz_mid[i-1] * self.rhoatmo_mid[i-1] / self.dz_mid[i-1] / self.dz[i] / self.rhoatmo[i]
        #     J[self.sz*0+i, self.sz*0+i+0] = - self.kzz_mid[i-1] * self.rhoatmo_mid[i-1] / self.dz_mid[i-1] / self.dz[i] / self.rhoatmo[i] - self.kzz_mid[i] * self.rhoatmo_mid[i] / self.dz_mid[i] / self.dz[i] / self.rhoatmo[i]
        #     J[self.sz*0+i, self.sz*0+i+1] = self.kzz_mid[i] * self.rhoatmo_mid[i] / self.dz_mid[i] / self.dz[i] / self.rhoatmo[i]
        #     J[self.sz*1+i, self.sz*1+i-1] = self.kzz_mid[i-1] * self.rhoatmo_mid[i-1] / self.dz_mid[i-1] / self.dz[i] / self.rhoatmo[i]
        #     J[self.sz*1+i, self.sz*1+i+0] = - self.kzz_mid[i-1] * self.rhoatmo_mid[i-1] / self.dz_mid[i-1] / self.dz[i] / self.rhoatmo[i] - self.kzz_mid[i] * self.rhoatmo_mid[i] / self.dz_mid[i] / self.dz[i] / self.rhoatmo[i]
        #     J[self.sz*1+i, self.sz*1+i+1] = self.kzz_mid[i] * self.rhoatmo_mid[i] / self.dz_mid[i] / self.dz[i] / self.rhoatmo[i]
        #     J[self.sz*2+i, self.sz*2+i-1] = self.kzz_mid[i-1] * self.rhoatmo_mid[i-1] / self.dz_mid[i-1] / self.dz[i] / self.rhoatmo[i]
        #     J[self.sz*2+i, self.sz*2+i+0] = - self.kzz_mid[i-1] * self.rhoatmo_mid[i-1] / self.dz_mid[i-1] / self.dz[i] / self.rhoatmo[i] - self.kzz_mid[i] * self.rhoatmo_mid[i] / self.dz_mid[i] / self.dz[i] / self.rhoatmo[i]
        #     J[self.sz*2+i, self.sz*2+i+1] = self.kzz_mid[i] * self.rhoatmo_mid[i] / self.dz_mid[i] / self.dz[i] / self.rhoatmo[i]

        # source terms:
        dx = np.maximum(1e-15, np.abs(x) * 1e-3)
        y0 = fex(t, x)  # , src_only=True)
        for i in range(self.sz * 3):
            x0 = x.copy()
            x0[i] += dx[i]
            y1 = fex(t, x0)  # , src_only=True)
            J[:, i] += (y1 - y0) / dx[i]

        return J

    # ==== Set the functions
    self.fex = fex
    self.jac = jac
    self.isset_solver = True
