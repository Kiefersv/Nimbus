import numpy as np
from scipy.integrate import solve_ivp
import xarray as xr

from .plotter import plot_initial_conditions, plot_full_structure
from .solver import set_initial_condidtions
from .data_storage import save_run

def compute_ddw(self, output_stepping=10, save_file=None, tag=None):
    def fex_chem(t, x):
        xw = x.reshape((3, self.sz))
        # xw = np.nan_to_num(xw)
        xw[xw < 1e-30] = 1e-30
        xv = xw[0]
        xc = xw[1]
        xn = xw[2]

        # calcualte physical parameters
        # rg = self.rg
        mp = np.nan_to_num(xc * self.m_ccn / xn)
        rg = np.cbrt(3 * mp / (4 * np.pi * self.rhop))
        rg = np.maximum(rg, self.r_ccn)
        self.rg = rg
        ncl = xn * self.rhoatmo / self.m_ccn
        n1 = xv * self.rhoatmo / self.m1

        acc_rate = self.acc_rate(rg, self.temp, n1, ncl)
        nuc_rate = np.nan_to_num(self.nuc_rate(n1, self.temp))
        vsed = self.vsed(rg)

        # source terms
        dxv_src = - acc_rate * self.m1 / self.rhoatmo - nuc_rate * self.m_ccn / self.rhoatmo
        dxc_src = acc_rate * self.m1 / self.rhoatmo + nuc_rate * self.m_ccn / self.rhoatmo
        dxn_src = nuc_rate * self.m_ccn / self.rhoatmo

        dxv_src[~self.mask_psupsat] = 0
        dxc_src[~self.mask_psupsat] = 0
        dxn_src[~self.mask_psupsat] = 0

        dx = np.hstack([dxv_src, dxc_src, dxn_src])
        return dx

    def fex_dyn(t, x):
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
        # rg = self.rg
        mp = np.nan_to_num(xc * self.m_ccn / xn)
        rg = np.cbrt(3 * mp / (4 * np.pi * self.rhop))
        rg = np.maximum(rg, self.r_ccn)
        self.rg = rg
        vsed = self.vsed(rg)

        # diffusion
        dxv_dif[0] = self.kzz[0] * self.rhoatmo[0] * np.diff(xv[:2]) / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
        dxv_dif[-1] = 0
        dxv_dif[1:-1] = np.diff(self.kzz_mid * self.rhoatmo_mid * np.diff(xv) / self.dz_mid) / self.dz[
                                                                                               1:-1] / self.rhoatmo[
                                                                                                       1:-1]

        # diffusion
        dxn_dif[0] = self.kzz[0] * self.rhoatmo[0] * np.diff(xn[:2]) / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
        dxn_dif[-1] = 0
        dxn_dif[1:-1] = np.diff(self.kzz_mid * self.rhoatmo_mid * np.diff(xn) / self.dz_mid) / self.dz[
                                                                                               1:-1] / self.rhoatmo[
                                                                                                       1:-1]

        # diffusion
        dxc_dif[0] = self.kzz[0] * self.rhoatmo[0] * np.diff(xc[:2]) / self.dz_mid[0] / self.dz[0] / self.rhoatmo[0]
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

        dxv = dxv_dif
        dxc = dxc_dif + dxc_adv
        dxn = dxn_dif + dxn_adv

        dxv[~self.mask_psupsat] = 0
        dxc[~self.mask_psupsat] = 0
        dxn[~self.mask_psupsat] = 0

        dx = np.hstack([dxv, dxc, dxn])
        return dx

    # ==== initial conditions
    x0 = set_initial_condidtions(self)  # load initial conditions
    # plot initial conditions
    if self.do_plots:
        plot_initial_conditions(self, x0)

    # ==== Variables to save intermediate results
    self.rg_history = np.zeros((len(self.pres), output_stepping + 2))

    # ==== Main computational loop ==================================================
    # This loop iterates of cloud particle size. In each loop, rg is held constant
    # and updated at the end of the loop.
    yw = x0  # set initial condition for first loop

    dt = 20000
    tsteps = np.logspace(np.log10(self.tstart), np.log10(self.tend), self.tsteps)
    tsteps = np.logspace(-4, 20, )
    t = 0
    # ==== loop over time steps
    while True:

        # ==== Solve the chemistry
        sol = solve_ivp(
            fex_chem, [0, dt], yw, method='LSODA',
            rtol=self.ode_rtol, atol=self.ode_atol
        )
        yw = sol.y[:, -1]


        # ==== Solve the dynamics
        sol = solve_ivp(
            fex_dyn, [0, dt], yw, method='LSODA',
            rtol=self.ode_rtol, atol=self.ode_atol
        )
        yw = sol.y[:, -1]

        if t%2000000 == 0:
            plot_full_structure(self, sol.y, str(t))

        t += dt
        print(t)

    # ==== save data internally
    save_run(self, sol, save_file=save_file, tag=tag)

    return sol.y
