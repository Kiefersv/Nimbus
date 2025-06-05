import numpy as np
from scipy.integrate import solve_ivp
import xarray as xr

from .plotter import plot_initial_conditions, plot_full_structure, plot_spectrum
from .solver import set_initial_condidtions

def compute(self, output_stepping=10, save_file=None):
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
    yin = x0  # set initial condition for first loop
    for t, tit in enumerate(np.linspace(0, output_stepping, output_stepping + 1)):
        self.rg_in = self.rg
        ts = np.logspace(np.log10(self.tstart), np.log10(self.tend), self.tsteps)
        sol = solve_ivp(
            self.fex, [self.tstart, self.tend], yin, method='LSODA',
            rtol=self.ode_rtol, atol=self.ode_atol, t_eval=ts  # , jac=self.jac  !! jacobin does not lead to improvment
        )
        print(sol)
        yin = sol.y[:, -1]

        # ==== update cloud particle radius
        mp = np.nan_to_num(sol.y[self.sz:self.sz * 2, -1] * self.m_ccn / sol.y[self.sz * 2:, -1])
        rg = np.cbrt(3 * mp / (4 * np.pi * self.rhop))
        rg = np.maximum(rg, self.r_ccn)
        # create a polnom fit to the cloud particle radius to prevent sudden changes
        fit = np.polyval(np.polyfit(np.log10(self.pres[self.mask_psupsat]), np.log10(rg[self.mask_psupsat]), deg=3),
                         np.log10(self.pres))
        rg = 10 ** fit
        rg = 10 ** ((np.log10(rg) + np.log10(self.rg)) / 2)
        self.rg = np.maximum(rg, self.r_ccn)
        self.rg_history[:, t + 1] = self.rg

        # plot_full_structure(self, sol.y, str(tit))

    # save data if a save file is given
    if not isinstance(save_file, type(None)):
        ds = xr.Dataset(
            data_vars={
                'qv': (['pressure'], sol.y[:self.sz, -1]),
                'qc': (['pressure'], sol.y[self.sz:self.sz * 2, -1]),
                'qn': (['pressure'], sol.y[self.sz * 2:self.sz * 3, -1]),
                'rg': (['pressure'], self.rg),
            },
            coords={
                'pressure': self.pres * 1e-6,
            },
            attrs={
                'mmw': self.mmw,
                'y': sol.y[:, -1],
            },
        )
        ds.to_netcdf(save_file + '.nc')

    return sol.y