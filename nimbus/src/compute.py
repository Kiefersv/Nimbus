""" Main computation functions for single cloud particle species calculations """

import numpy as np
from scipy.integrate import solve_ivp
import xarray as xr
from time import time

from .plotter import plot_initial_conditions, plot_full_structure
from .solver import set_initial_condidtions, set_up_solver
from .data_storage import save_run
from .atmosphere_physics import mass_to_radius

def compute(self, itterations=10, save_file=None, tag=None, static_rg=True):

    # ==== Preparations =================================================================
    # check itterations
    if itterations < 1:
        print('[WARN] Itterations cannot be less than 1.')
        itterations = 1
    # remember number of itterations
    self.itterations = itterations

    # set rg calculation settings, True: Itterate over fixed rg, False: calcualte rg
    self.static_rg = static_rg

    # starting time to evaluate runtime
    start_time = time()

    # check if solver is setup, and do set up if necessary
    if not self.isset_solver:
        print('[WARN] Solver set up automatically. '
              'Use set_up_solver() for more control.')
        set_up_solver()
        self.isset_solver = True

    # ==== initial conditions
    yin = set_initial_condidtions(self)  # load initial conditions
    # plot initial conditions
    if self.do_plots:
        plot_initial_conditions(self, yin)

    # ==== Variables to save intermediate results
    self.rg_history = np.zeros((len(self.pres), itterations + 2))

    # ==== Main computational loop ==================================================
    # There are two ways to calculate the cloud structure:
    #   1) Keep rg constant, and itterate (fast, but less precises)
    #   2) Calculate rg on the fly (precises put computationally expansive)

    # ==== Itterate over static rg
    # This loop iterates of cloud particle size. In each loop, rg is held constant
    # and updated at the end of the loop.
    if self.static_rg:
        for t in range(itterations):
            # ==== preparations
            self.loop_nr = t+1  # loop number
            self.rg_in = self.rg # remember input radius
            ts = np.logspace(np.log10(self.tstart), np.log10(self.tend), self.tsteps)

            # ==== print progress if verbose
            if self.verbose:
                print('\r[INFO] Loop ' + str(self.loop_nr) + '/' + str(itterations)
                      + ' || Current loop progress 00.00%', end='')

            # ==== call the solver
            sol = solve_ivp(
                self.fex, [self.tstart, self.tend], yin, method='LSODA',
                rtol=self.ode_rtol, atol=self.ode_atol, t_eval=ts
            )

            # ==== prepare next run
            yin = sol.y[:, -1]  # set initial conditions to last run
            # calculate acutal radius from output
            rg = mass_to_radius(self, sol.y[self.sz * 2:, -1],
                                sol.y[self.sz:self.sz * 2, -1])
            # create a polnom fit to the cloud particle radius to prevent sudden changes
            fit = np.polyval(np.polyfit(np.log10(self.pres[self.mask_psupsat]),
                                        np.log10(rg[self.mask_psupsat]),
                                        deg=self.rg_fit_deg),
                             np.log10(self.pres))
            rg = 10 ** np.maximum(np.minimum(fit, 50), -50)  # prevent extreme values
            rg = 10 ** ((np.log10(rg) + np.log10(self.rg)) / 2)
            self.rg = np.maximum(rg, self.r_ccn)
            self.rg_history[:, t + 1] = self.rg

            # analytic cloud structure plot (a bit messy, not gonna lie)
            if self.do_plots:
                plot_full_structure(self, sol.y, str(t))

    # ==== Calculate rg on the fly
    else:
        # ==== preparations
        self.loop_nr = 1  # there is only 1 loop
        self.rg_in = self.rg  # remember input radius
        ts = np.logspace(np.log10(self.tstart), np.log10(self.tend), self.tsteps)

        # ==== call the solver
        sol = solve_ivp(
            self.fex, [self.tstart, self.tend], yin, method='LSODA',
            rtol=self.ode_rtol, atol=self.ode_atol, t_eval=ts
        )

        # ==== analytic cloud structure plot (a bit messy, not gonna lie)
        if self.do_plots:
            plot_full_structure(self, sol.y, str(0))

    # ==== save data internally
    save_run(self, sol, save_file=save_file, tag=tag)
    # print final informations
    print(f'\r[INFO] Itterations completed in {time() - start_time:.2f}s.')

    return sol.y