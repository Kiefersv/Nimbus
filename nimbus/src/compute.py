""" Main computation functions for single cloud particle species calculations """

import numpy as np
from scipy.integrate import solve_ivp
import xarray as xr
from time import time

from .plotter import plot_initial_conditions, plot_full_structure
from .solver import set_initial_condidtions, set_up_solver
from .data_storage import save_run
from .atmosphere_physics import mass_to_radius, get_rhop

def compute(self, type='convergence', rel_dif_in_mmr=1e-3, max_itterations=None,
            save_file=None, tag=None):
    """
    Compute the cloud structure.

    Parameters
    ----------
    type : str, optional
        This parameter determines the stopping creterion. Options are:
            - 'convergence': run itteratively until convergence (see rel_dif_in_mmr)
            - 'iterate': use a fixed number of itterations (see itterations)
            - 'full': fully time dependent simulation with variable radius
    rel_dif_in_mmr : float, optional
        Convergence criterion given as maximum change in the relative MMR between
        itterations. Only used if type = 'iterate'.
    max_itterations : int, optional
        Number of itterations conducted; each itteration keeps the cloud particle radius
        constant. If type is 'iterate', this variable gives the number of itterations
        conducted (default 10). If type is 'convergence', this gives an additional
        stopping creterion in case of non-convergence (default 50).
    save_file : str, optional
        Save the results under the given name as xarray dataset.
    tag : str, optional
        Save the run internally under the given tag.

    Return
    ------
    ds : xarray.Dataset
        Returns the result as an xarray dataset
    """

    # ==== Preparations =================================================================
    # print info
    print('\r[INFO] Computation started ...', end='')

    # ==== set up settings specific for the evaluation type
    self.it_str = ''
    if type == 'convergence':
        # settings if a convergence createrion is used
        self.static_rg = True  # keep radius constant in each itteration
        if max_itterations is None:
            max_itterations = 50  # default number of max_itterations
            print('[INFO] Max itterations set to 50')
    elif type == 'iterate':
        self.static_rg = True  # keep radius constant in each itteration
        if max_itterations is None:
            max_itterations = 10  # default number of itterations
            print('[INFO] Number of itterations set to 10')
        self.it_str = '/' + str(max_itterations)
    elif type == 'full':
        self.static_rg = False  # allow for a variable radius
        max_itterations = 1  # this value is not used
    else:
        raise ValueError("[ERROR] Compute type unkown. Please select one of the "
                         "following: 'convergence', 'itterate', 'full'.")

    # additionally check itterations
    if max_itterations < 1:
        print('[WARN] Itterations cannot be less than 1.')
        max_itterations = 1
        self.it_str = str(max_itterations)
    # remember number of itterations
    self.itterations = max_itterations

    # starting time to evaluate runtime
    start_time = time()

    # check if solver is setup, and do set up if necessary
    if not self.isset_solver:
        print('[WARN] Solver set up automatically. '
              'Use set_up_solver() for more control.')
        set_up_solver()
        self.isset_solver = True

    # initial conditions
    # yin = set_initial_condidtions(self)  # load initial conditions
    yin = self.x0.flatten()
    # plot initial conditions
    if self.do_plots:
        plot_initial_conditions(self, yin)

    # Variables to save intermediate results
    self.rg_history = np.zeros((len(self.pres), self.itterations + 2))
    self.all_runs = []
    # remember the number of itterations
    t = 1
    # warning flags to print warnings only once
    deg_warn_flag = True

    # ==== Main computational loop ==================================================
    # There are two ways to calculate the cloud structure:
    #   1) Keep rg constant, and itterate (convergence, itterate)
    #   2) Calculate rg on the fly (full)

    # ==== Itterate over static rg
    # This loop iterates of cloud particle size. In each loop, rg is held constant
    # and updated at the end of the loop.
    if self.static_rg:
        while True:
            # ==== preparations
            self.loop_nr = t  # loop number
            self.rg_in = self.rg # remember input radius
            ts = np.logspace(np.log10(self.tstart), np.log10(self.tend), self.tsteps)

            # ==== print progress if verbose
            if self.verbose:
                print('\r[INFO] Loop ' + str(self.loop_nr) + '' + self.it_str
                      + ' || Current loop progress 00.00%', end='')

            # ==== call the solver
            sol = solve_ivp(
                self.fex, [self.tstart, self.tend], yin, method='LSODA',
                rtol=self.ode_rtol, atol=self.ode_atol, t_eval=ts
            )
            self.all_runs.append(sol)
            self.rg_history[:, t]

            # ==== prepare next run
            yin = sol.y[:, -1]  # set initial conditions to last run
            # calculate acutal radius from output
            rhop = get_rhop(self, yin.reshape((len(self.species), self.sz)))  # mixed cloud particle dnesity [g/cm3]
            rg = mass_to_radius(self, sol.y[self.sz * 2:, -1],
                                sol.y[self.sz:self.sz * 2, -1], rhop)
            # find out if there are enough data points for full polynomial degree
            deg_fit = self.rg_fit_deg
            if sum(self.mask_psupsat) - 1 < self.rg_fit_deg:
                # This results in a preciese fit, but keep minimum of 1 degree
                deg_fit = np.maximum(1, sum(self.mask_psupsat) - 1)
                if deg_warn_flag:
                    print()
                    print('[WARN] Not enough data points, degree of radius fit '
                          'chagned to: ' + str(deg_fit))
                    deg_warn_flag = False
            # create a polnom fit to the cloud particle radius to prevent sudden changes
            fit = np.polyval(np.polyfit(np.log10(self.pres[self.mask_psupsat]),
                                        np.log10(rg[self.mask_psupsat]),
                                        deg=deg_fit),
                             np.log10(self.pres))
            rg = 10 ** np.maximum(np.minimum(fit, 50), -50)  # prevent extreme values
            rg = 10 ** ((np.log10(rg) + np.log10(self.rg)) / 2)
            self.rg = np.maximum(rg, self.r_ccn)

            # ==== analytic cloud structure plot (a bit messy, not gonna lie)
            if self.do_plots:
                plot_full_structure(self, sol.y, str(t))

            # ==== stopping creterion
            if type == 'iterate' and t >= self.itterations:
                break
            # at least two itterations are needed for convergence
            if type == 'convergence' and t > 1:
                # calculate max mmr offset
                ynew = sol.y[:, -1]
                yold = self.all_runs[-2].y[:, -1]
                max_mmr = np.max(np.abs((ynew - yold)/ynew))
                # check if it is small enough
                if max_mmr < rel_dif_in_mmr:
                    break
                # break if maximum number of itterations has been reached
                if t >= self.itterations:
                    print('[WARN] Maximum itterations reached with '
                          'precision: ' + str(max_mmr))
                    break


            # ==== incremment itterations and start new loop
            t += 1

    # ==== Calculate rg on the fly
    else:
        # ==== preparations
        self.rg_history = self.rg  # remember input radius
        self.loop_nr = 1  # there is only 1 loop
        self.rg_in = self.rg  # remember input radius for plotting
        ts = np.logspace(np.log10(self.tstart), np.log10(self.tend), self.tsteps)

        # ==== call the solver
        sol = solve_ivp(
            self.fex, [self.tstart, self.tend], yin, method='LSODA',
            rtol=self.ode_rtol, atol=self.ode_atol, t_eval=ts
        )

        # ==== analytic cloud structure plot (a bit messy, not gonna lie)
        if self.do_plots:
            plot_full_structure(self, sol.y, str(0))

    # ==== Finish up ====================================================================
    # ==== print final informations
    print(f'\r[INFO] Cloud structures completed in {time() - start_time:.2f}s ({self.loop_nr} iterations).')
    # ==== save data internally
    ds = save_run(self, sol, save_file=save_file, tag=tag)

    return ds