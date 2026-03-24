""" Functions to sotre and load data """
import xarray as xr
import numpy as np

from .atmosphere_physics import mass_to_radius

def save_run(self, sol, save_file=None, tag=None):
    """
    Save the run as xarray

    Parameters
    ----------
    self : Nimbus object
    sol : solve_ivp object
        Solution of the run that should be saved.
    save_file : str
        Name of file to save to.
    tag : str
        Internal tag to remember run.

    Return
    ------
    ds : xarray.Dataset
        Xarray dataset containing the run

    """
    if self.static_rg:
        # ==== set up dataset stuff
        coordinates={
            'iteration': range(len(self.all_runs)),
            'pressure': self.pres * 1e-6,
            'species': self.species,
        }
        co = ['species', 'iteration', 'pressure']
        co2 = ['species', 'pressure']

        # ==== initialise variables
        gas_mmr = np.zeros((self.nspec, len(self.all_runs), self.sz))
        solid_mmr = np.zeros((self.nspec, len(self.all_runs), self.sz))
        nuc_rate = np.zeros((self.nspec, len(self.all_runs), self.sz))
        acc_rate = np.zeros((self.nspec, len(self.all_runs), self.sz))
        n1 = np.zeros((self.nspec, len(self.all_runs), self.sz))
        total_mmr = np.zeros((len(self.all_runs), self.sz))
        ncl = np.zeros((len(self.all_runs), self.sz))
        rg = np.zeros((len(self.all_runs), self.sz))
        for r, run in enumerate(self.all_runs):
            xrun = run.y[:, -1].reshape((self.nspec*2 + 1, self.sz))
            # calculate the physics
            xrun[xrun < self.ode_minimum_mmr] = self.ode_minimum_mmr
            total_mmr[r] = np.sum(xrun[1::2], axis=0)
            xn = xrun[-1]  # cloud number density mmr
            ncl[r] = xn * self.rhoatmo / self.m_ccn  # cloud particle number density [1/cm3]
            rg[r] = self.rg_history[r]
            for s, _ in enumerate(self.species):
                # gas-phase number density [1/cm3]
                n1[s, r] = xrun[s*2] * self.rhoatmo / self.m1[s]
                # accretion rate [1/cm3/s]
                acc_rate[s, r] = self.acc_rate(rg[r], self.temp, n1[s, r], ncl[r], s)
                # nucleation rate [1/cm3/s]
                nuc_rate[s, r] = self.nuc_rate(n1[s, r], self.temp, s)
                gas_mmr[s, r] = xrun[s*2]
                solid_mmr[s, r] = xrun[s*2 + 1]
        data = {
            'gas_mmr': (co2, gas_mmr[:, -1]),
            'cloud_mmr': (co2, solid_mmr[:, -1]),
            'nucleation_rate': (co2, nuc_rate[:, -1]),
            'growth_rate': (co2, acc_rate[:, -1]),
            'cloud_number_density': (co2[1:], ncl[-1]),
            'gas_number_density': (co2, gas_mmr[:, -1]),
            'cloud_radius': (co2[-1], rg[-1]),
            'temperature': (co2[-1], self.temp),
            'rhoatmo': (co2[-1], self.rhoatmo),
            'Kzz': (co2[-1], self.kzz),
            'all_gas_mmr': (co, gas_mmr),
            'all_cloud_mmr': (co, solid_mmr),
            'all_nucleation_rate': (co, nuc_rate),
            'all_growth_rate': (co, acc_rate),
            'all_cloud_number_density': (co[1:], ncl),
            'all_gas_number_density': (co, gas_mmr),
            'all_cloud_radius': (co[1:], rg),
            'all_temperature': (co[2:], self.temp),
            'all_rhoatmo': (co[2:], self.rhoatmo),
            'all_Kzz': (co[2:], self.kzz),
        }
    else:
        # ==== set up dataset stuff
        coordinates={
            'time': self.evaltimes,
            'pressure': self.pres * 1e-6,
            'species': self.species,
        }
        co = ['species', 'time', 'pressure']
        co2 = ['species', 'pressure']

        # ==== initialise variables
        gas_mmr = np.zeros((self.nspec, self.tsteps, self.sz))
        solid_mmr = np.zeros((self.nspec, self.tsteps, self.sz))
        nuc_rate = np.zeros((self.nspec, self.tsteps, self.sz))
        acc_rate = np.zeros((self.nspec, self.tsteps, self.sz))
        n1 = np.zeros((self.nspec, self.tsteps, self.sz))
        total_mmr = np.zeros((self.tsteps, self.sz))
        ncl = np.zeros((self.tsteps, self.sz))
        rg = np.zeros((self.tsteps, self.sz))
        for t, _ in enumerate(self.evaltimes):
            xrun = sol.y[:, t].reshape((self.nspec*2 + 1, self.sz))
            # calculate the physics
            xrun[xrun < self.ode_minimum_mmr] = self.ode_minimum_mmr
            xtot = np.sum(xrun[1::2], axis=0)
            total_mmr[t] = xtot
            rhotot = np.sum(xrun[1::2]*self.rhop[:, np.newaxis], axis=0)/xtot
            xn = xrun[-1]  # cloud number density mmr
            ncl[t] = xn * self.rhoatmo / self.m_ccn  # cloud particle number density [1/cm3]
            rg[t] = mass_to_radius(self, xrun[-1], xtot, rhotot)
            for s, _ in enumerate(self.species):
                n1[s, t] = xrun[s*2] * self.rhoatmo / self.m1[s]  # gas-phase number density [1/cm3]
                # assign the values
                acc_rate[s, t] = self.acc_rate(rg[t], self.temp, n1[s, t], ncl[t], s)  # accretion rate [1/cm3/s]
                nuc_rate[s, t] = self.nuc_rate(n1[s, t], self.temp, s)  # nucleation rate [1/cm3/s]
                gas_mmr[s, t] = xrun[s*2]
                solid_mmr[s, t] = xrun[s*2 + 1]
        data = {
            'gas_mmr': (co2, gas_mmr[:, -1]),
            'total_cloud_mmr': (co2[1:], total_mmr[-1]),
            'cloud_mmr': (co2, solid_mmr[:, -1]),
            'nucleation_rate': (co2, nuc_rate[:, -1]),
            'growth_rate': (co2, acc_rate[:, -1]),
            'cloud_number_density': (co2[1:], ncl[-1]),
            'gas_number_density': (co2, gas_mmr[:, -1]),
            'cloud_radius': (co2[-1], rg[-1]),
            'temperature': (co2[1:], self.temp),
            'rhoatmo': (co2[1:], self.rhoatmo),
            'Kzz': (co2[1:], self.kzz),
            'all_gas_mmr': (co, gas_mmr),
            'all_total_cloud_mmr': (co[1:], total_mmr),
            'all_cloud_mmr': (co, solid_mmr),
            'all_nucleation_rate': (co, nuc_rate),
            'all_growth_rate': (co, acc_rate),
            'all_cloud_number_density': (co[1:], ncl),
            'all_gas_number_density': (co, gas_mmr),
            'all_cloud_radius': (co[1:], rg),
            'all_temperature': (co[2:], self.temp),
            'all_rhoatmo': (co[2:], self.rhoatmo),
            'all_Kzz': (co[2:], self.kzz),
        }

    # ==== How data is stored
    ds = xr.Dataset(
        data_vars=data,
        coords=coordinates,
        attrs={
            'mmw': self.mmw,
            'total_iterations': self.loop_nr,
            'tstart': self.tstart,
            'tend': self.tend,
            'tsteps': self.tsteps,
            'ode_rtol': self.ode_rtol,
            'ode_atol': self.ode_atol,
            'ode_minimum_mmr': self.ode_minimum_mmr,
            'static_rg': int(self.static_rg),
            'r_ccn': self.r_ccn ,
            'cs_mol': self.cs_mol,
            'eps_k': self.eps_k,
            'rg_fit_deg': self.rg_fit_deg,
        },
    )

    # ==== store data in Nimbus class
    # define the tag
    if tag is None:
        tag = 'last_run'
    self.results[tag] = ds

    # ==== Print info
    if not self.mute:
        print('[INFO] Saved run under tag: ' + tag)

    # ==== save data to file if a save file is given
    if not isinstance(save_file, type(None)):
        ds.to_netcdf(save_file + '.nc')
        if not self.mute:
            print('       -> File name: ' + save_file)

    return ds


def load_previous_run(self, file_name, tag=None):
    """
    Load previously saved Nimbus runs.

    Parameters
    ----------
    self : Nimbus class
        current nimbus object.
    file_name : str
        Name of the file to load from working directory.
    tag : str, optional
        Name to store data in Nimbus.
    """

    # load file
    ds = xr.open_dataset(file_name)

    # if no tag is given use file name
    if tag is None:
        tag = file_name.split('.')[0]

    # load results into Nimbus
    self.results[tag] = ds

    # ==== Print info
    print('[INFO] Loaded previous run with tag: ' + tag)
    print('       -> File name: ' + file_name)

    # return results
    return ds
