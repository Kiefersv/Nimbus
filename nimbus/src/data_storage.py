""" Functions to sotre and load data """
import xarray as xr
import numpy as np

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
    # ==== calcualte dependent variables
    nc = sol.y[self.sz * 2:self.sz * 3, -1] * self.rhoatmo / self.m_ccn
    n1 = sol.y[:self.sz, -1] * self.rhoatmo / self.m1
    nuc_rate = np.nan_to_num(self.nuc_rate(n1, self.temp))
    growth_rate = self.acc_rate(self.rg, self.temp, n1, nc)

    # ==== How data is stored
    ds = xr.Dataset(
        data_vars={
            'qv': (['pressure'], sol.y[:self.sz, -1]),
            'qc': (['pressure'], sol.y[self.sz:self.sz * 2, -1]),
            'qn': (['pressure'], sol.y[self.sz * 2:self.sz * 3, -1]),
            'rg': (['pressure'], self.rg),
            'nc': (['pressure'], nc),
            'n1': (['pressure'], n1),
            'J': (['pressure'], nuc_rate),
            'G': (['pressure'], growth_rate),
            'rho_atmo': (['pressure'], self.rhoatmo),
            'temperature': (['pressure'], self.temp),
            'kzz': (['pressure'], self.kzz),
            'full_y': (['pressurex3', 'evaltimes'], sol.y),
        },
        coords={
            'pressure': self.pres * 1e-6,
            'evaltimes': self.evaltimes,
            'pressurex3': np.append(np.append(self.pres, self.pres), self.pres),
        },
        attrs={
            'mmw': self.mmw,
            'y': sol.y[:, -1],
            'itterations': self.loop_nr,
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
