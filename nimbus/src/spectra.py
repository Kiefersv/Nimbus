""" Functions to calculate spectra of the resutls """
# pylint: disable=C0415

import os
import numpy as np
import pandas as pd

def picaso_formater(self, tag=None, ds_prev=None, path_to_opacities=None, sig=2,
                    mie_type='full', nradii=100, mieai_object=None, time_stamps=None):
    """
    Create pandas dataframe of opacities that can be read into PICASO. This function is
    a simplified version of VIRGA code. Please cite Batalha et al. (2026) if you use it.

    Parameters
    ----------
    tag : str, optional
        Tag of the model to be calculated. Default is the last run. If neither tag nor
        ds_prev are given, the 'last_run' is loaded.
    ds_prev : xarray.Dataset, optional
        Dataset of previous run. If neither tag nor ds_prev are given, the 'last_run'
        is loaded.
    path_to_opacities : str, optional
        Path to the folder containing opacity files. If none, default files are used.
    sig : float
        Standard deviation for lognormal size radius distribution.
    mie_type : str, optional
        Can be 'full' for normal calculations, 'grid' for grid interpolation, or 'ai' for
        an Ai model. Grids you need to produce yourself using MieAi, and only certain
        species are available within Ai models.
    nradii : int, optional
        Number of radius bins, 100 or more is recommended.
    mieai_object : MieAi class, optional
        Giving a mieai class rather than creating the default one. Allows for more
        felxibility and mutliprocessing.
    time_stamps : list, optional
        If none, only final solution is returned, otherwise, a snapshot of each given
        time is returned.

    Return
    ------
    df_cloud : pd.dataframe
        Opacities in PICASO format
    """

    # ===================================================================================
    # Initialisation
    # ===================================================================================

    # ==== import mieai only here so Nimbus can be run without it
    if mieai_object is not None:
        ma = mieai_object
    else:
        from mieai import Mieai
        ma = Mieai(use_ai=False)  # set up mieai class

    # ==== Set optional inputs
    if path_to_opacities is None:
        self.dir_opac = os.path.dirname(__file__) + '/../data/opacities/'
    else:
        self.dir_opac = path_to_opacities + '/'

    # ==== get data from tag
    if tag is not None:
        ds = self.results[tag]
    elif ds_prev is not None:
        ds = ds_prev
    else:
        ds = self.results['last_run']

    # === wavelength grid (fixed to picaso 193 format)
    wave_in = np.asarray([
        0.268,   0.287,   0.312,   0.337,   0.362,   0.387,   0.408,   0.426,   0.445,
        0.465,   0.485,   0.506,   0.523,   0.537,   0.552,   0.565,   0.578,   0.591,
        0.604,   0.617,   0.628,   0.642,   0.657,   0.677,   0.697,   0.714,   0.729,
        0.739,   0.752,   0.765,   0.777,   0.792,   0.81 ,   0.822,   0.834,   0.852,
        0.865,   0.877,   0.891,   0.904,   0.919,   0.931,   0.944,   0.963,   0.980,
        0.996,   1.016,   1.040,   1.061,   1.080,   1.102,   1.126,   1.148,   1.169,
        1.189,   1.206,   1.224,   1.241,   1.259,   1.276,   1.296,   1.318,   1.339,
        1.358,   1.374,   1.386,   1.400,   1.416,   1.434,   1.452,   1.471,   1.488,
        1.506,   1.523,   1.538,   1.552,   1.567,   1.584,   1.604,   1.627,   1.646,
        1.660,   1.675,   1.692,   1.713,   1.740,   1.766,   1.788,   1.812,   1.837,
        1.863,   1.889,   1.916,   1.947,   1.982,   2.010,   2.030,   2.050,   2.070,
        2.091,   2.114,   2.129,   2.137,   2.148,   2.163,   2.183,   2.209,   2.240,
        2.276,   2.314,   2.354,   2.394,   2.437,   2.482,   2.527,   2.574,   2.623,
        2.676,   2.730,   2.787,   2.847,   2.909,   2.965,   3.014,   3.055,   3.085,
        3.118,   3.156,   3.193,   3.231,   3.269,   3.308,   3.363,   3.438,   3.513,
        3.587,   3.662,   3.749,   3.849,   3.949,   4.074,   4.202,   4.301,   4.395,
        4.504,   4.605,   4.662,   4.700,   4.75 ,   4.796,   4.827,   4.882,   5.000,
        5.149,   5.265,   5.379,   5.572,   5.759,   5.899,   6.064,   6.297,   6.515,
        6.664,   6.873,   7.094,   7.284,   7.498,   7.714,   7.904,   8.074,   8.345,
        8.673,   8.995,   9.348,   9.648,  10.091,  10.692,  11.292,  11.986,  12.787,
        13.588, 14.575,  15.777,  16.978,  18.18 ,  19.382,  21.395,  24.839,  29.700,
        35.753, 42.212,  53.879,  73.692,  97.752, 138.793, 227.531
    ])
    nw = len(wave_in)

    # ==== check if timestamps are given
    if time_stamps is None:
        time_it = [None]
    else:
        time_it = np.asarray(time_stamps)
    nt = len(time_it)  # number of time steps

    # ==== Get cloud structure
    qc = np.asarray(ds['cloud_mmr'].values).T
    ndz = np.asarray(-ds['cloud_number_density'].values * self.dz).T
    rg = np.asarray(ds['cloud_radius'].values).T
    ndz[~self.mask_psupsat] = 0  # all values below cloud deck are zero
    # pressure of the output
    p_out = [[i] * len(wave_in) for i in np.exp(self.logp_mid) * 1e-6]
    # wavelength of the output
    w_out = [1 / wave_in / 1e-4] * len(self.logp_mid)

    # ==== radius grid
    rmin = 1e-8
    rmax = 0.1
    radius = np.logspace(np.log10(rmin),np.log10(rmax), nradii)
    rat = radius[1]/radius[0]
    rup = 2*rat / (rat+1) * radius
    dr = np.zeros(rup.shape)
    dr[1:] = rup[1:]-rup[:-1]
    dr[0] = dr[1]**2/dr[2]

    # ==== general variables
    nz = qc.shape[0]  # number of atmospheric lauyers
    ngas = qc.shape[1]  # number of gas-phase species
    nrad = len(radius)  # number of cloud particle radii

    # ==== working and output arrays
    scat_gas = np.zeros((nz, nw))  # working array scattering coefficient
    ext_gas = np.zeros((nz, nw))  # working array extinction coefficient
    cqs_gas = np.zeros((nz, nw))  # working array asymmetry coefficient
    opd = np.zeros((nz, nw))  # total optical depth
    w0 = np.zeros((nz, nw))  # single scattering albedo
    g0 = np.zeros((nz, nw))  # asymmetry parameter
    vmr_test = np.zeros((2, nrad, ngas))  # check if vmr changes between pressures
    df = []  # output list

    # ===================================================================================
    # Data readout
    # ===================================================================================

    # ==== iterate over all timestamps (or only the last one)
    for t, ti in enumerate(time_it):

        # ==== get data of current timestamp
        if ti is not None:
            ds_t = ds.interp(time=ti)
            qc = np.asarray(ds_t['all_cloud_mmr'].values).T
            ndz = np.asarray(-ds_t['all_cloud_number_density'].values * self.dz).T
            rg = np.asarray(ds_t['all_cloud_radius'].values).T
            ndz[~self.mask_psupsat] = 0  # all values below cloud deck are zero

        # ===============================================================================
        # Mixed opacity precalculation
        # ===============================================================================

        # ==== calculate volume fractions from mass mixing ratio
        vol = qc / self.rhop[np.newaxis,]
        vf = vol / np.sum(vol, axis=1)[:, np.newaxis]
        vf[vf < 1e-50] = 1e-50

        # ==== loop over all height layers
        for iz in range(nz):
            if ndz[iz] <= 0:
                continue

            # calculate size distributions of each material
            arg1 = dr / (np.sqrt(2. * np.pi) * radius * np.log(sig))
            arg2 = -np.log(radius / rg[iz]) ** 2 / (2 * np.log(sig) ** 2)
            dist = arg1 * np.exp(arg2) / np.sum(arg1 * np.exp(arg2))
            ndr_mixed = ndz[iz] * dist

            # set volume mixing ratios
            vmr = {}
            for g, gas in enumerate(self.species):
                vmr[gas] = np.ones((nrad,)) * vf[iz, g]
                vmr_test[0, :, g] = vmr[gas]

            # check if vmrs have changed, and only then re-calculate opaciteis
            if np.any(np.abs((vmr_test[0] - vmr_test[1]) / vmr_test[0]) > 1e-4):
                if mie_type == 'full':
                    qet, qst, cqt = ma.efficiencies(wave_in, radius * 1e4, vmr)
                elif mie_type == 'grid':
                    qet, qst, cqt = ma.grid_efficiencies(wave_in, radius * 1e4, vmr)
                cqt = qet * cqt
                # remeber the vmrs for the next run
                vmr_test[1] = vmr_test[0]

            # total geometric cross-section
            pir2ndz = np.pi * radius ** 2 * ndr_mixed

            # opacity of mixed particles
            scat_gas[iz] = np.sum(qst.T * pir2ndz[np.newaxis], axis=1)
            ext_gas[iz] = np.sum(qet.T * pir2ndz[np.newaxis], axis=1)
            cqs_gas[iz] = np.sum(cqt.T * pir2ndz[np.newaxis], axis=1)

        # ===============================================================================
        #  Prepare output
        # ===============================================================================

        # ==== Sublayering to prevent sharp boundaries
        for iz in range(nz - 1, -1, -1):
            if np.sum(ext_gas[iz, :]) > 0:
                ibot = iz
                break
            if iz == 0:
                ibot = 0
        if ibot >= nz - 3:
            print("Not doing sublayer as cloud deck at the bottom of pressure grid")
        else:
            for arr in [scat_gas, ext_gas, cqs_gas]:
                arr[ibot + 1, :] = arr[ibot, :] * 0.1
                arr[ibot + 2, :] = arr[ibot, :] * 0.05
                arr[ibot + 3, :] = arr[ibot, :] * 0.01

        # ==== Sum over gases and compute spectral optical depth profile etc
        mask = scat_gas > 0
        opd[mask] = ext_gas[mask]
        w0[mask] = scat_gas[mask] / ext_gas[mask]
        g0[mask] = cqs_gas[mask] / scat_gas[mask]

        # ==== Create opacities in picaso format
        df_t = pd.DataFrame(
            dict(opd=opd[:-1].flatten(),
                 w0=w0[:-1].flatten(),
                 g0=g0[:-1].flatten()))
        df_t['pressure'] = np.concatenate(p_out)
        df_t['wavenumber'] = np.concatenate(w_out)
        df.append(df_t)

    # if no timestamps are given, return only one dataframe
    if time_stamps is None:
        return df[0]
    # otherwise return a list of dataframes
    return df

def virga_opacities(self, tag=None, ds_prev=None, path_to_opacities=None, sig=2):
    """
    This routine uses Virga functions to calculate opacities. It does not support more
    than 1 cloud material.
    """

    # ==== Import Virga only here to allow nimbus to be run without it installed
    import virga.justdoit as jdi

    # ==== get data from tag
    if tag is not None:
        ds = self.results[tag]
    elif ds_prev is not None:
        ds = ds_prev
    else:
        ds = self.results['last_run']

    # ==== Set optional inputs
    if path_to_opacities is None:
        # if no path is given, assume that the opacities are within the data folder
        dir_opac = os.path.dirname(__file__) + '/../data/opacities/'
    else:
        dir_opac = path_to_opacities + '/'

    # ==== check that only 1 species is in the dataset
    if len(ds.attrs['species']) > 1:
        raise ValueError("Virga does not support more than one cloud species. Please "
                         "use the picaso_formater function instead.")

    # ==== Get cloud particle opaciteis from precalculated file
    qext, qscat, cqscat, nwave, radius, wave_in = jdi.get_mie(ds.attrs['species'][0], dir_opac)

    # ==== Use Virga routines to calculate opacities
    radii, _, dr = jdi.get_r_grid_w_max(np.min(radius), np.max(radius), len(radius))
    opd, w0, g0, _ = jdi.calc_optics(
    #     nwave, ds['cloud_mmr'].T, ds['cloud_radius'].T, ds['cloud_number_density'].T, radii,
    #     dr, qext, qscat, cqscat, sig, np.min(radius), np.max(radius), wavelength=None, gas_name=None, mixed=False
    # )
        nwave, ds['cloud_mmr'].values.T, ds['cloud_mmr'].values.T, ds['cloud_radius'].values[:, np.newaxis], ds['cloud_radius'].values[:, np.newaxis],
        ds['cloud_number_density'].values[:, np.newaxis] * (-self.dz), radii, dr, qext[:, :, np.newaxis], qscat[:, :, np.newaxis], cqscat[:, :, np.newaxis], sig, np.min(radius), len(radius)
    )

    # ==== Use Picaso Formater to produce pandas file
    df = pd.DataFrame(
        dict(opd=opd[:-1].flatten(),
             w0=w0[:-1].flatten(),
             g0=g0[:-1].flatten()))
    df['pressure'] = np.concatenate([[i] * len(wave_in[:, 0]) for i in np.exp(self.logp_mid) * 1e-6])
    df['wavenumber'] = np.concatenate([1 / wave_in[:, 0] / 1e-4] * len(self.logp_mid))

    # ==== Return all opacity information
    return opd, w0, g0, df
