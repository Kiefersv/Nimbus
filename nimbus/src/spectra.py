""" Functions to calculate spectra of the resutls """
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import pandas as pd

def set_up_spectra_calculation(self, mass_planet, radius_planet, temperature_star,
                               radius_star, metalicity_star, logg_star,
                               path_to_opacities=None):
    """
    Set up calculation for spectra

    Parameters
    ----------
    mass_planet : float
        Mass of the planet in Jupiter mass
    radius_planet : float
        Radius of the planet in Jupiter radius
    temperature_star : float
        Temperature of the star in K
    radius_star : float
        Radius of the star in Solar radius
    metalicity_star : float
        Metallicity of the star in solar metalicity
    logg_star : float
        Gravity of the star in cgs
    path_to_opacities : str, optional
        Path to opacities files. Default takes the Nimbus opacities.
    """

    # ==== Set variables
    self.mass_planet = mass_planet
    self.radius_planet = radius_planet
    self.temperature_star = temperature_star
    self.radius_star = radius_star
    self.metalicity_star = metalicity_star
    self.logg_star = logg_star

    # ==== Set optional inputs
    if path_to_opacities is None:
        self.dir_opac = os.path.dirname(__file__) + '/../data/opacities/'
    else:
        self.dir_opac = path_to_opacities + '/'

    # ==== Confrim that setup is done
    self.isset_transmission_spectrum = True
    self.isset_emission_spectrum = True
    print('[INFO] Spectra calculation set up.')

def plot_spectrum(self, chem_data=None, data=None, chemical_abbundences=None, type='transmission', cloud_fraction=1, tag='last_run', plot=True):

    # ==== check if spectrum can be calculated
    if not self.isset_transmission_spectrum and not self.isset_emission_spectrum:
        raise ValueError("[ERROR] set_up_spectra_calculation needs to be run before plot_spectrum")

    # ==== picaso is only imported within this function so Nimbus can be run without it
    import picaso.justdoit as jdip
    import virga.justdoit as jdi

    # ==== get data from tag
    ds = self.results[tag]

    # ==== Get cloud particle opacities
    qext, qscat, cos_qscat, nwave, radius, wave_in = jdi.get_mie(self.specie, self.dir_opac)
    qc = np.asarray([ds['qc'].values]).T
    ndz = np.asarray([-ds['qn'] / self.m_ccn * self.dz * self.rhoatmo]).T
    rgin = np.asarray([ds['rg'].values]).T
    ndz[~self.mask_psupsat] = 0
    qext = qext[:, :, np.newaxis]
    qscat = qscat[:, :, np.newaxis]
    cos_qscat = cos_qscat[:, :, np.newaxis]
    nradii = len(radius)
    rmin = np.min(radius)
    rmax = np.max(radius)

    radius, rup, dr = jdi.get_r_grid_w_max(rmin, rmax, n_radii=nradii)
    opd, w0, g0, opd_gas = jdi.calc_optics(nwave, qc, qc, rgin, rgin, ndz, radius,
                                           dr, qext, qscat, cos_qscat, 2, rmin, nradii)

    df_cloud = jdi.picaso_format(
        opd[:-1], w0[:-1], g0[:-1],
        pressure=10 ** self.logp_mid * 1e-6, wavenumber=1 / wave_in[:, 0] / 1e-4,
    )

    opa = jdip.opannection(wave_range=[1, 15])
    case1 = jdip.inputs()
    case1.phase_angle(0)
    case1.gravity(mass=self.mass_planet, mass_unit=u.Unit('M_jup'),
                  radius=self.radius_planet, radius_unit=u.Unit('R_jup'))
    case1.star(opa, self.temperature_star, self.metalicity_star, self.logg_star,
               radius=self.radius_star, radius_unit=jdip.u.Unit('R_sun'))

    # ==== Prepare atmosphere for picaso input
    d = {
        'pressure': self.pres * 1e-6,
        'temperature': self.temp,
    }
    for key in chem_data:
        d[key] = chem_data[key]
    df = pd.DataFrame(data=d)
    case1.atmosphere(df=df)
    case1.clouds(df=df_cloud)

    # ==== Calculate transmission and emission spectra
    t_df = case1.spectrum(opa, full_output=True, calculation=type)

    # ==== Regrid output
    if type == 'transmission':
        trans_wavn, trans_rprs2 = t_df['wavenumber'], t_df['transit_depth']
        trans_wavn_bins, trans_wavn_rprs2 = jdip.mean_regrid(trans_wavn, trans_rprs2, R=100)
        wvl = 1 / trans_wavn_bins * 1e4
        spectrum = trans_wavn_rprs2
    elif type == 'thermal':
        emis_wavn, _, emis_fp = t_df['wavenumber'], t_df['fpfs_thermal'], t_df['thermal']
        emis_wavn_bin, emis_fp_bin = jdip.mean_regrid(emis_wavn, emis_fp, R=200)
        wvl = 1 / emis_wavn_bin * 1e4
        spectrum = emis_fp_bin


    if plot:
        if type == 'transmission':
            plt.figure()
            offset = 0
            if data is not None:
                mask = (wvl > data[0, 0]) * (wvl < data[0, -1])
                offset =  np.mean(data[1]) / np.mean(spectrum[mask])
                plt.errorbar(data[0], data[1], yerr=data[2], fmt='.', color='black', alpha=0.5, linewidth=0.7)
            plt.plot(wvl, spectrum * offset)
            plt.xscale('log')
            plt.savefig(self.working_dir + '/transmission_spectrum.png')
        elif type == 'thermal':
            pass

    return wvl, spectrum